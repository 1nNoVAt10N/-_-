import os
import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm
from transformers import BioGptTokenizer, BioGptForCausalLM,BioGptModel
# 数据增强函数
import cv2
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
mean = IMAGENET_DEFAULT_MEAN
std = IMAGENET_DEFAULT_STD

def get_augmentations():
    return A.Compose([
        #A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)),
        A.RandomBrightnessContrast(p=0.2),  # 颜色增强可不同步
        A.Normalize(mean, std)
    ],additional_targets={'right':'image'})
def get_augmentations2():
    return A.Compose([
        A.Normalize(mean, std),

    ],additional_targets={'right':'image'})

class EyeDataset(Dataset):
    def __init__(self, cache_dir, file_list, augment=False, augment_times=1):
        with open(file_list, "r") as f:
            self.lists = f.readlines()
        self.cache_dir = cache_dir
        self.file_list = sorted(self.lists)
        self.augment = augment  
        # 数据增强的次数
        self.augment_times = augment_times
        self.img_data = []
        self.text_data = []
        self.labels = []
        self.tranform = get_augmentations()
        self.tranform2 = get_augmentations2()
        self.tokenizer = AutoTokenizer.from_pretrained("/public/home/gjgao/users/waibao/BFPC/biobert_model/")
        self.model = AutoModel.from_pretrained("/public/home/gjgao/users/waibao/BFPC/biobert_model/")

        for i in tqdm.tqdm(range(len(self.lists))):
            cache_path = self.lists[i].strip()
            data = np.load(cache_path)
            self.labels.append(data["label"])
        
        minority_indices = set()
        threshold = 0.1 * len(self.labels)  # 假设少数类阈值为总样本的10%

        self.labels = np.array(self.labels)

        for label_idx in range(self.labels.shape[1]):
    
            positive_count = np.sum(self.labels[:, label_idx])
            print(positive_count)
            if positive_count < threshold:
                # 获取该标签为1的样本索引
                indices = np.where(self.labels[:, label_idx] == 1)[0]
                minority_indices.update(indices.tolist())

        minority_indices = list(minority_indices)

        for i in tqdm.tqdm(range(len(self.lists))):
            cache_path = self.lists[i].strip()
            data = np.load(cache_path)
            img = data["img"] 
            left_img = img[:, :3, :]
            right_img = img[:, 3:, :]
            label = data["label"]

            self.img_data.append((np.concatenate((left_img,right_img), axis=1),label))
            self.text_data.append(data["left_keywords"]+","+data["right_keywords"])
            

    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return len(self.img_data)

    def __getitem__(self, idx):
        if self.augment:
            data = self.img_data[idx]
            left_img = data[0][:, :224, :]
            right_img = data[0][:, 224:, :]
            au = self.tranform(image = left_img,right=right_img)
            img = np.concatenate((au['image'],au['right']), axis=1)
            img = torch.tensor(img, dtype=torch.float32).clone().detach().permute(2, 0, 1)
            label = torch.tensor(data[1], dtype=torch.float32)
            inputs = self.tokenizer(self.text_data[idx], return_tensors="pt")  # PyTorch 格式输入
            with torch.no_grad():  # 关闭梯度计算
                outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)
            cls_embedding = last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)
            text = cls_embedding.squeeze(0)
        else:
            data = self.img_data[idx]
            left_img = data[0][:, :224, :]
            right_img = data[0][:, 224:, :]

            au = self.tranform2(image = left_img,right=right_img)
            img = np.concatenate((au['image'],au['right']), axis=1)
            img = torch.tensor(img, dtype=torch.float32).clone().detach().permute(2, 0, 1)
            label = torch.tensor(data[1], dtype=torch.float32)
            inputs = self.tokenizer(self.text_data[idx], return_tensors="pt")  # PyTorch 格式输入
            with torch.no_grad():  
                outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)
            cls_embedding = last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)
            text = cls_embedding.squeeze(0)
       #print(img.shape)
        return img, text,label
if __name__ == "__main__":
    train_data = EyeDataset(cache_dir="./cache_384",file_list="./train_images.txt",augment=True,augment_times=5)
    print(len(train_data))
