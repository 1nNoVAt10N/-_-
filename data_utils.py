import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import tqdm
# 数据增强函数
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

import cv2
import random
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
mean = IMAGENET_DEFAULT_MEAN
std = IMAGENET_DEFAULT_STD

def get_augmentations():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # 亮度调整
        #A.RandomResizedCrop(height=224, width=224,p=0.5,scale=(0.8, 1.0),ratio=(0.75, 1.33),interpolation=1),
        A.Affine(translate_percent={"x": 0.1, "y": 0.1}, p=0.5),  # 平移
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2),  # 模拟病灶遮挡
        A.Normalize(mean, std)
    ])
def get_augmentations2():
    return A.Compose([
        A.Normalize(mean, std),

    ])
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
        self.labels = []
        self.tranform = get_augmentations()
        self.tranform2 = get_augmentations2()
        

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
            left_img = data["img"][:,:,:3] 
            right_img = data["img"][:,:,3:] 


            label = data["label"]

            if self.augment:
                self.img_data.append((np.concatenate((self.tranform2(image=left_img)['image'], self.tranform2(image=right_img)['image'] ), axis=2),label))
                #if i in minority_indices:
                for _ in range(self.augment_times):
                        left_augmented_img = self.tranform(image=left_img)['image']
                        right_augmented_img = self.tranform(image=right_img)['image']
                        left_augmented_img = torch.tensor(left_augmented_img)
                        right_augmented_img = torch.tensor(right_augmented_img)

                        augmented_img = np.concatenate((left_augmented_img, right_augmented_img), axis=2)
                        self.img_data.append((augmented_img,label))
                    
            
            if not self.augment:
                self.img_data.append((np.concatenate((self.tranform2(image=left_img)['image'], self.tranform2(image=right_img)['image'] ), axis=2),label))
            

    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return len(self.img_data)

    def __getitem__(self, idx):
        
        data = self.img_data[idx]
        img = torch.tensor(data[0], dtype=torch.float32).permute(2, 0, 1) 
        label = torch.tensor(data[1], dtype=torch.float32)
        
        return img, label
    
if __name__ == "__main__":
    train_data = EyeDataset(cache_dir="./cache_384",file_list="./train_images.txt",augment=True,augment_times=5)
    print(len(train_data))
