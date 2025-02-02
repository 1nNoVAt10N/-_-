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

import cv2
import random
import numpy as np

def augment_image(img):
    img = img * 255.0
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # 随机左右翻转
    if random.random() > 0.6:
        img = cv2.flip(img, 1)

    # 随机旋转（优化填充方式）
    if random.random() > 0.7:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)


    if random.random() > 0.6:
            scale = random.uniform(0.9, 1.1)
            try:
                scaled = cv2.resize(img, None, fx=scale, fy=scale, 
                                  interpolation=cv2.INTER_AREA)
                # 处理极端缩放情况
                if scaled.shape[0] == 0 or scaled.shape[1] == 0:
                    scaled = img.copy()
            except:
                scaled = img.copy()

            # 尺寸恢复逻辑
            if scale < 1:
                img = cv2.copyMakeBorder(scaled,
                                       top=(h-scaled.shape[0])//2,
                                       bottom=(h-scaled.shape[0]+1)//2,
                                       left=(w-scaled.shape[1])//2,
                                       right=(w-scaled.shape[1]+1)//2,
                                       borderType=cv2.BORDER_REPLICATE)
            else:
                start_y = max(0, (scaled.shape[0]-h)//2)
                start_x = max(0, (scaled.shape[1]-w)//2)
                img = scaled[start_y:start_y+h, start_x:start_x+w]


    # 空间增强 --------------------------------------------------------------------
    # 随机裁剪（保持尺寸）
    if random.random() > 0.5:
        crop_scale = random.uniform(0.85, 1.0)
        crop_h = int(h * crop_scale)
        crop_w = int(w * crop_scale)
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)
        img = img[y:y+crop_h, x:x+crop_w]
        img = cv2.resize(img, (w, h))

    # 随机平移
    if random.random() > 0.5:
        tx = random.randint(-int(w*0.1), int(w*0.1))
        ty = random.randint(-int(h*0.1), int(h*0.1))
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)

    # 高斯噪声
    if random.random() > 0.7:
        # 动态调整噪声强度
        max_val = img.max()
        safe_std = min(25, max_val/4)  # 根据图像亮度自动限制噪声强度
        std = random.uniform(1, safe_std)
        
        # 生成带保护机制的噪声
        noise = np.random.normal(0, std, img.shape)
        noise = np.clip(noise, -max_val*0.3, (255-max_val)*0.3)  # 动态截断
        
        # 安全添加噪声
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255)

    # 模糊/锐化增强 --------------------------------------------------------------
    # 高斯模糊
    if random.random() > 0.7:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # 运动模糊
    elif random.random() > 0.8:
        size = random.randint(5, 15)
        kernel = np.zeros((size, size))
        if random.random() > 0.5:
            kernel[:, size//2] = 1  # 垂直模糊
        else:
            kernel[size//2, :] = 1  # 水平模糊
        kernel /= size
        img = cv2.filter2D(img, -1, kernel)

    return img / 255.0

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
        

        for i in tqdm.tqdm(range(len(self.lists))):
            cache_path = self.lists[i].strip()
            data = np.load(cache_path)
            self.labels.append(data["label"])
        
        minority_indices = set()
        threshold = 0.1 * len(self.labels)  # 假设少数类阈值为总样本的10%

        self.labels = np.array(self.labels)

        for label_idx in range(self.labels.shape[1]):
    
            positive_count = np.sum(self.labels[:, label_idx])
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
                if i in minority_indices:
                    for _ in range(self.augment_times):
                        left_augmented_img = augment_image(left_img)
                        right_augmented_img = augment_image(right_img)

                        augmented_img = np.concatenate((left_augmented_img, right_augmented_img), axis=2)
                        self.img_data.append((augmented_img,label))
                else:
                    self.img_data.append((np.concatenate((left_img, right_img), axis=2),label))
            
            if not self.augment:
                self.img_data.append((np.concatenate((left_img, right_img), axis=2),label))
            

    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return len(self.img_data)

    def __getitem__(self, idx):
        data = self.img_data[idx]
        img = torch.tensor(data[0], dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(data[1], dtype=torch.float32)
        
        return img, label
    
if __name__ == "__main__":
    train_data = EyeDataset(cache_dir="./cache_384",file_list="./train_images.txt",augment=True,augment_times=5)
    print(len(train_data))
