from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np

import os
import torch
from torch.utils.data import Dataset
import numpy as np

class EyeDataset(Dataset):
    def __init__(self, cache_dir, file_list):
        with open(file_list, "r") as f:
            self.lists = f.readlines()
        self.cache_dir = cache_dir
        self.file_list = sorted(self.lists)

    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return len(self.lists)

    def __getitem__(self, idx):

        # 获取文件名
        file_name = self.lists[idx]
        cache_path = file_name.strip()
        
        # 加载数据
        data = np.load(cache_path)
        img = torch.tensor(data["img"], dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(data["label"], dtype=torch.long)
        
        return img, label


