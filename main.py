from train import train_val_test
from model import DualResNet,ViTModel
import torch
from data_utils import EyeDataset
from train import FocalLoss
import numpy as np
from vit_pytorch import ViT as V


if __name__ == "__main__":
    #model = ViTModel(num_classes=8)
    model = ViTModel(num_classes=8)
    critrion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    train_data = EyeDataset(cache_dir="./cache_384",file_list="./train_images.txt",augment=True,augment_times=2)
    
    val_data = EyeDataset(cache_dir="./cache_384",file_list="./test_images.txt")
    test_data = EyeDataset(cache_dir="./cache_384",file_list="./test_images.txt")

    genshin = train_val_test(epoch=200,lr=0.00001,batch_size=8,num_workers=4,device="cuda",model=model,opitimizer=optimizer,criterion=critrion)
    genshin.train(train_data=train_data,val_data=val_data)
    genshin.test(test_data=test_data)
