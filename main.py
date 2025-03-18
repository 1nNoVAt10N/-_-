from train import train_val_test
import torch
import torch
from model_without_bert import BFPCNet1
from data_utils import EyeDataset
from train import FocalLoss
from model_without_bert import BFPCNet1 as hhh

import numpy as np
# from vit_pytorch import ViT as V
# from main_model import HiFuse_Small as create_model
import models_vit

if __name__ == "__main__":

    model = hhh(num_classes=8).to("cuda")
    critrion = FocalLoss(alpha=0.75,gamma=2)
    train_data = EyeDataset(cache_dir="./cache_384",file_list="./train_images11.txt",augment=False,augment_times=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    lr = 1e-4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.3,min_lr=1e-7,verbose=1,patience=3)
    val_data = EyeDataset(cache_dir="./cache_384",file_list="./test_images11.txt")


    genshin = train_val_test(epoch=15,lr=1e-4,batch_size=16,num_workers=4,device="cuda",model=model,optimizer=optimizer,criterion=critrion,scheduler=scheduler)
    genshin.train(train_data=train_data,val_data=val_data)
