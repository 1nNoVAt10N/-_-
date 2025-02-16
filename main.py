from train import train_val_test
from model import ViTModel,BFPCNet,BiometricModel,simple_resnet
import torch

from data_utils import EyeDataset
from train import FocalLoss,focalLoss
import numpy as np
from vit_pytorch import ViT as V
from main_model import HiFuse_Small as create_model

if __name__ == "__main__":
    #model = ViTModel(num_classes=8)
    model = simple_resnet
    critrion = FocalLoss(alpha=0.75, gamma=2, reduction='mean')
    criterion= torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    train_data = EyeDataset(cache_dir="./cache_384",file_list="./train_images.txt",augment=True,augment_times=1)
    
    val_data = EyeDataset(cache_dir="./cache_384",file_list="./test_images.txt")
    #test_data = EyeDataset(cache_dir="./cache_384",file_list="./test_images.txt")

    genshin = train_val_test(epoch=200,lr=1e-4,batch_size=4,num_workers=4,device="cuda",model=model,opitimizer=optimizer,criterion=critrion)
    genshin.train(train_data=train_data,val_data=val_data)

