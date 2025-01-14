from train import train_val_test
from model import ResNetModel
import torch
from data_utils import EyeDataset

if __name__ == "__main__":
    model = ResNetModel(num_classes=8)
    critrion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_data = EyeDataset(cache_dir="./cache",file_list="./train_images.txt")
    val_data = EyeDataset(cache_dir="./cache",file_list="./val_images.txt")
    test_data = EyeDataset(cache_dir="./cache",file_list="./test_images.txt")

    genshin = train_val_test(epoch=50,lr=0.001,batch_size=32,num_workers=4,device="cuda",model=model,opitimizer=optimizer,criterion=critrion)
    genshin.train(train_data=train_data)
    genshin.val(val_data=val_data)
    genshin.test(test_data=test_data)
