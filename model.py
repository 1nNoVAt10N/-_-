import torch
import torch.nn as nn
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        # 调整输入层以适应双目图像（6 通道）
        self.resnet.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        # 替换最后的全连接层以适应多标签分类
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # 多标签任务采用 Sigmoid 激活
        )
    
    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    # 测试模型
    model = ResNetModel(num_classes=8)
    dummy_input = torch.randn(4, 6, 256, 256)  
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应输出 (4, 8)
