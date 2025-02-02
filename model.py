import torch
import torch.nn as nn
from torchvision import models
from TKSA import TKSA,TKESA
from BIE_BIEF import BIEF
from CVIM import CVIM
from vit_model import ViT
from vit_pytorch import ViT as V
from MAB import MAB
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        # 空间注意力
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # 通道注意力应用
        x = self.channel_attention(x) * x
        # 空间注意力应用
        x = self.spatial_attention(x) * x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享的全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        # 合并结果
        channel_weights = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return channel_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度上的最大和平均池化
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # 合并通道统计量
        combined = torch.cat([max_pool, avg_pool], dim=1)
        # 生成空间注意力图
        spatial_weights = self.sigmoid(self.conv(combined))
        return spatial_weights
    
class DualResNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # 骨干网络
        self.left_stream = models.resnet50(pretrained=False)
        self.right_stream = models.resnet50(pretrained=False)
        self.left_stream.fc = nn.Sequential(
            nn.Linear(self.left_stream.fc.in_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.right_stream.fc = nn.Sequential(
            nn.Linear(self.right_stream.fc.in_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.MAB_left = MAB(2048)
        self.MAB_right = MAB(2048)
        

        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(4096, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        left = self.left_stream(x[:, :3])
        right = self.right_stream(x[:, 3:])
        

        
        fused = torch.cat([left, right], dim=1)
        fused = self.fusion(fused).squeeze()
        return torch.sigmoid(self.classifier(fused))
class ViTModel(nn.Module):
    def __init__(self, num_classes=8):
        super(ViTModel, self).__init__()
        self.vit = ViT(image_size=384,name = 'B_16_imagenet1k',in_channels = 6,num_classes=8,pretrained=False) 
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256)

        )
        self.resnet.conv1 = nn.Conv2d(6,64,kernel_size=7)
        self.classifer = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,num_classes)
        )
    def forward(self, x):
        vit_fea = self.vit(x)
        resnet_fea = self.resnet(x)
        all_fea = torch.cat([vit_fea,resnet_fea],dim=1)
        classes = self.classifer(all_fea)
        return torch.sigmoid(classes)
    



    
if __name__ == "__main__":
    # 测试模型
    model = ViTModel(num_classes=8)
    dummy_input = torch.randn(16, 6, 384, 384)
    label = torch.randn(16,8)  
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应输出 (4, 8)
