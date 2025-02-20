import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights,resnet101,ResNet101_Weights
from vit_model import ViT
from vit_pytorch import ViT as V
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet




class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ResNet基础残差结构
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)  # 添加ReLU函数
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        # 通道注意力（SENet风格）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 4, (out_channels * 4) // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d((out_channels * 4) // reduction, out_channels * 4, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力（CBAM风格）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 下采样层（如果输入输出维度不匹配）
        if in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        # 残差路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # 通道注意力
        channel_att = self.channel_att(out)
        out = out * channel_att

        # 空间注意力
        spatial_avg = torch.mean(out, dim=1, keepdim=True)
        spatial_max, _ = torch.max(out, dim=1, keepdim=True)
        spatial_cat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_att = self.spatial_att(spatial_cat)
        out = out * spatial_att

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out


# ----------------------
# 3. 特征融合模块 (FFM)
# ----------------------
class FeatureFusion(nn.Module):
    def __init__(self, in_channels,num_classes=8):
        super().__init__()
        # 特征融合（输入通道数为in_channels*2，输出为in_channels）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )
        #self.ffa = FFA(inchannel=in_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的融合权重
    
        # 分类头（多标签分类使用Sigmoid）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_channels, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            #nn.Sigmoid()  # 多标签分类需要Sigmoid
        )
        self.relu = nn.ReLU(inplace=True)
        self.last = nn.Linear(512,8)

    def forward(self, left_feat, right_feat):
        # 特征拼接（通道维度）
        fused = torch.cat([left_feat, right_feat], dim=1)
        fused = self.conv(fused) * self.alpha
        fused = self.relu(fused)
        fused = nn.AdaptiveAvgPool2d(1)(fused)
        fused = nn.Flatten()(fused)
        #logits = self.classifier(fused)

        #logits = self.last(logits)
        return fused


class LabelAwareAttention(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.projection = nn.Linear(feat_dim, num_classes)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, features):
        # features: [B, D]
        # 生成标签注意力权重
        att = self.projection(features)  # [B, C]
        att = torch.sigmoid(att / self.temperature)
        return att.unsqueeze(-1) * features.unsqueeze(1)  # [B, C, D]
# ----------------------
# 4. 完整的 BFPC-Net
# ----------------------
class BFPCNet1(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # 修改的ResNet50主干（替换最后一个残差块为RAM）
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.backbone_front = nn.Sequential(
            *list(resnet.children())[:4],

        )
        self.backbone_layer1 = nn.Sequential(

            *list(resnet.children())[4],

        )
        self.backbone_layer2= nn.Sequential(

            *list(resnet.children())[5],  # 取到layer3（输出通道数1024）

        )


        self.backbone_layer3= nn.Sequential(
            #self.tksa,
            *list(resnet.children())[6:7],  # 取到layer3（输出通道数1024）
            ResidualAttentionBlock(in_channels=1024, out_channels=256)  # 替换layer4
        )

        # 特征融合模块 (FFM)
        self.ffm = FeatureFusion(in_channels=2048, num_classes=num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768*2  , 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.last = nn.Linear(512,8)
        #self.labelatt = LabelAwareAttention(num_classes=num_classes, feat_dim=2048)
        self.vit = ViT(image_size=224,name = 'B_16_imagenet1k',in_channels = 3,num_classes=8,pretrained=False) 
    def forward(self, x):
        # 图像增强
        x = x.to(torch.float32)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        left_aug = x[:,:3,:,:]
        right_aug = x[:,3:,:,:]   


        # 特征提取（输出形状：[batch_size, 1024, 14, 14]）
        left_feat = self.backbone_front(left_aug)
        right_feat = self.backbone_front(right_aug)
        left_feat = self.backbone_layer1(left_feat)
        right_feat = self.backbone_layer1(right_feat)
        left_feat = self.backbone_layer2(left_feat)
        right_feat = self.backbone_layer2(right_feat)
        left_feat = self.backbone_layer3(left_feat)
        right_feat = self.backbone_layer3(right_feat)
        l_vf = self.vit(left_aug)
        r_vf = self.vit(right_aug)
        vf = torch.cat([l_vf,r_vf],dim=1)

        # 特征融合与分类
        logits = self.ffm(left_feat, right_feat)
        logits = torch.cat([vf,logits],dim=1)
        logits = self.classifier(logits)
        logits = self.last(logits)
        return logits

if __name__ == "__main__":
    x = torch.randn(4,6,224,224)
    model = BFPCNet1(num_classes=8)
    print(model(x).shape)
