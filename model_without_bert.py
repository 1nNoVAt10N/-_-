import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights,resnet101,ResNet101_Weights
from vit_model import ViT
import models_vit
from timm import create_model
class EnhancedSemanticAttentionModule(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(EnhancedSemanticAttentionModule, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads

        # 使用线性层匹配维度
        self.adjust_global_dim = nn.Linear(global_dim, local_dim)
        self.adjust_local_dim = nn.Linear(local_dim, global_dim)

        # Cross-Attention layers
        self.global_to_local_attention = nn.MultiheadAttention(local_dim, num_heads)
        self.local_to_global_attention = nn.MultiheadAttention(global_dim, num_heads)

        # Self-Attention layer for the concatenated features
        self.self_attention = nn.MultiheadAttention(global_dim + local_dim, num_heads)

        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(global_dim + local_dim)

    def forward(self, global_features, local_features):


        # 调整全局和局部特征的维度
        adjusted_global_features = self.adjust_global_dim(global_features)
        adjusted_local_features = self.adjust_local_dim(local_features)

        # Cross-attention operations
        global_to_local_attn, _ = self.global_to_local_attention(local_features, adjusted_global_features, adjusted_global_features)
        local_to_global_attn, _ = self.local_to_global_attention(global_features, adjusted_local_features, adjusted_local_features)

        # Concatenate the cross-attention outputs
        concatenated_features = torch.cat((global_to_local_attn, local_to_global_attn), dim=-1)

        # Self-attention to enhance the features further
        enhanced_features, _ = self.self_attention(concatenated_features, concatenated_features, concatenated_features)

        # Optional: Layer normalization
        enhanced_features = self.layer_norm(enhanced_features)


        return enhanced_features
    
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

    def forward(self, left_feat):
        # 特征拼接（通道维度）

        fused = self.conv(left_feat) * self.alpha
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
class GatedFusion(nn.Module):
    def __init__(self, text_dim, img_dim, hidden_dim):
        super().__init__()
        # 门控信号生成网络
        self.gate_net = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出[0,1]间的门控值
        )
        # 特征维度对齐（可选）
        self.text_proj = nn.Linear(text_dim, 512)
        self.img_proj = nn.Linear(img_dim, 512)

    def forward(self, text_feats, img_feats):
        # 对齐特征维度（假设最终统一为img_dim）
        text_proj = self.text_proj(text_feats)  # [batch, img_dim]
        img_proj = self.img_proj(img_feats)      # [batch, img_dim]
        
        # 拼接特征生成门控值
        combined = torch.cat([text_proj, img_proj], dim=1)  # [batch, text_dim + img_dim]
        gate = self.gate_net(combined)  # [batch, 1]
        
        # 加权融合
        fused_feats = gate * text_proj + (1 - gate) * img_proj
        return fused_feats
    
class BFPCNet1(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # 修改的ResNet50主干（替换最后一个残差块为RAM）

        resnet = resnet101(weights = ResNet101_Weights.IMAGENET1K_V2)
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
        self.ffm = FeatureFusion(in_channels=1024, num_classes=num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(1024+768 , 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        #self.classifier = KAN([2048+768*2, 1024, 512])
        self.last = nn.Linear(128,8)
        #self.gated = GatedFusion(768,1024+768,512)
        #self.labelatt = LabelAwareAttention(num_classes=num_classes, feat_dim=2048)
        self.l_att = EnhancedSemanticAttentionModule(global_dim=768, local_dim=768, num_heads=8
                                                     )
        self.r_att = EnhancedSemanticAttentionModule(global_dim=768, local_dim=768, num_heads=8
                                                     )
        self.vit = ViT(image_size=(224,448),name = 'B_16_imagenet1k',in_channels = 3,num_classes=8,pretrained=False) 
    def forward(self, x,texts=None):
        # 图像增强


        left_feat = self.backbone_front(x)

        left_feat = self.backbone_layer1(left_feat)

        left_feat = self.backbone_layer2(left_feat)

        left_feat = self.backbone_layer3(left_feat)
        left_feat = self.ffm(left_feat)
        #print(left_feat.shape)

        l_vf = self.vit(x)


        # 特征融合与分类
        logits = torch.cat([l_vf,left_feat],dim=1) 

        logits = self.classifier(logits)
        logits = self.last(logits)
        return logits






if __name__ == "__main__":
    x = torch.randn(4,3,224,448)
    text = torch.randn(4,768)
    model = BFPCNet1(num_classes=8)
    print(model(x,text).shape)
    # 测试
    # model = SwinFeatureExtractor()
    # x = torch.randn(1, 3, 224, 448)  # 示例输入
    # features = model(x)
    # print(features.shape)  # (B, N, C) 或 (B, C, H', W')
