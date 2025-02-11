import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights,resnet101,ResNet101_Weights
from vit_model import ViT
from vit_pytorch import ViT as V
import torch
import torch.nn as nn
import torch.nn.functional as F
from WTFD import WTFD,WTFDown
from FFA import FFA
from HFF import HFF_block
from main_model import HiFuse_Small 
from efficientnet_pytorch import EfficientNet
#from FreqFusion import FreqFusion


    
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
            nn.Dropout(0.5),
            nn.Linear(128,num_classes)
        )
    def forward(self, x):
        vit_fea = self.vit(x)
        resnet_fea = self.resnet(x)
        all_fea = torch.cat([vit_fea,resnet_fea],dim=1)
        classes = self.classifer(all_fea)
        return classes
    


class ResidualAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        #backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        #self.resnet = nn.Sequential(*list(backbone.children())[:-2])
        self.left_model = EfficientNet.from_pretrained('efficientnet-b3')
        self.conv = nn.Conv2d(1536,1024,kernel_size=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.7),
            nn.GELU(),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.Dropout(0.7),
            nn.Sigmoid()
        )
        self.wtfd = WTFDown(in_ch=1536,out_ch=2048)
        self.hff = HFF_block(ch_1=2048,ch_2 = 2048,r_2 = 32,ch_int=512,ch_out=2048,drop_rate=0.3)


    def forward(self, x):
        features = self.left_model.extract_features(x)  # [batch, 2048, 7, 7]
        #print(features.shape)
        fusion_fea= self.wtfd(features)

        #fusion_fea = self.hff(l=jf,g=qf,f=None)
        attention_weights = self.attention(fusion_fea)  # [batch, 2048, 1, 1]
        return fusion_fea * attention_weights   # 逐通道加权
    
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Query, Key, Value投影层（共享权重）
        self.qkv_left = nn.Linear(dim, dim * 3)
        self.qkv_right = nn.Linear(dim, dim * 3)
        
        # 输出投影层
        self.proj = nn.Linear(dim * 2, dim)
        
        # SE Block（通道注意力）
        self.se = nn.Sequential(

            nn.Linear(dim, dim // 16),  # reduction=16
            nn.ReLU(),
            nn.Linear(dim // 16, dim),
            nn.Sigmoid()
        )

    def forward(self, x_left, x_right):
        B = x_left.size(0)

        # 生成左右眼的QKV
        qkv_l = self.qkv_left(x_left).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_l, k_l, v_l = qkv_l[0], qkv_l[1], qkv_l[2]  # [B, num_heads, seq_len, head_dim]
        
        qkv_r = self.qkv_right(x_right).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_r, k_r, v_r = qkv_r[0], qkv_r[1], qkv_r[2]

        # 双向注意力计算
        # 左眼作为Query，右眼作为Key/Value
        attn_l2r = (q_l @ k_r.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_l2r = F.softmax(attn_l2r, dim=-1)
        out_l = (attn_l2r @ v_r).transpose(1, 2).reshape(B, -1, self.dim)
        
        # 右眼作为Query，左眼作为Key/Value
        attn_r2l = (q_r @ k_l.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_r2l = F.softmax(attn_r2l, dim=-1)
        out_r = (attn_r2l @ v_l).transpose(1, 2).reshape(B, -1, self.dim)
        
        # 双向结果拼接
        fused = torch.cat([out_l, out_r], dim=-1)  # [B, seq_len, 2*dim]
        fused = self.proj(fused)  # 投影回原始维度 [B, seq_len, dim]
        fused = fused.reshape(B,fused.shape[2])
        # 添加SE Block（通道注意力）
        se_weight = self.se(fused)  # [B, dim]

        fused = fused * se_weight  # 通道加权
        
        return fused

# 特征融合模块（引入加权融合）
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        self.ffa = FFA(inchannel=in_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的融合权重

    def forward(self, left_feat, right_feat):
        combined = torch.cat([left_feat, right_feat], dim=1)  # [batch, 4096, H, W]
        return self.conv(combined) * self.alpha  # 加权融合

# 主模型
class BFPCNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.ram_left = ResidualAttentionModule()
        self.ram_right = ResidualAttentionModule()
        self.ffm = FeatureFusionModule(4096, 2048)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(512, 8)
        )
        
        self.last = nn.Linear(2048,num_classes)
        self.fusion = BidirectionalCrossAttention(dim=2048)
        self.dp = nn.Dropout(0.7)

    def forward(self, x):
        # left_img = x[:,:3,:,:]
        # right_img = x[:,3:,:,:]
        # left_feat = self.ram_left(left_img)  # [batch, 2048, 7, 7]
        # right_feat = self.ram_right(right_img)  # [batch, 2048, 7, 7]
        # fused_feat = self.ffm(left_feat, right_feat)  # [batch, 2048, 7, 7]

        # pooled = nn.AdaptiveAvgPool2d(1)(fused_feat)  # [batch, 2048, 1, 1]
        # flattened = pooled.view(pooled.size(0), -1)  # [batch, 2048]
        # return self.last(self.classifier(flattened))  # [batch, num_classes]
        left_img = x[:,:3,:,:]
        right_img = x[:,3:,:,:]
        
        left_feat = self.ram_left(left_img)  # [batch, 2048, 7, 7]
        right_feat = self.ram_right(right_img)  # [batch, 2048, 7, 7]
          # [batch, 2048, 7, 7]
        left_feat = self.dp(left_feat)
        right_feat = self.dp(right_feat)
        left_pooled = nn.AdaptiveAvgPool2d(1)(left_feat)  # [batch, 2048, 1, 1]
        left_flattened = left_pooled.view(left_pooled.size(0), -1)  # [batch, 2048]
        right_pooled = nn.AdaptiveAvgPool2d(1)(right_feat)  # [batch, 2048, 1, 1]
        right_flattened = right_pooled.view(right_pooled.size(0), -1)  # [batch, 2048]
        fused_feat = self.fusion.forward(x_left=left_flattened,x_right=right_flattened)

        #flattened = self.classifier(fused_feat)  # [batch, 256]ssifier
 
        return self.classifier(fused_feat)  # [batch, num_classes]


class BiometricModel(nn.Module):
    def __init__(self, img_shape):
        super(BiometricModel, self).__init__()

        # 加载 EfficientNetB3 模型
        self.left_model = EfficientNet.from_pretrained('efficientnet-b3')
        self.right_model = EfficientNet.from_pretrained('efficientnet-b3')

        # 去除 EfficientNetB3 中的分类部分 (最后的全连接层)
        self.left_model._fc = nn.Identity()
        self.right_model._fc = nn.Identity()

        # Dropout 层
        self.dropout = nn.Dropout(0.5)

        # 全连接层，注意输入大小要根据 EfficientNetB3 的输出通道数来设置
        self.fc1 = nn.Linear(2 * self.left_model._conv_head.out_channels, 256)  # EfficientNet 输出通道数 * 2
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 8)

    def forward(self, x):
        # 通过左侧和右侧的 EfficientNet 模型进行前向传播
        left_input = x[:, :3, :, :]
        right_input = x[:, 3:, :, :]
        left_features = self.left_model.extract_features(left_input)
        right_features = self.right_model.extract_features(right_input)

        # 合并左侧和右侧模型的输出
        x = torch.cat((left_features, right_features), dim=1)  # 在通道维度上合并

        # 应用 Dropout
        x = self.dropout(x)

        # Global Average Pooling (GAP)
        x = torch.mean(x, dim=[2, 3])  # 对空间维度（高度和宽度）进行均值池化

        # 通过全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        out = self.out(x)

        return out
    
class simple_resnet(nn.Module):
    def __init__(self):
        super(simple_resnet, self).__init__()

        self.resnetleft = resnet50(pretrained=True)
        self.resnetright = resnet50(pretrained=True)
        self.resnetleft.fc = nn.Linear(self.resnetleft.fc.in_features, 256)
        self.resnetright.fc = nn.Linear(self.resnetright.fc.in_features, 256)
        self.classfier = nn.Linear(512, 8)
    def forward(self, x):
        left_input = x[:, :3, :, :]
        right_input = x[:, 3:, :, :]
        left_features = self.resnetleft(left_input)
        right_features = self.resnetright(right_input)
        x = torch.cat((left_features, right_features), dim=1)
        out = self.classfier(x)
        return out

if __name__ == "__main__":
    # 测试模型
    model = simple_resnet()
    dummy_input = torch.randn(8,6,224,224)
    label = torch.randn(8,8)  
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应输出 (4, 8)
