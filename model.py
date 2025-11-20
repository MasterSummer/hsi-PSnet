

import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, AlexNet_Weights, DenseNet121_Weights, resnet34, \
    ResNet34_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Block  # ViT Block
from torch.nn import LayerNorm, Linear, Dropout, Softmax
from Vit_pytorch import *
# from efficientnet_pytorch import EfficientNet
# from transformers import ViTModel, ViTConfig

class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # 修改输入层
        if in_channels != 3:
            base_model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7,
                stride=2, padding=3, bias=False
            )
            # 初始化新输入层
            nn.init.kaiming_normal_(self.resnet34.conv1.weight, mode="fan_out")

        # 构建编码器
        self.features = nn.Sequential(*list(self.resnet34.children())[:-1])

    def forward(self, x):
        x = self.features(x)  # [batch, 512, 1, 1]
        x = x.flatten(1)  # [batch, 512]
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4): # reduction可以调整
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): # 输入 x 形状为 (B, D)
        b, d = x.shape
        y = self.avg_pool(x.view(b, d, 1)).view(b, d) # Squeeze
        y = self.fc(y).view(b, d) # Excitation
        return x * y # Rescale

class MultiModalNet(nn.Module):
    def __init__(self, num_classes=4, fuse_method='concat', dropout=0.5, heads=4, dim_head=16,token_dim=1024, mlp_dim=512, activation=nn.GELU() , c=313):
        super().__init__()

        hsi_dim = 64
        rgb_dim = 64
        token_dim = token_dim
        mid_dim = 256
        feature_dim = hsi_dim + rgb_dim
        self.activation = activation

        # RGB 和 HSI 编码器
        self.rgb_encoder = ResNet34Encoder()
        self.hsi_encoder = ViT(mode='CAF', dim=token_dim, dropout=dropout, emb_dropout=dropout, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,  activation=activation, channels= c )
        
        self.fc_hsi = nn.Sequential(
            nn.Linear(token_dim, hsi_dim),
            # nn.BatchNorm1d(hsi_dim),
            self.activation,
            # nn.Dropout(dropout),
        )
        self.fc_rgb = nn.Sequential(
            nn.Linear(512, rgb_dim),
            # nn.BatchNorm1d(hsi_dim),
            self.activation,
            # nn.Dropout(dropout),
        )
        # 最终融合 + 分类器
        self.mlp = nn.Sequential(
            # nn.Linear(feature_dim, num_classes),
            nn.Linear(feature_dim, mid_dim),
            # nn.BatchNorm1d(mid_dim),
            self.activation,
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, rgb, hsi, return_features: bool = False, return_attn: bool = False):
        hsi_out = self.hsi_encoder(
            hsi,
            return_tokens=return_attn,  # 只在需要注意力时取回token，避免额外开销
            return_attn=return_attn
        )
        if isinstance(hsi_out, tuple):
            hsi_feat_cls, hsi_info = hsi_out
        else:
            hsi_feat_cls, hsi_info = hsi_out, {}

        f_rgb = self.fc_rgb(self.rgb_encoder(rgb))  # RGB 特征
        f_hsi = self.fc_hsi(hsi_feat_cls)  # HSI 特征 (缩放后)
        fused = torch.cat([f_rgb, f_hsi], dim=1)  # 拼接融合

        # 手动拆开分类器，便于拿到分类前的隐藏向量
        hidden = self.mlp[0](fused)
        hidden = self.mlp[1](hidden)
        logits = self.mlp[2](hidden)

        if return_features or return_attn:
            extras = {
                'fused': fused,
                'pre_logits': hidden,
                'hsi_proj': f_hsi,
                'hsi_encoder_output': hsi_feat_cls,
                'rgb_feat': f_rgb,
            }
            if return_attn:
                extras.update({
                    'hsi_tokens': hsi_info.get('tokens'),
                    'hsi_attention': hsi_info.get('attn_maps'),
                    'hsi_cls_token': hsi_info.get('cls_token'),
                })
            return logits, extras

        return logits
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50
from torchvision.models.vision_transformer import vit_b_16


# ========== Patch Embedding for HSI ==========
class HSIPatchEmbed(nn.Module):
    def __init__(self, in_chans=313, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.num_patches = (224 // patch_size) * (224 // patch_size)

    def forward(self, x):
        # x: (B, 313, 224, 224)
        x = self.patch_embed(x)   # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ========== ResNet Encoder ==========
class ResNetEncoder(nn.Module):
    def __init__(self, in_chans=3, out_dim=512, backbone="resnet34"):
        super().__init__()
        if backbone == "resnet34":
            net = resnet34(weights=None)
            feat_dim = 512
        elif backbone == "resnet50":
            net = resnet50(weights=None)
            feat_dim = 2048
        else:
            raise ValueError("Unsupported backbone")

        if in_chans != 3:
            # 替换第一层以适配 HSI
            net.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = nn.Sequential(*list(net.children())[:-1])  # 去掉分类头
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        # x: (B, C, 224, 224)
        feat = self.encoder(x)  # (B, feat_dim, 1, 1)
        feat = feat.flatten(1)
        feat = self.fc(feat)    # (B, out_dim)
        return feat


# ========== ViT Encoder ==========
class ViTEncoder(nn.Module):
    def __init__(self, in_chans=3, out_dim=512, pretrained=False):
        super().__init__()
        vit = vit_b_16(weights=None if not pretrained else "IMAGENET1K_V1")
        if in_chans != 3:
            vit.conv_proj = nn.Conv2d(in_chans, vit.conv_proj.out_channels,
                                      kernel_size=vit.conv_proj.kernel_size,
                                      stride=vit.conv_proj.stride,
                                      padding=vit.conv_proj.padding,
                                      bias=False)
        self.encoder = vit
        self.fc = nn.Linear(vit.hidden_dim, out_dim)

    def forward(self, x):
        # x: (B, C, 224, 224)
        feat = self.encoder(x)[:, 0]  # CLS token
        feat = self.fc(feat)          # (B, out_dim)
        return feat


# ========== MultiModal Net (for comparison) ==========
class MultiModalNetCompare(nn.Module):
    def __init__(self, num_classes=4, fuse_method='concat', dropout=0.5, heads=4, dim_head=16,token_dim=1024, mlp_dim=512, activation=nn.GELU() , c=313 , backbone="resnet34", use_vit=False):
#         super().__init__()
#         if use_vit:
#             self.rgb_encoder = ViTEncoder(in_chans=3, out_dim=token_dim)
#             self.hsi_encoder = ViTEncoder(in_chans=313, out_dim=token_dim)
#         else:
#             self.rgb_encoder = ResNetEncoder(in_chans=3, out_dim=token_dim, backbone=backbone)
#             self.hsi_encoder = ResNetEncoder(in_chans=313, out_dim=token_dim, backbone=backbone)

#         self.fusion = fusion
#         if fusion == "concat":
#             self.classifier = nn.Linear(token_dim * 2, 4)  # 4 classes: Healthy, 2dpi, 4dpi, 6dpi
#         elif fusion == "add":
#             self.classifier = nn.Linear(token_dim, 4)
#         else:
#             self.classifier = nn.Linear(token_dim, 4)

#     def forward(self, rgb, hsi):
#         rgb_feat = self.rgb_encoder(rgb)
#         hsi_feat = self.hsi_encoder(hsi)
#         if self.fusion == "concat":
#             feat = torch.cat([rgb_feat, hsi_feat], dim=1)
#             out = self.classifier(feat)
#         elif self.fusion == "rgb:
#             # feat = rgb_feat + hsi_feat
#             out = self.classifier(rgb_feat)
#         elif self.fusion == "hsi":
#             out = self.classifier(hsi_feat)
        
#         return out
        super().__init__()
        
        self.fusion = fuse_method
        
        if use_vit:
            self.rgb_encoder = ViTEncoder(in_chans=3, out_dim=token_dim)
            self.hsi_encoder = ViTEncoder(in_chans=c, out_dim=token_dim)
        else:
            self.rgb_encoder = ResNetEncoder(in_chans=3, out_dim=token_dim, backbone=backbone)
            self.hsi_encoder = ResNetEncoder(in_chans=c, out_dim=token_dim, backbone=backbone)

        # 根据融合方式决定分类器维度
        if self.fusion == "concat":
            self.classifier = nn.Linear(token_dim * 2, num_classes)
        elif self.fusion == "add":
            self.classifier = nn.Linear(token_dim, num_classes)
        elif self.fusion in ["rgb", "hsi"]:
            self.classifier = nn.Linear(token_dim, num_classes)
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

    def forward(self, rgb, hsi):
        rgb_feat = self.rgb_encoder(rgb)   # shape [B, token_dim]
        hsi_feat = self.hsi_encoder(hsi)   # shape [B, token_dim]

        if self.fusion == "concat":
            feat = torch.cat([rgb_feat, hsi_feat], dim=1)
        elif self.fusion == "add":
            feat = rgb_feat + hsi_feat
        elif self.fusion == "rgb":
            feat = rgb_feat
        elif self.fusion == "hsi":
            feat = hsi_feat
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

        out = self.classifier(feat)
        return out
