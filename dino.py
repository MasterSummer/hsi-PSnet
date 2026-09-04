import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from transformers import Dinov2Model, Dinov2Config
import os

# 设置 HuggingFace 镜像（如使用国内环境）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size=(8, 16, 16), activation=nn.GELU()):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.band = 64

        self.spectral_net = nn.Sequential(
            nn.Conv2d(313, self.band, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band),
            activation,
        )

        self.patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.conv = nn.Sequential(
            nn.Conv1d(self.patch_volume, 1024,  kernel_size=1),
            nn.BatchNorm1d(256),
            self.activation,
            nn.Conv1d(1024, embed_dim, kernel_size=1),
        )
        self.mlp = nn.Linear(self.patch_volume,embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.spectral_net(x)

        p_c, p_h, p_w = self.patch_size
        new_C = math.ceil(self.band / p_c) * p_c
        new_H = math.ceil(H / p_h) * p_h
        new_W = math.ceil(W / p_w) * p_w

        x = F.pad(x, (0, new_W - W, 0, new_H - H, 0, new_C - self.band))
        x = x.unfold(1, p_c, p_c).unfold(2, p_h, p_h).unfold(3, p_w, p_w)
        x = x.contiguous().view(B, -1, self.patch_volume)

        # x = x.transpose(1, 2)
        # x = self.conv(x)
        # x = x.transpose(1, 2)
        x = self.mlp(x)

        return x, x.shape[1]  # [B, N, D], N


class Dino(nn.Module):
    def __init__(self, num_classes=4, model_name="facebook/dinov2-base",
                 local_model_path="dinov2-base", patch_size=(3, 16, 16)):
        super().__init__()

        local_model_path = Path(local_model_path).absolute()
        required_files = ['config.json', 'pytorch_model.bin']
        missing_files = [f for f in required_files if not (local_model_path / f).exists()]

        if missing_files:
            raise FileNotFoundError(
                f"缺少模型文件: {missing_files}\n"
                f"请确认目录 '{local_model_path}' 包含这些文件\n"
                f"当前目录内容: {os.listdir(local_model_path) if local_model_path.exists() else '目录不存在'}"
            )

        self.embed_dim = 768
        self.patch_embed = PatchEmbedding(embed_dim=self.embed_dim, patch_size=patch_size)

        try:
            self.config = Dinov2Config.from_pretrained(local_model_path)
            self.transformer = Dinov2Model.from_pretrained(
                local_model_path,
                config=self.config,
                local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(
                f"模型加载失败: {e}\n"
                "请检查模型文件是否完整/版本兼容。"
            )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def interpolate_pos_embed(self, pos_embed, x):
        # 插值位置编码：支持任意 patch 数
        cls_token = pos_embed[:, 0:1, :]        # [1,1,D]
        pos_tokens = pos_embed[:, 1:, :]        # [1,N,D]
        num_patches = x.shape[1] - 1

        orig_size = int(pos_tokens.shape[1] ** 0.5)
        new_size = int(num_patches ** 0.5)

        if orig_size != new_size:
            pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, -1, pos_tokens.shape[1])

        return torch.cat([cls_token, pos_tokens], dim=1)

    def forward(self, x):
        B = x.size(0)
        x, _ = self.patch_embed(x)  # [B, N, D]

        # 拼接 cls token
        cls_token = self.transformer.embeddings.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 插值位置编码
        # pos_embed_interp = self.interpolate_pos_embed(self.transformer.embeddings.position_embeddings, x)
        # x = x + pos_embed_interp
        # x = self.transformer.embeddings.dropout(x)

        # 进入 Transformer
        x = self.transformer.encoder(x).last_hidden_state  # [B, N+1, D]
        cls_token_final = x[:, 0]  # 取出 [CLS] token

        x = self.norm(cls_token_final)
        # logits = self.head(x)
        return x
    
model = Dino(
    num_classes=4,
    local_model_path="dinov2-base"  # 指向本地 DINOv2 模型目录
)

# x = torch.randn(2, 313, 132, 135)
# logits = model(x)
# print(logits.shape)  # [2, 4]
