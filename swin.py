import torch
import torch.nn as nn


# 1. 光谱降维模块（313 → band）
class SpectralReduction1D(nn.Module):
    def __init__(self, in_channels=313, out_channels=32):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)      # [B, C, H*W]
        x = self.conv1d(x)        # [B, band, H*W]
        x = x.view(B, -1, H, W)   # [B, band, H, W]
        return x


# 2. Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x, (H, W)


# 3. SwinBlock（非窗口注意力 + MLP）
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, H, W):
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + shortcut
        return x


# ✅ 修复核心：SwinStage 包装多个 SwinBlock，允许传入 H, W
class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads=num_heads) for _ in range(depth)
        ])

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


# 4. Patch Merging（2×下采样）
class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.reduction(x)  # [B, C_out, H/2, W/2]
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


# 主体结构：4层非窗口 Swin Transformer
class SwinTransformer(nn.Module):
    def __init__(self, in_chans=313, band=32, patch_size=4,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.spectral_reduction = SpectralReduction1D(in_chans, band)
        self.patch_embed = PatchEmbedding(band, embed_dims[0], patch_size)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(4):
            stage = SwinStage(embed_dims[i], depths[i], num_heads[i])
            self.stages.append(stage)
            if i < 3:
                self.downsamples.append(PatchMerging(embed_dims[i], embed_dims[i + 1]))

    def forward(self, x):
        x = self.spectral_reduction(x)
        x, (H, W) = self.patch_embed(x)

        features = []
        for i in range(4):
            x = self.stages[i](x, H, W)
            features.append((x, H, W))
            if i < 3:
                x, H, W = self.downsamples[i](x, H, W)

        return features  # [(x1, H1, W1), ..., (x4, H4, W4)]


# ✅ 测试入口
if __name__ == "__main__":
    model = SwinTransformer()
    dummy = torch.randn(2, 313, 132, 135)  # [B, C=313, H, W]
    feats = model(dummy)

    for i, (x, H, W) in enumerate(feats):
        print(f"Stage {i+1}: token = {x.shape}, spatial = ({H}, {W})")
