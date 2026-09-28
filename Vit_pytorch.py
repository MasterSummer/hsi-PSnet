import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from einops import rearrange, repeat
import math
from einops import rearrange
import pywt

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., activation=nn.GELU()):
        super().__init__()
        self.activation = activation

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None, return_attn: bool = False):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if return_attn:
            return out, attn
        return out

class SEBlockViT(nn.Module):
    """适用于 ViT 的 SE 模块，输入形状 [B, N, C]"""
    def __init__(self, dim, reduction=4, activation=nn.GELU()):
        super().__init__()
        self.reduction = reduction
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            activation,  # 保持与 ViT 激活函数一致
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()  # 输出通道权重 [0, 1]
        )

    def forward(self, x):
        # x: [B, N, C]
        b, n, c = x.shape
        y = x.mean(dim=1)  # 全局平均池化 [B, C]
        y = self.fc(y).view(b, 1, c)  # 生成通道权重 [B, 1, C]
        return x * y  # 广播乘法 [B, N, C] * [B, 1, C]


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))
            # 初始化 Conv2d 层权重
            # nn.init.xavier_uniform_(self.skipcat[-1].weight)
         # 添加归一化层
        self.skipcat_norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask = None, return_attn: bool = False):
        attn_maps = [] if return_attn else None
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                if return_attn:
                    x, attn_map = attn(x, mask = mask, return_attn=True)
                    attn_maps.append(attn_map)
                else:
                    x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = [] 
            nl = 0
            for i, (attn, ff) in enumerate(self.layers):
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                    # x = self.skipcat_norm(x)
                    # print(f"Layer {i+1} - After Skip Connection: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
                if return_attn:
                    x, attn_map = attn(x, mask=mask, return_attn=True)
                    attn_maps.append(attn_map)
                else:
                    x = attn(x, mask=mask)
                # print(f"Layer {i+1} - After Attention: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
                x = ff(x)
                # print(f"Layer {i+1} - After FeedForward: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
                nl += 1

        if return_attn:
            return x, attn_maps
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size=(8, 16, 16), activation=nn.GELU()):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.band=64
        
       
        # 新增光谱降维部分：使用3×3卷积将313通道降维到64
        self.spectral_net = nn.Sequential(
            nn.Conv2d(313, self.band, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band),
            activation,
        )


        self.patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.conv = nn.Sequential(
            nn.Conv1d(self.patch_volume, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            self.activation,
            nn.Conv1d(256, embed_dim, kernel_size=1),
        )

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
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        num_patches = x.shape[1]
        return x, num_patches

class WaveletFeatureExtractor(nn.Module):
    def __init__(self, num_bands, dim, wavelet_type='db4', decomposition_levels=3):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.decomposition_levels = decomposition_levels
        
        # 1. 预先计算小波滤波器
        wavelet = pywt.Wavelet(self.wavelet_type)
        self.register_buffer('dec_lo', torch.tensor(wavelet.dec_lo).float())
        self.register_buffer('dec_hi', torch.tensor(wavelet.dec_hi).float())
        
        # 2. 动态计算 total_coeffs
        # 我们用一个与实际数据相似的dummy tensor来计算
        dummy_data = torch.zeros(1, num_bands)
        coeffs = self._wavelet_transform(dummy_data, check_dim=True)
        total_coeffs = coeffs.shape[-1]
        
        # 3. 特征嵌入层
        self.feature_embedder = nn.Sequential(
            nn.Linear(total_coeffs, dim),
            nn.GELU()
        )
        
    def _wavelet_transform(self, x, check_dim=False):
        # x shape: [N, C] where N is num_pixels, C is num_bands
        x_current = x.unsqueeze(1) # [N, 1, C]
        coeffs = []
        for _ in range(self.decomposition_levels):
            A = F.conv1d(x_current, self.dec_lo.unsqueeze(0).unsqueeze(0), padding='same')
            D = F.conv1d(x_current, self.dec_hi.unsqueeze(0).unsqueeze(0), padding='same')
            A = A[:, :, ::2]
            D = D[:, :, ::2]
            x_current = A
            coeffs.append(D)
        coeffs.append(x_current)
        
        flattened_coeffs = torch.cat(coeffs, dim=-1).squeeze(1) # [N, total_coeffs]
        
        return flattened_coeffs

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        x_reshaped = rearrange(x, 'b c h w -> (b h w) c') # [B*H*W, C]
        
        flattened_coeffs = self._wavelet_transform(x_reshaped)

        # 对所有像素的特征进行平均，得到一个代表整张图像的全局特征
        global_feature = flattened_coeffs.mean(dim=0).unsqueeze(0) # [1, total_coeffs]
        
        wavelet_token = self.feature_embedder(global_feature) # [1, D]
        wavelet_token = wavelet_token.expand(B, -1) # [B, D]
        
        return wavelet_token.unsqueeze(1) # [B, 1, D]
    
import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbeddingHSI(nn.Module):
    """
    高光谱图像 Patch Embedding
    输入: [B, C, H, W]
    输出: 
        - patch_tokens: [B, N, dim]
        - num_patches: N
    """
    def __init__(self, img_size=(132, 135), patch_size=15, in_chans=313, dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.dim = dim

        # 计算 patch 数量
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # 使用 Conv2d 实现 patch 划分和线性投影
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入尺寸不对: 需要 {(self.img_size[0], self.img_size[1])}, 实际 {(H, W)}"

        # [B, C, H, W] -> [B, dim, H/patch, W/patch]
        x = self.proj(x)

        # 展平成序列 [B, N, dim]
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x, self.num_patches

class ViT(nn.Module):
    def __init__(self, num_classes=4, dim=1024, depth=5, heads=4, mlp_dim=512, pool='cls', channels=313, dim_head = 16, dropout=0.1, emb_dropout=0.1, mode='CAF', activation=nn.GELU()):
        super().__init__()

        self.wavelet_extractor = WaveletFeatureExtractor(num_bands=channels, dim=dim)

        self.patch_to_embedding = PatchEmbeddingHSI(img_size=(132,135), patch_size=15, in_chans=313, dim=dim)

        dummy_input = torch.randn(1, channels, 132, 135)
        _ , self.num_patches = self.patch_to_embedding(dummy_input)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, self.num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x, mask = None, return_tokens: bool = False, return_attn: bool = False):

        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')
        wavelet_token = self.wavelet_extractor(x)  # [B, 1, D]

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x, num_patches= self.patch_to_embedding(x) #[b,n,dim]
        if self.pos_embedding.shape[1] != num_patches + 1:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.pos_embedding.shape[2], device=x.device))

        b, n, _ = x.shape
        # print(n, _)
        x_patches = x
        
        x = torch.zeros(b, n + 1, self.pos_embedding.shape[-1], device=x.device)
        
        #    - 将小波 token 作为第一个 token (即 [CLS] token)
        x[:, 0] = wavelet_token.squeeze(1)

        #    - 将局部 patch token 拼接到后面
        x[:, 1:] = x_patches
        
        
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) #[b,1,dim]
#         x = torch.cat((cls_tokens, x), dim = 1) #[b,n+1,dim]

        

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        if return_attn:
            x, attn_maps = self.transformer(x, mask, return_attn=True)
        else:
            x = self.transformer(x, mask)
            attn_maps = None

        # classification: using cls_token output
        tokens = x
        cls_token_out = self.to_latent(tokens[:,0])

        # MLP classification layer
        if return_tokens or return_attn:
            extras = {
                'tokens': tokens,
                'attn_maps': attn_maps,
                'cls_token': tokens[:, 0],
            }
            return cls_token_out, extras

        return cls_token_out
