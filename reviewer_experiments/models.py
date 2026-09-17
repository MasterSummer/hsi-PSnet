from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class RGBEncoder(nn.Module):
    def __init__(self, output_dim: int = 128, backbone: str = "resnet34", pretrained: bool = False):
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            network = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            network = models.resnet34(weights=weights)
        else:
            raise ValueError("RGB backbone must be resnet18 or resnet34")
        feature_dim = network.fc.in_features
        network.fc = nn.Identity()
        self.network = network
        self.project = nn.Linear(feature_dim, output_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.project(self.network(rgb))


class Spectral1DEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, output_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        spectrum = hsi.mean(dim=(-2, -1)).unsqueeze(1)
        return self.network(spectrum).flatten(1)


class Compact3DEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1, 16, (7, 5, 5), stride=(2, 2, 2), padding=(3, 2, 2)),
            nn.BatchNorm3d(16), nn.GELU(),
            nn.Conv3d(16, 32, (5, 3, 3), stride=(2, 2, 2), padding=(2, 1, 1)),
            nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, output_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        return self.network(hsi.unsqueeze(1)).flatten(1)


class HaarToken(nn.Module):
    def __init__(self, bands: int, dim: int, levels: int = 3):
        super().__init__()
        self.levels = levels
        self.project = nn.Sequential(nn.Linear(bands, dim), nn.GELU())

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        current = hsi.mean(dim=(-2, -1))
        coefficients = []
        for _ in range(self.levels):
            if current.shape[-1] % 2:
                current = F.pad(current, (0, 1), mode="replicate")
            even, odd = current[..., 0::2], current[..., 1::2]
            coefficients.append((even - odd) / math.sqrt(2.0))
            current = (even + odd) / math.sqrt(2.0)
        coefficients.append(current)
        flattened = torch.cat(coefficients, dim=-1)
        flattened = flattened[..., : self.project[0].in_features]
        if flattened.shape[-1] < self.project[0].in_features:
            flattened = F.pad(flattened, (0, self.project[0].in_features - flattened.shape[-1]))
        return self.project(flattened).unsqueeze(1)


class RawSpectrumToken(HaarToken):
    """Same projection and parameter count as HaarToken; omit only Haar transform."""

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        return self.project(hsi.mean(dim=(-2, -1))).unsqueeze(1)


class CrossLayerTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dropout: float, use_caf: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
            ) for _ in range(depth)
        ])
        self.use_caf = use_caf
        self.caf_source = "history"
        self.fusion = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(max(depth - 2, 0))])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        history = []
        for index, layer in enumerate(self.layers):
            if self.use_caf and index >= 2:
                previous = history[index - 2] if self.caf_source == "history" else tokens
                tokens = self.fusion[index - 2](torch.cat([tokens, previous], dim=-1))
            history.append(tokens)
            tokens = layer(tokens)
        return self.norm(tokens)


class HSITransformerEncoder(nn.Module):
    def __init__(
        self,
        bands: int,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        token_mode: str = "wavelet",
        use_caf: bool = True,
        use_3d_patch: bool = True,
        max_tokens: int = 1024,
        mean_depth_patch: bool = False,
    ):
        super().__init__()
        self.use_3d_patch = use_3d_patch
        self.token_mode = token_mode
        self.mean_depth_patch = mean_depth_patch
        if mean_depth_patch and not use_3d_patch:
            raise ValueError("mean_depth_patch requires the reduced 3-D patch path")
        if use_3d_patch:
            self.spectral_reduce = nn.Sequential(
                nn.Conv2d(bands, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU()
            )
            self.patch = (nn.Conv3d(1, dim, kernel_size=(1, 16, 16), stride=(1, 16, 16))
                          if mean_depth_patch else
                          nn.Conv3d(1, dim, kernel_size=(8, 16, 16), stride=(8, 16, 16)))
        else:
            self.spectral_reduce = None
            self.patch = nn.Conv2d(bands, dim, kernel_size=16, stride=16)
        if token_mode == "wavelet":
            self.global_token = HaarToken(bands, dim)
        elif token_mode == "raw_spectrum":
            self.global_token = RawSpectrumToken(bands, dim)
        elif token_mode == "learnable":
            self.global_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        elif token_mode == "mean":
            self.global_token = None
        else:
            raise ValueError("token_mode must be wavelet, raw_spectrum, learnable, or mean")
        self.position = nn.Parameter(torch.randn(1, max_tokens + 1, dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.transformer = CrossLayerTransformer(dim, depth, heads, dropout, use_caf)

    def patch_tokens(self, hsi: torch.Tensor) -> torch.Tensor:
        if self.use_3d_patch:
            reduced = self.spectral_reduce(hsi).unsqueeze(1)
            if self.mean_depth_patch:
                reduced = F.avg_pool3d(reduced, kernel_size=(8, 1, 1), stride=(8, 1, 1))
            embedded = self.patch(reduced)
            return embedded.flatten(2).transpose(1, 2)
        embedded = self.patch(hsi)
        return embedded.flatten(2).transpose(1, 2)

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        patches = self.patch_tokens(hsi)
        if patches.shape[1] + 1 > self.position.shape[1]:
            raise ValueError("too many patch tokens; increase max_tokens")
        if self.token_mode in {"wavelet", "raw_spectrum"}:
            global_token = self.global_token(hsi)
        elif self.token_mode == "learnable":
            global_token = self.global_token.expand(hsi.shape[0], -1, -1)
        else:
            global_token = patches.mean(dim=1, keepdim=True)
        tokens = torch.cat([global_token, patches], dim=1)
        tokens = self.dropout(tokens + self.position[:, : tokens.shape[1]])
        return self.transformer(tokens)[:, 0]


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, input_kind: str, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.input_kind = input_kind
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> torch.Tensor:
        value = rgb if self.input_kind == "rgb" else hsi
        return self.head(self.encoder(value))


class SimpleMultimodal(nn.Module):
    def __init__(self, rgb: nn.Module, hsi: nn.Module, feature_dim: int):
        super().__init__()
        self.rgb_encoder = rgb
        self.hsi_encoder = hsi
        self.head = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.GELU(), nn.Linear(feature_dim, 2))

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.rgb_encoder(rgb), self.hsi_encoder(hsi)], dim=1))


class PSNet(nn.Module):
    def __init__(
        self,
        bands: int,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        token_mode: str = "wavelet",
        use_caf: bool = True,
        use_3d_patch: bool = True,
        rgb_backbone: str = "resnet34",
        rgb_pretrained: bool = False,
        caf_source: str = "history",
        mean_depth_patch: bool = False,
    ):
        super().__init__()
        self.rgb_encoder = RGBEncoder(dim, rgb_backbone, rgb_pretrained)
        self.hsi_encoder = HSITransformerEncoder(
            bands, dim, depth, heads, dropout, token_mode, use_caf, use_3d_patch,
            mean_depth_patch=mean_depth_patch,
        )
        if caf_source not in {"history", "self"}:
            raise ValueError("caf_source must be history or self")
        self.hsi_encoder.transformer.caf_source = caf_source
        self.head = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, 2))

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.rgb_encoder(rgb), self.hsi_encoder(hsi)], dim=1))


MODEL_NAMES = (
    "psnet_full",
    "psnet_no_caf",
    "psnet_no_wavelet",
    "psnet_no_3d_patch",
    "psnet_learnable_cls",
    "plain_multimodal",
    "rgb_resnet18",
    "rgb_resnet34",
    "hsi_transformer",
    "spectral_1d_cnn",
    "compact_3d_cnn",
    "simple_multimodal",
    "psnet_raw_spectrum_token",
    "psnet_caf_self",
    "psnet_raw_token_caf_self",
    "psnet_mean_depth_patch",
)


def build_model(
    name: str,
    bands: int,
    dim: int = 128,
    depth: int = 4,
    heads: int = 4,
    dropout: float = 0.1,
    rgb_pretrained: bool = False,
) -> nn.Module:
    if name == "rgb_resnet18":
        return Classifier(RGBEncoder(dim, "resnet18", rgb_pretrained), "rgb", dim)
    if name == "rgb_resnet34":
        return Classifier(RGBEncoder(dim, "resnet34", rgb_pretrained), "rgb", dim)
    if name == "hsi_transformer":
        return Classifier(
            HSITransformerEncoder(bands, dim, depth, heads, dropout, "wavelet", True, True),
            "hsi",
            dim,
        )
    if name == "spectral_1d_cnn":
        return Classifier(Spectral1DEncoder(dim), "hsi", dim)
    if name == "compact_3d_cnn":
        return Classifier(Compact3DEncoder(dim), "hsi", dim)
    if name == "simple_multimodal":
        return SimpleMultimodal(
            RGBEncoder(dim, "resnet18", rgb_pretrained), Spectral1DEncoder(dim), dim
        )
    settings = {
        "psnet_full": ("wavelet", True, True),
        "psnet_no_caf": ("wavelet", False, True),
        "psnet_no_wavelet": ("mean", True, True),
        "psnet_no_3d_patch": ("wavelet", True, False),
        "psnet_learnable_cls": ("learnable", True, True),
        "plain_multimodal": ("learnable", False, True),
        "psnet_raw_spectrum_token": ("raw_spectrum", True, True),
        "psnet_caf_self": ("wavelet", True, True),
        "psnet_raw_token_caf_self": ("raw_spectrum", True, True),
        "psnet_mean_depth_patch": ("wavelet", True, True),
    }
    if name not in settings:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    token_mode, use_caf, use_3d_patch = settings[name]
    return PSNet(
        bands=bands,
        dim=dim,
        depth=depth,
        heads=heads,
        dropout=dropout,
        token_mode=token_mode,
        use_caf=use_caf,
        use_3d_patch=use_3d_patch,
        rgb_pretrained=rgb_pretrained,
        caf_source="self" if name in {"psnet_caf_self", "psnet_raw_token_caf_self"} else "history",
        mean_depth_patch=name == "psnet_mean_depth_patch",
    )
