from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


PT_BUNDLE_PREFIX = "ptbundle:"


def resolve_path(value: str, root: str | Path | None) -> Path | str:
    if str(value).startswith(PT_BUNDLE_PREFIX):
        return str(value)
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    return path


def make_pt_bundle_uri(path: str | Path, index: int) -> str:
    return f"{PT_BUNDLE_PREFIX}{Path(path).expanduser().resolve()}::{int(index)}"


def parse_pt_bundle_uri(value: str) -> tuple[Path, int]:
    if not value.startswith(PT_BUNDLE_PREFIX) or "::" not in value:
        raise ValueError(f"invalid PT bundle URI: {value}")
    path, index = value[len(PT_BUNDLE_PREFIX):].rsplit("::", 1)
    return Path(path), int(index)


@lru_cache(maxsize=4)
def _load_pt_bundle(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_hsi(path: str | Path, layout: str = "auto") -> np.ndarray:
    path_value = str(path)
    if path_value.startswith(PT_BUNDLE_PREFIX):
        bundle_path, index = parse_pt_bundle_uri(path_value)
        bundle = _load_pt_bundle(str(bundle_path))
        if index < 0 or index >= len(bundle):
            raise IndexError(f"PT bundle index {index} is outside {bundle_path}")
        item = bundle[index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError(f"unexpected item {index} in {bundle_path}")
        loaded = item[1]
        if isinstance(loaded, torch.Tensor):
            loaded = loaded.detach().cpu().numpy()
    else:
        loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.files)
        if len(keys) != 1:
            raise ValueError(f"{path} must contain one NPZ array; found {keys}")
        array = loaded[keys[0]]
    else:
        array = loaded
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 3:
        raise ValueError(f"HSI cube must be 3-D; found {array.shape} in {path}")
    layout = str(layout).upper()
    if layout == "HWC":
        array = np.moveaxis(array, -1, 0)
    elif layout == "CHW":
        pass
    elif layout == "AUTO":
        channel_axis = int(np.argmax(array.shape))
        if channel_axis != 0:
            array = np.moveaxis(array, channel_axis, 0)
    else:
        raise ValueError("hsi_layout must be auto, HWC, or CHW")
    return np.ascontiguousarray(array)


@dataclass
class BandStats:
    mean: np.ndarray
    std: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def compute_band_stats(frame, data_root: str | Path | None = None) -> BandStats:
    total = None
    total_sq = None
    count = 0
    for row in frame.itertuples(index=False):
        layout = getattr(row, "hsi_layout", "auto")
        cube = load_hsi(resolve_path(row.hsi_path, data_root), layout)
        flat = cube.reshape(cube.shape[0], -1).astype(np.float64)
        if total is None:
            total = np.zeros(cube.shape[0], dtype=np.float64)
            total_sq = np.zeros(cube.shape[0], dtype=np.float64)
        if cube.shape[0] != len(total):
            raise ValueError("all HSI cubes must have the same number of bands")
        total += flat.sum(axis=1)
        total_sq += np.square(flat).sum(axis=1)
        count += flat.shape[1]
    if total is None or count == 0:
        raise ValueError("cannot compute statistics from an empty training set")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return BandStats(mean.astype(np.float32), np.sqrt(variance).astype(np.float32))


class PairedDataset(Dataset):
    def __init__(
        self,
        frame,
        band_stats: BandStats,
        data_root: str | Path | None = None,
        training: bool = False,
        rgb_size: int = 224,
    ):
        self.frame = frame.reset_index(drop=True)
        self.band_stats = band_stats
        self.data_root = data_root
        self.training = training
        self.rgb_size = rgb_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        rgb = Image.open(resolve_path(row.rgb_path, self.data_root)).convert("RGB")
        rgb = TF.resize(rgb, [self.rgb_size, self.rgb_size], antialias=True)
        rgb = TF.to_tensor(rgb)
        rgb = TF.normalize(rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        cube = load_hsi(
            resolve_path(row.hsi_path, self.data_root),
            row.get("hsi_layout", "auto"),
        )
        if cube.shape[0] != len(self.band_stats.mean):
            raise ValueError(f"unexpected band count in {row.hsi_path}")
        cube = (cube - self.band_stats.mean[:, None, None]) / self.band_stats.std[:, None, None]
        hsi = torch.from_numpy(np.ascontiguousarray(cube)).float()

        if self.training:
            if random.random() < 0.5:
                rgb = torch.flip(rgb, dims=[2])
                hsi = torch.flip(hsi, dims=[2])
            if random.random() < 0.5:
                rgb = torch.flip(rgb, dims=[1])
                hsi = torch.flip(hsi, dims=[1])
        label = int(row.label)
        return rgb, hsi, label, str(row.sample_id), str(row.plant_id)
