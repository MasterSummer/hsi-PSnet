from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import torch

from .models import build_model, trainable_parameters
from .train import choose_device


def profile_models(models: list[str], bands: int, height: int, width: int, output: str | Path,
                   device: str = "auto", batch_size: int = 1, warmup: int = 5, repeats: int = 20,
                   dim: int = 128, depth: int = 4, heads: int = 4) -> None:
    target = choose_device(device)
    rgb = torch.randn(batch_size, 3, 224, 224, device=target)
    hsi = torch.randn(batch_size, bands, height, width, device=target)
    rows = []
    for name in models:
        model = build_model(name, bands, dim, depth, heads).to(target).eval()
        with torch.inference_mode():
            for _ in range(warmup):
                model(rgb, hsi)
            if target.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(repeats):
                model(rgb, hsi)
            if target.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000 / repeats
        rows.append({"model": name, "trainable_parameters": trainable_parameters(model),
                     "batch_size": batch_size, "device": str(target), "latency_ms_per_batch": elapsed,
                     "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if target.type == "cuda" else float("nan"),
                     "rgb_shape": "3x224x224", "hsi_shape": f"{bands}x{height}x{width}"})
        del model
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
