from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import BandStats, PairedDataset
from .metrics import classification_metrics, plant_predictions
from .models import build_model
from .train import choose_device, predict


def band_occlusion(
    checkpoint: str | Path, task_csv: str | Path, output: str | Path,
    data_root: str | Path | None = None, group_size: int = 10,
    batch_size: int = 8, workers: int = 0, device: str = "auto",
) -> None:
    target_device = choose_device(device)
    saved = torch.load(checkpoint, map_location=target_device, weights_only=False)
    config = saved["config"]
    stats = BandStats.from_dict(saved["band_stats"])
    frame = pd.read_csv(task_csv, dtype={"sample_id": str, "plant_id": str})
    validation = frame[frame.fold == int(config["fold"])].copy()
    dataset = PairedDataset(validation, stats, data_root, False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = build_model(config["model"], config["bands"], config["dim"], config["depth"], config["heads"], config["dropout"], False).to(target_device)
    model.load_state_dict(saved["model_state"])
    baseline_predictions = predict(model, loader, target_device, config["model"], config["task"], config["seed"], config["fold"])
    baseline = classification_metrics(plant_predictions(baseline_predictions))
    rows = []
    model.eval()
    for start in range(0, int(config["bands"]), group_size):
        end = min(start + group_size, int(config["bands"]))
        predictions = []
        with torch.no_grad():
            for rgb, hsi, labels, sample_ids, plant_ids in loader:
                hsi[:, start:end] = 0.0
                probability = torch.softmax(model(rgb.to(target_device), hsi.to(target_device)), dim=1)[:, 1].cpu().numpy()
                for sample_id, plant_id, label, value in zip(sample_ids, plant_ids, labels.numpy(), probability):
                    predictions.append({"model": config["model"], "task": config["task"], "seed": config["seed"],
                                        "sample_id": sample_id, "plant_id": plant_id, "label": int(label), "prob_infected": float(value)})
        metric = classification_metrics(plant_predictions(pd.DataFrame(predictions)))
        rows.append({"band_start": start, "band_end_exclusive": end,
                     "original_band_indices": json.dumps((stats.band_indices or list(range(config["bands"])))[start:end]),
                     "model": config["model"], "task": config["task"],
                     "seed": int(config["seed"]), "fold": int(config["fold"]),
                     "balanced_accuracy": metric["balanced_accuracy"], "auroc": metric["auroc"],
                     "balanced_accuracy_drop": float(baseline["balanced_accuracy"]) - float(metric["balanced_accuracy"]),
                     "auroc_drop": float(baseline["auroc"]) - float(metric["auroc"])})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_predictions.to_csv(output / "baseline_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "band_occlusion.csv", index=False)
    (output / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def aggregate_occlusion(
    inputs: list[str | Path], output: str | Path, wavelengths: str | Path | None = None
) -> None:
    frame = pd.concat([pd.read_csv(path) for path in inputs], ignore_index=True)
    required = {
        "model", "task", "seed", "fold", "band_start", "band_end_exclusive",
        "balanced_accuracy_drop", "auroc_drop",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"occlusion files are missing columns: {sorted(missing)}")
    # Legacy full-spectrum files did not store source indices.
    def original_indices(row):
        value = row.get("original_band_indices")
        indices = json.loads(value) if pd.notna(value) else list(range(int(row.band_start), int(row.band_end_exclusive)))
        return json.dumps(indices)
    frame["original_band_indices"] = frame.apply(original_indices, axis=1)
    keys = ["model", "task", "band_start", "band_end_exclusive", "original_band_indices"]
    summary = frame.groupby(keys, as_index=False).agg(
        runs=("fold", "size"),
        balanced_accuracy_drop_mean=("balanced_accuracy_drop", "mean"),
        balanced_accuracy_drop_std=("balanced_accuracy_drop", "std"),
        auroc_drop_mean=("auroc_drop", "mean"),
        auroc_drop_std=("auroc_drop", "std"),
    )
    if wavelengths is not None:
        table = pd.read_csv(wavelengths)
        if not {"band", "wavelength_nm"} <= set(table.columns):
            raise ValueError("wavelength CSV must contain band,wavelength_nm")
        if table.band.duplicated().any():
            raise ValueError("Duplicate wavelength band indices")
        values = table.set_index("band").wavelength_nm.to_dict()
        summary["wavelengths_nm"] = summary.original_band_indices.map(
            lambda value: json.dumps([values.get(index) for index in json.loads(value)])
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "band_occlusion_all_runs.csv", index=False)
    summary.to_csv(output / "band_occlusion_summary.csv", index=False)
