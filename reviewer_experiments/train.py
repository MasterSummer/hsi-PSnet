from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .data import BandStats, PairedDataset, compute_band_stats, load_hsi, resolve_path
from .metrics import classification_metrics, plant_predictions
from .models import build_model, trainable_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device, model_name: str, task: str, seed: int, fold: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for rgb, hsi, labels, sample_ids, plant_ids in loader:
        probabilities = torch.softmax(model(rgb.to(device), hsi.to(device)), dim=1)[:, 1].cpu().numpy()
        for sample_id, plant_id, label, probability in zip(sample_ids, plant_ids, labels.numpy(), probabilities):
            rows.append({
                "model": model_name, "task": task, "seed": seed, "fold": fold,
                "sample_id": sample_id, "plant_id": plant_id, "label": int(label),
                "prob_infected": float(probability),
            })
    return pd.DataFrame(rows)


def train_fold(
    task_csv: str | Path, model_name: str, seed: int, fold: int, output: str | Path,
    data_root: str | Path | None = None, epochs: int = 50, patience: int = 10,
    batch_size: int = 8, workers: int = 0, learning_rate: float = 3e-4,
    weight_decay: float = 1e-4, device: str = "auto", dim: int = 128,
    depth: int = 4, heads: int = 4, dropout: float = 0.1, rgb_pretrained: bool = False,
    inner_val_fraction: float = 0.2,
) -> Path:
    set_seed(seed)
    task_csv = Path(task_csv)
    task = task_csv.stem
    frame = pd.read_csv(task_csv, dtype={"sample_id": str, "plant_id": str})
    if "fold" not in frame or fold not in set(frame.fold):
        raise ValueError(f"fold {fold} is absent from {task_csv}")
    outer_training = frame[frame.fold != fold].copy()
    testing = frame[frame.fold == fold].copy()
    plants = outer_training[["plant_id", "label"]].drop_duplicates()
    fit_plants, early_stop_plants = train_test_split(
        plants, test_size=inner_val_fraction, random_state=seed + fold,
        stratify=plants["label"],
    )
    training = outer_training[outer_training.plant_id.isin(fit_plants.plant_id)].copy()
    validation = outer_training[outer_training.plant_id.isin(early_stop_plants.plant_id)].copy()
    memberships = [set(part.plant_id) for part in (training, validation, testing)]
    if any(memberships[i] & memberships[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("plant leakage across fit, early-stop, and outer-test sets")
    stats = compute_band_stats(training, data_root)
    first = training.iloc[0]
    bands = load_hsi(resolve_path(first.hsi_path, data_root), first.get("hsi_layout", "auto")).shape[0]
    train_loader = DataLoader(PairedDataset(training, stats, data_root, True), batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(PairedDataset(validation, stats, data_root, False), batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(PairedDataset(testing, stats, data_root, False), batch_size=batch_size, shuffle=False, num_workers=workers)
    target_device = choose_device(device)
    model = build_model(model_name, bands, dim, depth, heads, dropout, rgb_pretrained).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()
    output = Path(output) / model_name / task / f"seed_{seed}" / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pt"
    best_score = -np.inf
    stale = 0
    history = []
    config = {
        "model": model_name, "task": task, "seed": seed, "fold": fold, "bands": bands,
        "dim": dim, "depth": depth, "heads": heads, "dropout": dropout,
        "rgb_pretrained": rgb_pretrained, "parameters": trainable_parameters(model),
        "inner_val_fraction": inner_val_fraction,
    }
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for rgb, hsi, labels, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(rgb.to(target_device), hsi.to(target_device))
            loss = loss_function(logits, labels.to(target_device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_predictions = predict(model, val_loader, target_device, model_name, task, seed, fold)
        score = float(classification_metrics(plant_predictions(validation_predictions))["balanced_accuracy"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "plant_balanced_accuracy": score})
        if score > best_score:
            best_score, stale = score, 0
            torch.save({"model_state": model.state_dict(), "config": config, "band_stats": stats.as_dict()}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    saved = torch.load(checkpoint, map_location=target_device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    predict(model, test_loader, target_device, model_name, task, seed, fold).to_csv(output / "predictions.csv", index=False)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "run.json").write_text(json.dumps({**config, "best_balanced_accuracy": best_score}, indent=2), encoding="utf-8")
    return output / "predictions.csv"


def run_matrix(
    task_dir: str | Path,
    output: str | Path,
    tasks: list[str],
    models: list[str],
    seeds: list[int],
    folds: list[int],
    force: bool = False,
    **kwargs,
) -> list[Path]:
    predictions = []
    for task in tasks:
        for model in models:
            for seed in seeds:
                for fold in folds:
                    expected = Path(output) / model / task / f"seed_{seed}" / f"fold_{fold}" / "predictions.csv"
                    if expected.is_file() and not force:
                        print(f"resume: keeping {expected}")
                        predictions.append(expected)
                    else:
                        predictions.append(train_fold(Path(task_dir) / f"{task}.csv", model, seed, fold, output, **kwargs))
    manifest = Path(output) / "prediction_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(map(str, predictions)) + "\n", encoding="utf-8")
    return predictions
