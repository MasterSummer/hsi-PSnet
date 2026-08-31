from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def plant_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "task", "seed", "plant_id", "label", "prob_infected"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction files are missing columns: {sorted(missing)}")
    keys = ["model", "task", "seed", "plant_id", "label"]
    output = frame.groupby(keys, as_index=False)["prob_infected"].mean()
    output["prediction"] = (output["prob_infected"] >= 0.5).astype(int)
    return output


def classification_metrics(frame: pd.DataFrame) -> dict[str, float | int | list[list[int]]]:
    y = frame["label"].to_numpy(dtype=int)
    probability = frame["prob_infected"].to_numpy(dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result: dict[str, float | int | list[list[int]]] = {
        "n_plants": int(len(frame)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "auroc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return result


def stratified_bootstrap(
    frame: pd.DataFrame, repeats: int = 2000, seed: int = 157
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    groups = [group.index.to_numpy() for _, group in frame.groupby("label")]
    values: dict[str, list[float]] = {}
    for _ in range(repeats):
        sampled = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in groups])
        current = classification_metrics(frame.loc[sampled])
        for key in ("accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity", "f1", "auroc"):
            values.setdefault(key, []).append(float(current[key]))
    return {
        key: (float(np.nanpercentile(value, 2.5)), float(np.nanpercentile(value, 97.5)))
        for key, value in values.items()
    }


def evaluate_predictions(
    inputs: list[str | Path], output: str | Path, bootstrap: int = 2000, seed: int = 157
) -> None:
    sample = pd.concat([pd.read_csv(path) for path in inputs], ignore_index=True)
    plants = plant_predictions(sample)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plants.to_csv(output / "plant_predictions.csv", index=False)

    run_rows: list[dict[str, object]] = []
    detailed: dict[str, object] = {}
    metric_names = ("accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity", "f1", "auroc")
    for (model, task, run_seed), group in plants.groupby(["model", "task", "seed"]):
        point = classification_metrics(group)
        ci = stratified_bootstrap(group, bootstrap, seed + int(run_seed))
        row: dict[str, object] = {"model": model, "task": task, "seed": int(run_seed), **point}
        for name in metric_names:
            row[f"{name}_ci_low"] = ci[name][0]
            row[f"{name}_ci_high"] = ci[name][1]
        run_rows.append(row)
        detailed[f"{model}/{task}/seed_{run_seed}"] = row
    runs = pd.DataFrame(run_rows)
    runs.to_csv(output / "metrics_by_seed.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for (model, task), group in runs.groupby(["model", "task"]):
        row: dict[str, object] = {"model": model, "task": task, "seeds": int(len(group))}
        for name in metric_names:
            row[f"{name}_mean"] = float(group[name].mean())
            row[f"{name}_std"] = float(group[name].std(ddof=1)) if len(group) > 1 else 0.0
        for name in ("tn", "fp", "fn", "tp"):
            row[f"{name}_sum"] = int(group[name].sum())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output / "metrics_summary.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(detailed, indent=2, allow_nan=True), encoding="utf-8")

    paired_rows: list[dict[str, object]] = []
    reference = plants[plants["model"] == "psnet_full"]
    for (task, run_seed), ref in reference.groupby(["task", "seed"]):
        for model, candidate in plants[(plants.task == task) & (plants.seed == run_seed)].groupby("model"):
            if model == "psnet_full":
                continue
            merged = ref.merge(candidate, on=["task", "seed", "plant_id", "label"], suffixes=("_ref", "_candidate"))
            if len(merged) != len(ref):
                continue
            ref_frame = merged[["label", "prob_infected_ref"]].rename(columns={"prob_infected_ref": "prob_infected"})
            candidate_frame = merged[["label", "prob_infected_candidate"]].rename(columns={"prob_infected_candidate": "prob_infected"})
            ref_metrics = classification_metrics(ref_frame)
            candidate_metrics = classification_metrics(candidate_frame)
            paired_rows.append({
                "task": task, "seed": int(run_seed), "reference": "psnet_full", "candidate": model,
                "n_common_plants": len(merged),
                "balanced_accuracy_difference": float(ref_metrics["balanced_accuracy"]) - float(candidate_metrics["balanced_accuracy"]),
                "auroc_difference": float(ref_metrics["auroc"]) - float(candidate_metrics["auroc"]),
            })
    pd.DataFrame(paired_rows).to_csv(output / "paired_model_differences.csv", index=False)
