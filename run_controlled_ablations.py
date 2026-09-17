"""Run pooled-date PSNet ablations with one seed and five plant-disjoint folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd
import torch
import torchvision

from reviewer_experiments.core import prepare_tasks
from reviewer_experiments.ingest import build_split4re_metadata
from reviewer_experiments.metrics import evaluate_predictions
from reviewer_experiments.models import build_model, trainable_parameters
from reviewer_experiments.train import run_matrix
from reviewer_experiments.bands import parse_bands
from reviewer_experiments.data import load_hsi, resolve_path
from reviewer_experiments.models import MODEL_NAMES


MODELS = ["psnet_full", "psnet_raw_spectrum_token", "psnet_caf_self",
          "psnet_raw_token_caf_self", "psnet_mean_depth_patch"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_predictions(paths, task_dir):
    completed = {}
    for path in paths:
        predicted = pd.read_csv(path, dtype={"sample_id": str, "plant_id": str})
        fold = int(path.parent.name.split("_")[-1])
        task, model = path.parents[2].name, path.parents[3].name
        seed = int(path.parents[1].name.split("_")[-1])
        key = (model, task, seed)
        if fold in completed.setdefault(key, set()):
            raise ValueError(f"Repeated fold in evaluation inputs: {key}, fold {fold}")
        completed[key].add(fold)
        frame = pd.read_csv(task_dir / f"{task}.csv", dtype={"sample_id": str, "plant_id": str})
        expected = frame[frame.fold == fold].set_index("sample_id").sort_index()
        if predicted.sample_id.duplicated().any():
            raise ValueError(f"Duplicate predictions: {path}")
        predicted = predicted.set_index("sample_id").sort_index()
        if not predicted.index.equals(expected.index):
            raise ValueError(f"Incomplete/unexpected test observations: {path}")
        for column in ["plant_id", "label"]:
            if not predicted[column].equals(expected[column]):
                raise ValueError(f"Prediction metadata mismatch ({column}): {path}")
        for column, value in [("model", model), ("task", task), ("seed", seed), ("fold", fold)]:
            if not predicted[column].eq(value).all():
                raise ValueError(f"Prediction provenance mismatch ({column}): {path}")
        if not np.isfinite(predicted.prob_infected).all() or not predicted.prob_infected.between(0, 1).all():
            raise ValueError(f"Invalid probabilities: {path}")
        if not path.with_name("best.pt").is_file():
            raise ValueError(f"Missing checkpoint: {path}")
    if not completed or any(folds != {1, 2, 3, 4, 5} for folds in completed.values()):
        raise ValueError("OOF evaluation requires all five folds for every model/task/seed")


def label_oof_evaluation(output, seed):
    """Label pooled OOF metrics and avoid inventing between-seed variance."""
    summary_path = output / "metrics_summary.csv"
    summary = pd.read_csv(summary_path)
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = np.nan
    summary["evaluation_design"] = "five_fold_plant_oof_single_seed"
    summary["n_folds"] = 5
    summary.to_csv(summary_path, index=False)
    points = pd.read_csv(output / "metrics_by_seed.csv")
    points["n_folds"] = 5
    points["evaluation_design"] = "five_fold_plant_oof_single_seed"
    points.to_csv(output / "oof_metrics.csv", index=False)
    note = dict(seed=seed, test_folds=[1, 2, 3, 4, 5], training_runs_per_model_task=5,
                evaluation_design="Five plant-disjoint outer folds, one training seed. Pool each observation's held-out prediction, then average probabilities over all available leaves and dates within each plant.",
                uncertainty="Plant bootstrap CI is conditional on the saved OOF predictions. Between-seed SD is unavailable, not zero; folds are not independent biological experiments.",
                comparison="All models use identical folds. all_dpi pools 2, 4 and 6 dpi as one binary inoculated-versus-mock task; dates are not separate classes or training tasks. This includes symptomatic observations.")
    (output / "evaluation_design.json").write_text(json.dumps(note, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-dir", type=Path, help="Prefer existing reviewer task CSVs to preserve identical folds")
    source.add_argument("--split-dir", type=Path, help="Read trainval.pt and cleantest.pt and generate task tables automatically")
    parser.add_argument("--rgb-root", type=Path, help="Repair old RGB paths when generating metadata from PT")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="all_dpi", help="Default: pool all dates into one binary task")
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model names")
    parser.add_argument("--bands", default="all", help="Original band positions, e.g. 0:104 or 0:100,200:313; stop excluded")
    parser.add_argument("--top-k-bands", type=int, help="Select the top K absolute group-mean differences using fitting plants only, separately in each fold")
    parser.add_argument("--seed", type=int, default=157, help="One training seed, default 157")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true", help="Freeze configuration and data splits without training")
    args = parser.parse_args()
    if args.top_k_bands is not None and (args.top_k_bands < 1 or args.bands != "all"):
        parser.error("--top-k-bands must be positive and cannot be combined with fixed --bands")
    tasks = args.tasks.split(",")
    model_names = args.models.split(",")
    if not model_names or not set(model_names) <= set(MODEL_NAMES) or len(set(model_names)) != len(model_names):
        parser.error("Models must be unique supported names")
    seeds = [args.seed]
    folds = [1, 2, 3, 4, 5]
    if not tasks or not set(tasks) <= {"dpi_2", "dpi_4", "dpi_6", "presymptomatic_2_4", "all_dpi"}:
        parser.error("Unsupported tasks")
    if len(set(tasks)) != len(tasks):
        parser.error("Tasks must be unique")
    if not 0 <= args.seed < 2**32 - 5:
        parser.error("Seed must be nonnegative and leave room for the fold offset")
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.workers < 0:
        parser.error("Invalid training budget")
    args.output = args.output.expanduser().resolve()
    manifest_path = args.output / "experiment_manifest.json"
    root = Path(__file__).resolve().parent
    sources = [Path(__file__).resolve(), *sorted((root / "reviewer_experiments").glob("*.py"))]
    config = dict(dim=64, depth=5, heads=4, dropout=.5, rgb_pretrained=True,
                  epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                  workers=args.workers, learning_rate=3e-4, weight_decay=1e-4,
                  inner_val_fraction=.2, device=args.device,
                  data_root=str(args.data_root.resolve()) if args.data_root else None)
    identity = dict(models=model_names, tasks=tasks, seeds=seeds, folds=folds, band_spec=args.bands, top_k_bands=args.top_k_bands,
                    evaluation_design="five_fold_plant_oof_single_seed",
                    training=config, source_sha256={str(p.relative_to(root)): sha256(p) for p in sources},
                    task_source=str((args.task_dir or args.split_dir).expanduser().resolve()),
                    source_mode="tasks" if args.task_dir else "split4re",
                    rgb_root=str(args.rgb_root.resolve()) if args.rgb_root else None,
                    versions=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__))
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous:
        if previous["identity"] != identity:
            raise ValueError("Configuration/source differs from existing run; use a new output directory")
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError("New experiment requires an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    task_dir = args.task_dir.expanduser().resolve() if args.task_dir else args.output / "tasks"
    if args.split_dir and not previous:
        metadata = args.output / "metadata.csv"
        info = build_split4re_metadata(args.split_dir, metadata, args.rgb_root)
        if info["missing_rgb"]:
            raise ValueError("Training needs RGB files. Supply --rgb-root to resolve the paths; use a new output directory after fixing")
        prepare_tasks(metadata, task_dir, n_splits=5, seed=157, require_files=True)
    task_hashes = {task: sha256(task_dir / f"{task}.csv") for task in tasks}
    for task in tasks:
        frame = pd.read_csv(task_dir / f"{task}.csv")
        if set(frame.fold) != {1, 2, 3, 4, 5} or frame.groupby("plant_id").fold.nunique().max() != 1:
            raise ValueError(f"Invalid five-fold plant grouping: {task}")
    if previous and previous["task_sha256"] != task_hashes:
        raise ValueError("Task tables changed since run was created")
    parameter_bands = 313
    if args.bands != "all" or args.top_k_bands is not None:
        first = pd.read_csv(task_dir / f"{tasks[0]}.csv").iloc[0]
        total_bands = load_hsi(resolve_path(first.hsi_path, args.data_root), first.get("hsi_layout", "CHW")).shape[0]
        if args.top_k_bands is not None:
            if args.top_k_bands > total_bands:
                raise ValueError("--top-k-bands exceeds input dimensions")
            config["top_k_bands"] = args.top_k_bands
            parameter_bands = args.top_k_bands
        else:
            config["band_indices"] = parse_bands(args.bands, total_bands)
            parameter_bands = len(config["band_indices"])
            if previous and previous.get("selected_band_indices") != config["band_indices"]:
                raise ValueError("Resolved band selection changed since run was created")
    if not previous:
        # Band selection is stored separately: identity stays comparable before data loading.
        identity["training"] = {k: v for k, v in config.items() if k not in {"band_indices", "top_k_bands"}}
        manifest = dict(identity=identity, task_sha256=task_hashes,
                        selected_band_indices=config.get("band_indices"),
                        note="Same protocol for all configurations; no hyperparameter search. Source task IDs require biological verification. Input hashes cover task tables, not referenced data contents.")
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        for path in sources:
            target = args.output / "source_snapshot" / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (args.output / "task_snapshot").mkdir()
        for task in tasks:
            shutil.copy2(task_dir / f"{task}.csv", args.output / "task_snapshot" / f"{task}.csv")
        counts = []
        for name in model_names:
            model = build_model(name, bands=parameter_bands, dim=64, depth=5, heads=4, dropout=.5, rgb_pretrained=False)
            counts.append(dict(model=name, bands=parameter_bands, registered_parameters=trainable_parameters(model)))
            del model
        pd.DataFrame(counts).to_csv(args.output / "parameter_counts.csv", index=False)
    print(f"{len(model_names)*len(tasks)*len(folds)} trainings, tasks={tasks}, bands={args.bands}, top_k={args.top_k_bands}, one seed ({args.seed}), five plant-disjoint folds; manifest: {manifest_path}", flush=True)
    if args.prepare_only:
        return
    paths = run_matrix(task_dir, args.output / "runs", tasks, model_names, seeds, folds, **config)
    verify_predictions(paths, task_dir)
    evaluate_predictions(paths, args.output / "evaluation", bootstrap=5000, seed=157)
    label_oof_evaluation(args.output / "evaluation", args.seed)


if __name__ == "__main__":
    main()
