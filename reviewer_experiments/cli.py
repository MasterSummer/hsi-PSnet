from __future__ import annotations

import argparse
from pathlib import Path

from .attribution import band_occlusion
from .audit import metadata_audit
from .core import TASK_DPI, prepare_tasks
from .metrics import evaluate_predictions
from .models import MODEL_NAMES, build_model, trainable_parameters
from .profile import profile_models
from .spectral import spectral_analysis
from .train import run_matrix, train_fold


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in csv_strings(value)]


def add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rgb-pretrained", action="store_true")
    parser.add_argument("--inner-val-fraction", type=float, default=0.2)


def training_kwargs(args) -> dict:
    return {name.replace("-", "_"): getattr(args, name.replace("-", "_")) for name in (
        "data-root", "epochs", "patience", "batch-size", "workers", "learning-rate",
        "weight-decay", "device", "dim", "depth", "heads", "dropout", "rgb-pretrained",
        "inner-val-fraction",
    )}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Reviewer-requested computational experiments")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="validate metadata and make plant-disjoint folds")
    prepare.add_argument("--metadata", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--folds", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=157)
    prepare.add_argument("--require-files", action="store_true")

    train = commands.add_parser("train", help="train one model/task/seed/fold")
    train.add_argument("--task-csv", required=True)
    train.add_argument("--model", required=True, choices=MODEL_NAMES)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--output", required=True)
    add_training_options(train)

    matrix = commands.add_parser("matrix", help="run a reproducible experiment matrix")
    matrix.add_argument("--task-dir", required=True)
    matrix.add_argument("--tasks", type=csv_strings, default=list(TASK_DPI))
    matrix.add_argument("--models", type=csv_strings, required=True)
    matrix.add_argument("--seeds", type=csv_ints, default=[157, 257, 357])
    matrix.add_argument("--folds", type=csv_ints, default=[1, 2, 3, 4, 5])
    matrix.add_argument("--output", required=True)
    add_training_options(matrix)

    evaluate = commands.add_parser("evaluate", help="plant aggregation, confidence intervals, raw confusion matrices")
    evaluate.add_argument("--inputs", nargs="+", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--bootstrap", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=157)

    spectral = commands.add_parser("spectral", help="within-DPI spectra, effect sizes, FDR and red edge")
    spectral.add_argument("--metadata", required=True)
    spectral.add_argument("--data-root")
    spectral.add_argument("--wavelengths")
    spectral.add_argument("--output", required=True)

    audit = commands.add_parser("audit", help="exact sample counts and acquisition-confound audit")
    audit.add_argument("--metadata", required=True)
    audit.add_argument("--output", required=True)

    occlusion = commands.add_parser("occlusion", help="grouped-band occlusion on a saved fold")
    occlusion.add_argument("--checkpoint", required=True)
    occlusion.add_argument("--task-csv", required=True)
    occlusion.add_argument("--data-root")
    occlusion.add_argument("--group-size", type=int, default=10)
    occlusion.add_argument("--batch-size", type=int, default=8)
    occlusion.add_argument("--workers", type=int, default=0)
    occlusion.add_argument("--device", default="auto")
    occlusion.add_argument("--output", required=True)

    params = commands.add_parser("params", help="report trainable parameter counts")
    params.add_argument("--bands", type=int, required=True)
    params.add_argument("--dim", type=int, default=128)
    params.add_argument("--depth", type=int, default=4)
    params.add_argument("--heads", type=int, default=4)

    profile = commands.add_parser("profile", help="parameter, inference-time and peak-memory costs")
    profile.add_argument("--models", type=csv_strings, default=list(MODEL_NAMES))
    profile.add_argument("--bands", type=int, required=True)
    profile.add_argument("--height", type=int, required=True)
    profile.add_argument("--width", type=int, required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--device", default="auto")
    profile.add_argument("--batch-size", type=int, default=1)
    profile.add_argument("--warmup", type=int, default=5)
    profile.add_argument("--repeats", type=int, default=20)
    profile.add_argument("--dim", type=int, default=128)
    profile.add_argument("--depth", type=int, default=4)
    profile.add_argument("--heads", type=int, default=4)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare_tasks(args.metadata, args.output, args.folds, args.seed, args.require_files)
    elif args.command == "train":
        train_fold(args.task_csv, args.model, args.seed, args.fold, args.output, **training_kwargs(args))
    elif args.command == "matrix":
        run_matrix(args.task_dir, args.output, args.tasks, args.models, args.seeds, args.folds, **training_kwargs(args))
    elif args.command == "evaluate":
        evaluate_predictions(args.inputs, args.output, args.bootstrap, args.seed)
    elif args.command == "spectral":
        spectral_analysis(args.metadata, args.output, args.data_root, args.wavelengths)
    elif args.command == "audit":
        metadata_audit(args.metadata, args.output)
    elif args.command == "occlusion":
        band_occlusion(args.checkpoint, args.task_csv, args.output, args.data_root, args.group_size, args.batch_size, args.workers, args.device)
    elif args.command == "params":
        for name in MODEL_NAMES:
            model = build_model(name, args.bands, args.dim, args.depth, args.heads)
            print(f"{name},{trainable_parameters(model)}")
    elif args.command == "profile":
        profile_models(args.models, args.bands, args.height, args.width, args.output, args.device,
                       args.batch_size, args.warmup, args.repeats, args.dim, args.depth, args.heads)


if __name__ == "__main__":
    main()
