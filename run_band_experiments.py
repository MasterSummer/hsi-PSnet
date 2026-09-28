"""Compare full and partial HSI inputs with identical plant-disjoint folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd

from reviewer_experiments.bands import parse_bands
from reviewer_experiments.core import prepare_tasks
from reviewer_experiments.data import load_hsi, resolve_path, _load_pt_bundle
from reviewer_experiments.ingest import build_split4re_metadata
from run_controlled_ablations import sha256


def band_sets(specs, total):
    if specs is None:
        specs = ["full=all"]
    result = {}
    for entry in specs:
        name, sep, spec = entry.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name in result:
            raise ValueError("Use unique safe names: --band-set NAME=0:104")
        result[name] = parse_bands(spec, total)
    return result


def experiment_sets(specs, top_k, total):
    if specs:
        return {name: dict(indices=indices, top_k=None, count=len(indices))
                for name, indices in band_sets(specs, total).items()}
    sizes = [int(value.strip()) for value in top_k.split(",")]
    if len(set(sizes)) != len(sizes) or any(k < 1 or k > total for k in sizes):
        raise ValueError("Top-K sizes must be unique positive integers within the input dimensions")
    result = {"full": dict(indices=list(range(total)), top_k=None, count=total)}
    result.update({f"top{k}": dict(indices=None, top_k=k, count=k) for k in sizes})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--split-dir", type=Path)
    source.add_argument("--task-dir", type=Path)
    p.add_argument("--rgb-root", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--band-set", action="append", help="Repeat NAME=all or NAME=start:stop[:step]; original zero-based indices")
    selection.add_argument("--top-k", default="3,10,30", help="Full-spectrum control plus fit-only mean-difference Top-K sizes (default: 3,10,30)")
    p.add_argument("--models", default="psnet_full")
    p.add_argument("--tasks", default="all_dpi")
    p.add_argument("--seed", type=int, default=157)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.expanduser().resolve()
    manifest_path = output / "band_experiment_manifest.json"
    identity = {k: str(v.expanduser().resolve()) if isinstance(v, Path) else v
                for k, v in vars(args).items() if k != "prepare_only"}
    identity["source_sha256"] = {str(path.relative_to(root)): sha256(path) for path in
        [Path(__file__).resolve(), root / "run_controlled_ablations.py", *sorted((root / "reviewer_experiments").glob("*.py"))]}
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous["identity"] != identity:
        raise ValueError("Configuration or source changed; choose a new output directory")
    if not previous and output.exists() and any(output.iterdir()):
        raise ValueError("New experiment requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    task_dir = args.task_dir.expanduser().resolve() if args.task_dir else output / "tasks"
    if args.split_dir and not previous:
        metadata = output / "metadata.csv"
        info = build_split4re_metadata(args.split_dir, metadata, args.rgb_root)
        if info["missing_rgb"]:
            raise ValueError("Missing RGB images; fix --rgb-root and retry in a new output directory")
        prepare_tasks(metadata, task_dir, n_splits=5, seed=157, require_files=True)
    task_names = args.tasks.split(",")
    task_hashes = {name: sha256(task_dir / f"{name}.csv") for name in task_names}
    first = pd.read_csv(task_dir / f"{task_names[0]}.csv").iloc[0]
    total = load_hsi(resolve_path(first.hsi_path, args.data_root), first.get("hsi_layout", "CHW")).shape[0]
    _load_pt_bundle.cache_clear()  # Do not retain a full PT bundle while child training runs.
    selections = experiment_sets(args.band_set, args.top_k, total)
    if previous and (previous["task_sha256"] != task_hashes or previous["band_sets"] != selections):
        raise ValueError("Task membership or selected bands changed since preparation")
    if not previous:
        manifest_path.write_text(json.dumps(dict(identity=identity, task_sha256=task_hashes,
            source_bands=total, band_sets=selections,
            note="Input positions, not assumed wavelengths. Identical outer folds and inner splits. Each Top-K ranking uses fitting plants only: absolute difference between treatment group means of per-plant spatial mean spectra, pooled over available observations. Validation and test plants are excluded."), indent=2)+"\n")
    print(f"Band sets: { {name: selection['count'] for name, selection in selections.items()} }; "
          f"{len(selections)*len(args.models.split(','))*len(task_names)*5} fold trainings", flush=True)
    summaries = []
    for name, selection in selections.items():
        destination = output / name
        command = [sys.executable, str(root / "run_controlled_ablations.py"),
                   "--task-dir", str(task_dir), "--output", str(destination),
                   "--models", args.models, "--tasks", args.tasks,
                   "--seed", str(args.seed), "--device", args.device, "--workers", str(args.workers),
                   "--epochs", str(args.epochs), "--patience", str(args.patience), "--batch-size", str(args.batch_size)]
        if selection["top_k"] is not None:
            command.extend(["--top-k-bands", str(selection["top_k"])])
        else:
            command.extend(["--bands", ",".join(map(str, selection["indices"]))])
        if args.data_root:
            command.extend(["--data-root", str(args.data_root)])
        if args.prepare_only:
            command.append("--prepare-only")
        print(f"Running band set {name}", flush=True)
        subprocess.run(command, check=True)
        if not args.prepare_only:
            summary = pd.read_csv(destination / "evaluation/oof_metrics.csv")
            summary.insert(0, "band_set", name)
            summary.insert(1, "n_input_bands", selection["count"])
            summaries.append(summary)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(output / "band_comparison.csv", index=False)
        print(f"Comparison: {output / 'band_comparison.csv'}")


if __name__ == "__main__":
    main()
