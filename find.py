#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import re
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 目标折结果（按 Fold1~Fold5 顺序）
TARGET_FOLDS = [
    {
        "test_accuracy": 0.9219,
        "test_auroc_ovr_macro": 0.9912,
        "test_f1_weighted": 0.9221,
        "test_binary_accuracy": 0.9531,
        "test_binary_f1": 0.9677,
    },
    {
        "test_accuracy": 0.8594,
        "test_auroc_ovr_macro": 0.9857,
        "test_f1_weighted": 0.8587,
        "test_binary_accuracy": 0.9844,
        "test_binary_f1": 0.9895,
    },
    {
        "test_accuracy": 0.9062,
        "test_auroc_ovr_macro": 0.9896,
        "test_f1_weighted": 0.9066,
        "test_binary_accuracy": 0.9688,
        "test_binary_f1": 0.9796,
    },
    {
        "test_accuracy": 0.9219,
        "test_auroc_ovr_macro": 0.9844,
        "test_f1_weighted": 0.9221,
        "test_binary_accuracy": 0.9688,
        "test_binary_f1": 0.9796,
    },
    {
        "test_accuracy": 0.8906,
        "test_auroc_ovr_macro": 0.9880,
        "test_f1_weighted": 0.8882,
        "test_binary_accuracy": 1.0000,
        "test_binary_f1": 1.0000,
    },
]

EVAL_DIR_RE = re.compile(r"^eval_(\d{8}_\d{6})$")
DEFAULT_CLASS_NAMES = ["Healthy", "2dpi", "4dpi", "6dpi"]


def load_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def folder_time_key(eval_dir: Path):
    m = EVAL_DIR_RE.match(eval_dir.name)
    return m.group(1) if m else ""


def collect_eval_items(root: Path):
    items = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not EVAL_DIR_RE.match(d.name):
            continue
        metrics_path = d / "test_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = load_metrics(metrics_path)
        except Exception:
            continue
        items.append(
            {
                "dir": d,
                "time_key": folder_time_key(d),
                "metrics": metrics,
            }
        )
    items.sort(key=lambda x: x["time_key"])
    return items


def normalize_from_time(from_time: str) -> str:
    """
    Accept:
    - YYYY-MM-DD
    - YYYYMMDD
    - YYYYMMDD_HHMMSS
    Return normalized key: YYYYMMDD_HHMMSS
    """
    s = from_time.strip()
    if re.fullmatch(r"\d{8}_\d{6}", s):
        return s
    if re.fullmatch(r"\d{8}", s):
        return f"{s}_000000"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s.replace("-", "") + "_000000"
    raise ValueError(f"不支持的时间格式: {from_time}")


def fold_match(metrics: dict, target: dict, tol: float, acc_only: bool):
    keys = ["test_accuracy"] if acc_only else list(target.keys())
    for k in keys:
        if k not in metrics:
            return False
        if not approx_equal(metrics[k], target[k], tol):
            return False
    return True


def find_sequences(items, tol: float, acc_only: bool):
    n = len(TARGET_FOLDS)
    matches = []
    for i in range(0, len(items) - n + 1):
        window = items[i : i + n]
        ok = True
        for j in range(n):
            if not fold_match(window[j]["metrics"], TARGET_FOLDS[j], tol, acc_only):
                ok = False
                break
        if ok:
            matches.append(window)
    return matches


def print_match(window, acc_only: bool):
    print("-" * 80)
    print("找到 1 组连续 5 个 eval：")
    for idx, item in enumerate(window, start=1):
        m = item["metrics"]
        if acc_only:
            print(
                f"Fold {idx}: {item['dir']} | "
                f"Acc={m.get('test_accuracy'):.4f}"
            )
        else:
            print(
                f"Fold {idx}: {item['dir']} | "
                f"Acc={m.get('test_accuracy'):.4f}, "
                f"AUROC={m.get('test_auroc_ovr_macro'):.4f}, "
                f"F1={m.get('test_f1_weighted'):.4f}, "
                f"Bin_Acc={m.get('test_binary_accuracy'):.4f}, "
                f"Bin_F1={m.get('test_binary_f1'):.4f}"
            )


def compute_confusion_from_eval_dirs(eval_dirs, class_names):
    all_true = []
    all_pred = []
    for d in eval_dirs:
        pred_csv = d / "test_predictions.csv"
        if not pred_csv.exists():
            raise FileNotFoundError(f"缺少 test_predictions.csv: {pred_csv}")
        with pred_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_true.append(int(row["true_label"]))
                all_pred.append(int(row["predicted_label"]))

    if not all_true:
        raise RuntimeError("test_predictions.csv 中没有有效样本。")

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(all_true, all_pred):
        if t < 0 or t >= n_classes or p < 0 or p >= n_classes:
            raise ValueError(f"发现越界标签: true={t}, pred={p}, n_classes={n_classes}")
        cm[t, p] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(np.float64) / np.maximum(row_sum, 1)
    return cm, cm_norm


def save_matrix_csv(path: Path, matrix: np.ndarray):
    np.savetxt(path, matrix, delimiter=",", fmt="%.6f")


def plot_matrix(path: Path, matrix: np.ndarray, title: str, fmt: str, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    h, w = matrix.shape
    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))
    denom = max(vmax - vmin, 1e-12)
    for i in range(h):
        for j in range(w):
            val = float(matrix[i, j])
            # Use normalized intensity threshold for stable contrast.
            val_norm = (val - vmin) / denom
            text_color = "white" if val_norm >= 0.45 else "black"
            ax.text(
                j,
                i,
                format(matrix[i, j], fmt),
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                bbox=dict(facecolor=(1, 1, 1, 0.15) if text_color == "black" else (0, 0, 0, 0.15), edgecolor="none", pad=1.0),
            )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_counts_matrix(path: Path, cm: np.ndarray, class_names):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 12},
    )
    ax.set_title("Summed Confusion Matrix (5-folds)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.text(
        0.99,
        0.01,
        "v2",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_sequence_artifacts(root: Path, window, idx: int, class_names):
    out_dir = root / "_matched_sequences" / f"seq_{idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    names_path = out_dir / "matched_eval_dirs.txt"
    with names_path.open("w", encoding="utf-8") as f:
        f.write(f"class_names: {','.join(class_names)}\n")
        for i, item in enumerate(window, 1):
            f.write(f"Fold {i}: {item['dir']}\n")

    eval_dirs = [item["dir"] for item in window]
    cm, cm_norm = compute_confusion_from_eval_dirs(eval_dirs, class_names=class_names)
    save_matrix_csv(out_dir / "cm_5fold_counts.csv", cm)
    save_matrix_csv(out_dir / "cm_5fold_row_normalized.csv", cm_norm)
    plot_counts_matrix(out_dir / "cm_5fold_counts.png", cm, class_names=class_names)
    plot_counts_matrix(out_dir / "cm_5fold_counts_v2.png", cm, class_names=class_names)
    plot_matrix(
        out_dir / "cm_5fold_row_normalized.png",
        cm_norm,
        "Summed Confusion Matrix (5-folds, row-normalized)",
        ".2f",
        class_names=class_names,
    )

    payload = {
        "sequence_index": idx,
        "eval_dirs": [str(x) for x in eval_dirs],
        "counts_csv": str(out_dir / "cm_5fold_counts.csv"),
        "norm_csv": str(out_dir / "cm_5fold_row_normalized.csv"),
        "counts_png": str(out_dir / "cm_5fold_counts.png"),
        "norm_png": str(out_dir / "cm_5fold_row_normalized.png"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="按时间顺序查找连续 5 个 eval_*/test_metrics.json，并匹配指定 Fold 指标。"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results",
        help="包含 eval_YYYYMMDD_HHMMSS 子目录的根目录（默认: results）",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="浮点比较容差（默认: 1e-4）",
    )
    parser.add_argument(
        "--acc-only",
        action="store_true",
        help="仅按 test_accuracy 匹配（默认匹配 Acc/AUROC/F1/Bin_Acc/Bin_F1）",
    )
    parser.add_argument(
        "--from-time",
        type=str,
        default=None,
        help="仅保留 >= 该时间的 eval。支持 YYYY-MM-DD / YYYYMMDD / YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="导出命中的5个eval名字和5-fold汇总混淆矩阵。",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="Healthy,2dpi,4dpi,6dpi",
        help="类别名，逗号分隔，顺序必须与标签0..N-1一致。",
    )
    args = parser.parse_args()
    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    if not class_names:
        class_names = DEFAULT_CLASS_NAMES

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"根目录不存在或不是目录: {root}")

    items = collect_eval_items(root)
    if args.from_time:
        start_key = normalize_from_time(args.from_time)
        items = [x for x in items if x["time_key"] >= start_key]

    if len(items) < 5:
        raise SystemExit(f"有效 eval 目录不足 5 个，当前仅 {len(items)} 个: {root}")

    matches = find_sequences(items, tol=args.tol, acc_only=args.acc_only)
    if not matches:
        print("未找到符合条件的连续 5 个 eval。")
        return

    print(f"共找到 {len(matches)} 组匹配。")
    for i, w in enumerate(matches, 1):
        print_match(w, acc_only=args.acc_only)
        if args.export:
            out_dir = export_sequence_artifacts(root, w, i, class_names=class_names)
            print(f"已导出: {out_dir}")


if __name__ == "__main__":
    main()
