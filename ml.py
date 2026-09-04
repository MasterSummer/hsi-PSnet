#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_metrics(y_true, y_pred, class_names):
    metrics = {}
    metrics["test_accuracy"] = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics["test_precision_weighted"] = p
    metrics["test_recall_weighted"] = r
    metrics["test_f1_weighted"] = f1
    metrics["test_auroc_ovr_macro"] = float("nan")

    _, recalls, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    for i, name in enumerate(class_names):
        metrics[f"test_acc_{name.replace(' ', '_').lower()}"] = float(recalls[i])

    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    bp, br, bf1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="binary", zero_division=0)
    metrics["test_binary_accuracy"] = accuracy_score(y_true_bin, y_pred_bin)
    metrics["test_binary_precision"] = bp
    metrics["test_binary_recall"] = br
    metrics["test_binary_f1"] = bf1
    metrics["test_binary_auroc"] = float("nan")
    return metrics


def summarize_fold_metrics(fold_metrics):
    out = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        out[f"mean_{key}"] = float(np.mean(values))
        out[f"std_{key}"] = float(np.std(values))
    return out


def write_result_block(output_path, model_name, params, fold_metrics, mean_metrics, class_names, cm, cm_norm):
    with open(output_path, "a+", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Params: {params}\n\n")
        f.write("[Overall 5-Fold Results (Mean ± Std)]\n")
        f.write(f"  - Accuracy:           {mean_metrics['mean_test_accuracy']:.4f} ± {mean_metrics['std_test_accuracy']:.4f}\n")
        f.write(f"  - AUROC (Macro OvR):  {mean_metrics['mean_test_auroc_ovr_macro']:.4f} ± {mean_metrics['std_test_auroc_ovr_macro']:.4f}\n")
        f.write(f"  - F1 (Weighted):      {mean_metrics['mean_test_f1_weighted']:.4f} ± {mean_metrics['std_test_f1_weighted']:.4f}\n")
        f.write(f"  - Precision:          {mean_metrics['mean_test_precision_weighted']:.4f} ± {mean_metrics['std_test_precision_weighted']:.4f}\n")
        f.write(f"  - Recall:             {mean_metrics['mean_test_recall_weighted']:.4f} ± {mean_metrics['std_test_recall_weighted']:.4f}\n\n")
        f.write("  [Per-class Accuracy (Recall)]\n")
        for name in class_names:
            key = f"test_acc_{name.replace(' ', '_').lower()}"
            f.write(f"    - {name}: {mean_metrics[f'mean_{key}']:.4f} ± {mean_metrics[f'std_{key}']:.4f}\n")
        f.write("\n")
        f.write("  [Binary Metrics (Healthy vs Diseased)]\n")
        f.write(f"    - Accuracy:  {mean_metrics['mean_test_binary_accuracy']:.4f} ± {mean_metrics['std_test_binary_accuracy']:.4f}\n")
        f.write(f"    - AUROC:     {mean_metrics['mean_test_binary_auroc']:.4f} ± {mean_metrics['std_test_binary_auroc']:.4f}\n")
        f.write(f"    - F1-score:  {mean_metrics['mean_test_binary_f1']:.4f} ± {mean_metrics['std_test_binary_f1']:.4f}\n")
        f.write(f"    - Precision: {mean_metrics['mean_test_binary_precision']:.4f} ± {mean_metrics['std_test_binary_precision']:.4f}\n")
        f.write(f"    - Recall:    {mean_metrics['mean_test_binary_recall']:.4f} ± {mean_metrics['std_test_binary_recall']:.4f}\n\n")
        f.write("[Detailed Fold-by-Fold Metrics]\n")
        for i, m in enumerate(fold_metrics, 1):
            f.write(
                f"  Fold {i}: Acc={m['test_accuracy']:.4f}, "
                f"F1={m['test_f1_weighted']:.4f}, "
                f"Bin_Acc={m['test_binary_accuracy']:.4f}, "
                f"Bin_F1={m['test_binary_f1']:.4f}\n"
            )
        f.write("\n[Confusion Matrix (Counts)]\n")
        for i, row in enumerate(cm):
            f.write("  " + class_names[i] + ": " + " ".join(f"{int(x):4d}" for x in row) + "\n")
        f.write("\n[Confusion Matrix (Row-normalized)]\n")
        for i, row in enumerate(cm_norm):
            f.write("  " + class_names[i] + ": " + " ".join(f"{x:.2f}" for x in row) + "\n")
        f.write("=" * 80 + "\n\n")


def main():
    parser = argparse.ArgumentParser(description="Run traditional ML baselines (PCA+SVM / PCA+RF) independently.")
    parser.add_argument("--trainval", default="split_4re/trainval.pt")
    parser.add_argument("--test", default="split_4re/cleantest.pt")
    parser.add_argument("--seed", type=int, default=157)
    parser.add_argument("--output", default="store/ml_baseline_results.txt")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    trainval_pairs = torch.load(args.trainval)
    test_pairs = torch.load(args.test)

    X_trainval = torch.stack([p[1] for p in trainval_pairs])
    y_trainval = torch.tensor([p[2] for p in trainval_pairs]).cpu().numpy()
    X_test = torch.stack([p[1] for p in test_pairs])
    y_test = torch.tensor([p[2] for p in test_pairs]).cpu().numpy()

    # Use image-level mean spectrum as baseline feature.
    X_trainval_feat = X_trainval.mean(dim=(2, 3)).cpu().numpy()
    X_test_feat = X_test.mean(dim=(2, 3)).cpu().numpy()

    class_names = ["Healthy", "2dpi", "4dpi", "6dpi"]
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    baselines = {
        "PCA + SVM": (SVC(C=10.0, kernel="rbf", gamma="scale"), {"feature": "HSI mean spectrum", "pca": "0.99 variance"}),
        "PCA + RF": (
            RandomForestClassifier(n_estimators=500, random_state=args.seed, n_jobs=-1),
            {"feature": "HSI mean spectrum", "pca": "0.99 variance", "n_estimators": 500},
        ),
    }

    for name, (clf, params) in baselines.items():
        fold_metrics = []
        all_true = []
        all_pred = []
        for train_idx, _ in skf.split(X_trainval_feat, y_trainval):
            X_train_fold = X_trainval_feat[train_idx]
            y_train_fold = y_trainval[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled = scaler.transform(X_test_feat)

            pca = PCA(n_components=0.99, svd_solver="full")
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            clf.fit(X_train_pca, y_train_fold)
            y_pred = clf.predict(X_test_pca)
            fold_metrics.append(compute_metrics(y_test, y_pred, class_names))
            all_true.extend(y_test.tolist())
            all_pred.extend(y_pred.tolist())

        mean_metrics = summarize_fold_metrics(fold_metrics)
        cm = confusion_matrix(all_true, all_pred, labels=list(range(len(class_names))))
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        write_result_block(args.output, name, params, fold_metrics, mean_metrics, class_names, cm, cm_norm)
        print(f"Done: {name}")

    print(f"Saved baseline report to {args.output}")


if __name__ == "__main__":
    main()
