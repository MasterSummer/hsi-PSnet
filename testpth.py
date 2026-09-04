#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from Data_Loader import MultiModalDataset
from model import MultiModalNet
from train import evaluate


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_rgb_stats(paths, sample_size=200):
    sample_paths = random.sample(paths, min(sample_size, len(paths)))
    pixels = []
    for path in sample_paths:
        img = Image.open(path).convert("RGB")
        img = transforms.Resize((224, 224))(img)
        img = transforms.ToTensor()(img)
        pixels.append(img)
    pixels = torch.stack(pixels)
    return pixels.mean(dim=(0, 2, 3)), pixels.std(dim=(0, 2, 3))


def build_test_loader(trainval_pt: str, test_pt: str, batch_size: int):
    trainval_pairs = torch.load(trainval_pt)
    test_pairs = torch.load(test_pt)

    rgb_trainval = [p[0] for p in trainval_pairs]
    x_trainval = torch.stack([p[1] for p in trainval_pairs])

    rgb_test = [p[0] for p in test_pairs]
    x_test = torch.stack([p[1] for p in test_pairs])
    y_test = torch.tensor([p[2] for p in test_pairs])

    mean_hsi = x_trainval.mean(dim=(0, 2, 3))
    std_hsi = x_trainval.std(dim=(0, 2, 3))
    transform_hsi = transforms.Normalize(mean=mean_hsi.tolist(), std=std_hsi.tolist())

    rgb_mean, rgb_std = compute_rgb_stats(rgb_trainval)
    transform_rgb = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ]
    )

    test_dataset = MultiModalDataset(
        rgb_files=rgb_test,
        hsi_data=x_test,
        labels=y_test,
        transform_rgb=transform_rgb,
        transform_hsi=transform_hsi,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def build_model():
    # Must match MultiModalNet_NoPretrain config used in fold.py.
    token_dim = 64
    dim_head = 4
    heads = token_dim // dim_head
    return MultiModalNet(
        fuse_method="concat",
        dropout=0.5,
        heads=heads,
        dim_head=dim_head,
        token_dim=token_dim,
        mlp_dim=512,
        activation=torch.nn.LeakyReLU(),
        c=313,
        rgb_pretrained=True,
        use_caf=True,
        use_wavelet=True,
        use_3d_patch=True,
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved MultiModalNet_NoPretrain fold checkpoints on test set."
    )
    parser.add_argument("--trainval", default="split_4re/trainval.pt")
    parser.add_argument("--test", default="split_4re/cleantest.pt")
    parser.add_argument("--checkpoints-dir", default="saved_models")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=5)
    parser.add_argument("--seed", type=int, default=157)
    parser.add_argument("--output-dir", default="results/test_saved_multimodal_folds")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Healthy", "2dpi", "4dpi", "6dpi"]
    test_loader = build_test_loader(args.trainval, args.test, args.batch_size)

    summary = {}
    for fold in range(args.fold_start, args.fold_end + 1):
        ckpt_name = f"MultiModalNet_fold{fold}_best.pth"
        ckpt_path = os.path.join(args.checkpoints_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[Skip] checkpoint not found: {ckpt_path}")
            continue

        model = build_model().to(device)
        load_checkpoint(model, ckpt_path, device)
        print(f"[Eval] fold={fold} checkpoint={ckpt_path}")

        fold_save_dir = os.path.join(args.output_dir, f"fold{fold}")
        metrics = evaluate(model, test_loader, class_names=class_names, save_dir=fold_save_dir, device=device)
        summary[f"fold{fold}"] = {
            "test_accuracy": metrics.get("test_accuracy"),
            "test_f1_weighted": metrics.get("test_f1_weighted"),
            "test_binary_accuracy": metrics.get("test_binary_accuracy"),
            "test_binary_f1": metrics.get("test_binary_f1"),
            "test_auroc_ovr_macro": metrics.get("test_auroc_ovr_macro"),
            "test_binary_auroc": metrics.get("test_binary_auroc"),
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Done] summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
