# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from matplotlib import cm

from Data_Loader import MultiModalDataset
from model import MultiModalNet


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


def build_transforms(rgb_paths, x_trainval):
    mean_hsi = x_trainval.mean(dim=(0, 2, 3))
    std_hsi = x_trainval.std(dim=(0, 2, 3))
    transform_hsi = transforms.Normalize(mean=mean_hsi.tolist(), std=std_hsi.tolist())

    rgb_mean, rgb_std = compute_rgb_stats(rgb_paths)
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])
    return transform_rgb, transform_hsi


def get_resize_transform(transform_rgb):
    if isinstance(transform_rgb, transforms.Compose):
        for t in transform_rgb.transforms:
            if isinstance(t, transforms.Resize):
                return t
    return None


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, rgb_tensor, hsi_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(rgb_tensor, hsi_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        grads = self.gradients
        activs = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activs).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=rgb_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_heatmap(rgb_pil, cam, alpha=0.5):
    heatmap = cm.get_cmap("jet")(cam)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap).resize(rgb_pil.size, resample=Image.BILINEAR)
    return Image.blend(rgb_pil, heatmap_pil, alpha=alpha)


def save_grid(image_paths, save_path, cols=5, pad=12, bg_color=(255, 255, 255)):
    if not image_paths:
        return
    if cols is None or cols <= 0:
        cols = min(5, len(image_paths))
    images = [Image.open(p).convert("RGB") for p in image_paths]
    widths, heights = zip(*(im.size for im in images))
    cell_w, cell_h = max(widths), max(heights)
    rows = int(np.ceil(len(images) / cols))
    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), bg_color)
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        grid.paste(im, (x, y))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid.save(save_path)


def run_gradcam(
    model,
    test_dataset,
    rgb_paths,
    output_dir,
    transform_rgb,
    device,
    num_samples=None,
):
    os.makedirs(output_dir, exist_ok=True)
    resize_t = get_resize_transform(transform_rgb)
    total = len(test_dataset)
    if num_samples is None or num_samples >= total:
        indices = list(range(total))
    else:
        indices = list(range(num_samples))

    layers = {
        "layer4": model.rgb_encoder.resnet34.layer4,
        "layer3": model.rgb_encoder.resnet34.layer3,
    }

    for layer_name, target_layer in layers.items():
        cam_extractor = GradCAM(model, target_layer)
        overlay_paths = []
        for idx in indices:
            rgb_tensor, hsi_tensor, label = test_dataset[idx]
            rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
            hsi_tensor = hsi_tensor.unsqueeze(0).to(device)

            cam, pred = cam_extractor(rgb_tensor, hsi_tensor, class_idx=None)

            rgb_pil = Image.open(rgb_paths[idx]).convert("RGB")
            if resize_t is not None:
                rgb_pil = resize_t(rgb_pil)
            overlay = overlay_heatmap(rgb_pil, cam, alpha=0.5)

            plant_id = os.path.splitext(os.path.basename(rgb_paths[idx]))[0]
            filename = f"{plant_id}_true{int(label)}_pred{int(pred)}_{layer_name}.png"
            save_path = os.path.join(output_dir, filename)
            overlay.save(save_path)
            overlay_paths.append(save_path)

        cam_extractor.remove()
        grid_path = os.path.join(output_dir, f"gradcam_grid_{layer_name}.png")
        save_grid(overlay_paths, grid_path, cols=min(5, len(overlay_paths)))


def evaluate_accuracy(model, loader, device, mask_bands=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for rgb, hsi, labels in loader:
            rgb = rgb.to(device)
            hsi = hsi.to(device)
            labels = labels.to(device)
            if mask_bands is not None:
                hsi[:, mask_bands, :, :] = 0.0
            outputs = model(rgb, hsi)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def run_band_masking(model, test_loader, device, output_dir, group_size=10):
    os.makedirs(output_dir, exist_ok=True)
    baseline_acc = evaluate_accuracy(model, test_loader, device, mask_bands=None)
    first_batch = next(iter(test_loader))
    num_bands = first_batch[1].shape[1]

    results = []
    for start in tqdm(range(0, num_bands, group_size), desc="Masking band groups"):
        end = min(start + group_size, num_bands)
        mask_bands = list(range(start, end))
        acc = evaluate_accuracy(model, test_loader, device, mask_bands=mask_bands)
        drop = baseline_acc - acc
        results.append((start, end - 1, acc, drop))

    results.sort(key=lambda x: x[2], reverse=True)

    csv_path = os.path.join(output_dir, "band_masking_results.csv")
    with open(csv_path, "w") as f:
        f.write("band_start,band_end,accuracy,drop\n")
        for start, end, acc, drop in results:
            f.write(f"{start},{end},{acc:.6f},{drop:.6f}\n")

    top_txt = os.path.join(output_dir, "top_bands.txt")
    with open(top_txt, "w") as f:
        f.write(f"Baseline accuracy: {baseline_acc:.6f}\n")
        f.write("Top 10 band groups by accuracy drop:\n")
        for start, end, acc, drop in results[:10]:
            f.write(f"bands {start}-{end}: acc={acc:.6f}, drop={drop:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM and band masking experiments.")
    parser.add_argument("--weights", default="saved_models/MultiModalNet_fold4_best.pth")
    parser.add_argument("--trainval", default="splits/trainval.pt")
    parser.add_argument("--test", default="splits/cleantest.pt")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-gradcam", default="outputs/explain/gradcam")
    parser.add_argument("--output-mask", default="outputs/explain/mask_band")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    trainval_pairs = torch.load(args.trainval)
    test_pairs = torch.load(args.test)

    rgb_trainval = [p[0] for p in trainval_pairs]
    x_trainval = torch.stack([p[1] for p in trainval_pairs])

    rgb_test = [p[0] for p in test_pairs]
    x_test = torch.stack([p[1] for p in test_pairs])
    y_test = torch.tensor([p[2] for p in test_pairs])

    transform_rgb, transform_hsi = build_transforms(rgb_trainval, x_trainval)

    test_dataset = MultiModalDataset(
        rgb_test, x_test, y_test,
        transform_rgb=transform_rgb,
        transform_hsi=transform_hsi,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activation = nn.LeakyReLU()
    token_dim = 64
    dim_head = 4
    model = MultiModalNet(
        fuse_method="concat",
        dropout=0.5,
        heads=token_dim // dim_head,
        dim_head=dim_head,
        token_dim=token_dim,
        mlp_dim=512,
        activation=activation,
        c=313,
    ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    run_gradcam(
        model=model,
        test_dataset=test_dataset,
        rgb_paths=rgb_test,
        output_dir=args.output_gradcam,
        transform_rgb=transform_rgb,
        device=device,
        num_samples=args.num_samples,
    )

    run_band_masking(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_mask,
        group_size=10,
    )


if __name__ == "__main__":
    main()
