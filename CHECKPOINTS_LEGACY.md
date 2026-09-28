# Legacy `.pth` Model Checkpoints

This retained guide describes the original `model.py` / `fold.py` workflow. For the revised binary experiments, use [RERUN_ZH.md](RERUN_ZH.md) and the `reviewer_experiments` package; checkpoint names and architectures differ.

This guide explains how `.pth` files are used in this project, how to create them, how to load them, and how to evaluate saved checkpoints.

In this repository, `.pth` files are PyTorch model checkpoints. They store trained model weights and related training metadata. They are not dataset files, and they are not Python `site-packages` `.pth` path configuration files.

## Environment Setup

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
cd /absolute/path/to/hsi-PSnet
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch and torchvision for your machine first. Choose the CPU or CUDA command that matches your system from the official PyTorch installation instructions:

```text
https://pytorch.org/get-started/locally/
```

Then install the remaining Python dependencies used by the training and evaluation scripts:

```bash
python -m pip install \
  numpy \
  scipy \
  scikit-learn \
  scikit-image \
  pillow \
  matplotlib \
  seaborn \
  pandas \
  tqdm \
  timm \
  einops \
  PyWavelets
```

The import name for `PyWavelets` is `pywt`, and the import name for `scikit-image` is `skimage`.

## Checkpoint Format

Training saves the best model from each fold under `saved_models/`. The filename pattern is usually:

```text
saved_models/{model_name}_fold{fold}_best.pth
```

Examples:

```text
saved_models/MultiModalNet_NoPretrain_fold1_best.pth
saved_models/MultiModalNet_NoPretrain_fold2_best.pth
saved_models/MultiModalNet_fold4_best.pth
```

The checkpoint is usually a dictionary with a `model_state_dict` entry:

```python
{
    "model_state_dict": best_model.state_dict(),
    "params": params,
}
```

`model_state_dict` contains the model weights. `params` records the training parameters used when the checkpoint was saved.

## Creating `.pth` Files

The training script `fold.py` saves the best checkpoint for each fold. Before running it, make sure the split files expected by the script exist:

```text
split_4re/trainval.pt
split_4re/cleantest.pt
```

Run training from the repository directory:

```bash
cd /absolute/path/to/hsi-PSnet
python fold.py
```

During training, the script creates `saved_models/` if needed and saves checkpoint files like:

```text
saved_models/MultiModalNet_NoPretrain_fold1_best.pth
```

Training can take a long time and writes new model and result files.

## Evaluating Saved Checkpoints

Use `test_saved_multimodal_folds.py` to evaluate saved `MultiModalNet_NoPretrain` fold checkpoints.

```bash
cd /absolute/path/to/hsi-PSnet
python test_saved_multimodal_folds.py \
  --trainval split_4re/trainval.pt \
  --test split_4re/cleantest.pt \
  --checkpoints-dir saved_models \
  --fold-start 1 \
  --fold-end 5
```

The script looks for checkpoint files with this naming rule:

```text
saved_models/MultiModalNet_NoPretrain_fold{fold}_best.pth
```

Evaluation output is written to:

```text
results/test_saved_multimodal_folds/
```

The `summary.json` file in that directory contains the main metrics for each evaluated fold.

## Loading a `.pth` File Manually

If you want to load a checkpoint in your own script, build the same model architecture that was used during training, then load the checkpoint with `torch.load(..., map_location=device)` and `model.load_state_dict(...)`.

```python
import torch

from model import MultiModalNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token_dim = 64
dim_head = 4
model = MultiModalNet(
    fuse_method="concat",
    dropout=0.5,
    heads=token_dim // dim_head,
    dim_head=dim_head,
    token_dim=token_dim,
    mlp_dim=512,
    activation=torch.nn.LeakyReLU(),
    c=313,
    rgb_pretrained=True,
    use_caf=True,
    use_wavelet=True,
    use_3d_patch=True,
).to(device)

checkpoint = torch.load(
    "saved_models/MultiModalNet_NoPretrain_fold1_best.pth",
    map_location=device,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()
```

The model class and constructor arguments must match the checkpoint. If they do not match, PyTorch may raise `Missing key(s)` or `Unexpected key(s)` errors.

## Common Issues

### `Missing key(s)` or `Unexpected key(s)`

This usually means the current model architecture does not match the architecture used when the checkpoint was saved. Check:

- The model class, such as `MultiModalNet`.
- Constructor arguments such as `token_dim`, `dim_head`, `dropout`, `use_caf`, `use_wavelet`, and `use_3d_patch`.
- The hyperspectral input channel count, such as `c=313`.

### `checkpoint not found`

Check that `--checkpoints-dir` points to the correct directory and that filenames match the rule used by `test_saved_multimodal_folds.py`:

```text
saved_models/MultiModalNet_NoPretrain_fold{fold}_best.pth
```

If your checkpoint filenames use a different pattern, rename the files or update the `ckpt_name` rule in the script.

### Loading a GPU-trained checkpoint on CPU

Use `map_location` when calling `torch.load`:

```python
checkpoint = torch.load(path, map_location=device)
```

If `device` is CPU, PyTorch maps the weights to CPU memory.
