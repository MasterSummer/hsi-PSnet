# Reviewer-requested computational experiments

This directory is a reproducible, leakage-resistant analysis pipeline for the major revision. It deliberately does **not** claim external/field generalization: all outer folds remain plant-disjoint but come from the same experimental run.

## What it adds

- Date-matched binary classification at 2, 4, and 6 dpi.
- A presymptomatic-only 2+4 dpi analysis and a separate all-dpi analysis including symptomatic 6 dpi.
- Nested plant-disjoint evaluation: an untouched outer fold for testing and a plant-disjoint inner split for early stopping.
- Plant-level probability aggregation, stratified plant bootstrap confidence intervals, multiple seeds, and raw TN/FP/FN/TP counts.
- Reviewer baselines: 1D spectral CNN, compact 3D CNN, RGB ResNet-18, simple multimodal fusion, learnable global token, and a plain architectural multimodal comparator.
- Ablations for CAF, wavelet token, and 3D patch embedding.
- Within-dpi plant spectra, 95% CIs, Hedges' g, Welch tests with Benjamini-Hochberg FDR, red-edge position/slope, and grouped-band occlusion.
- Exact sample/plant counts, acquisition-setting audit, parameter counts, inference latency, and peak CUDA memory.

The code never pools mock plants across dates for a dpi-specific comparison. Normalization statistics are estimated from the inner fitting plants only.

## Data contract

Copy `metadata_template.csv` and create one row per paired RGB-HSI observation. Required fields are:

`sample_id, plant_id, leaf_id, treatment, dpi, acquisition_date, imaging_session, rgb_path, hsi_path, symptom_status`

Use `treatment=mock` or `infected` and `dpi=2`, `4`, or `6`. Add `hsi_layout=HWC` or `CHW`. Strongly recommended optional columns are `exposure_ms` (or `exposure_us`), `gain`, and `white_reference_id`. Paths may be absolute or relative to `--data-root`. HSI files must be `.npy`, or a single-array `.npz`.

For red-edge calculations, supply a CSV with exactly two columns, `band,wavelength_nm`, one row per usable band.

For the original Run-1 filenames (`Plant7_Infected_Leaf3_Day2.RGB888`), build metadata with:

```bash
python -m reviewer_experiments.cli ingest-run1 \
  --rgb-root /absolute/path/to/Run\ 1 \
  --hsi-root /absolute/path/to/hsi_npy \
  --output /absolute/path/to/metadata.csv
```

The importer treats `.RGB888` files as the JPEG images they contain, matches HSI files by filename stem, and restores the 24 repeatedly measured mock plants from source number ranges 1-24, 25-48, and 49-72. It refuses to report multimodal readiness when any HSI file is missing.

If the server dataset is already stored as `split_4re/trainval.pt` and
`split_4re/cleantest.pt`, use the PT bundles directly without extracting or
duplicating the HSI tensors:

```bash
python -m reviewer_experiments.cli ingest-split4re \
  --split-dir split_4re \
  --output split_4re/metadata.csv
```

If the RGB paths embedded in the PT files belong to another machine, provide
the server directory containing the matching RGB files:

```bash
python -m reviewer_experiments.cli ingest-split4re \
  --split-dir split_4re \
  --rgb-root /absolute/server/path/to/RGB \
  --output split_4re/metadata.csv
```

The generated `ptbundle:` references read HSI tensors from `split_4re` in
place. The old train/test membership is retained only as provenance; the
reviewer analysis creates new plant-disjoint folds from all 96 biological
plants.

## Installation

```bash
cd /path/to/hsi-PSnet
python -m venv .venv
source .venv/bin/activate
python -m pip install -r reviewer_experiments/requirements.txt
```

## Exact experiment commands

Set these paths once:

```bash
export HSI_DATA_ROOT=/absolute/path/to/paired_dataset
export HSI_METADATA=/absolute/path/to/metadata.csv
export HSI_RESULTS=/absolute/path/to/reviewer_results
```

1. Validate metadata, audit acquisition confounds, and make fixed plant-level folds:

```bash
python -m reviewer_experiments.cli audit \
  --metadata "$HSI_METADATA" \
  --output "$HSI_RESULTS/metadata_audit"

python -m reviewer_experiments.cli prepare \
  --metadata "$HSI_METADATA" \
  --output "$HSI_RESULTS/tasks" \
  --folds 5 \
  --seed 157 \
  --require-files
```

2. Run the full requested model/ablation matrix. The command is intentionally explicit and will be computationally expensive: 10 models × 5 tasks × 3 seeds × 5 folds.

```bash
python -m reviewer_experiments.cli matrix \
  --task-dir "$HSI_RESULTS/tasks" \
  --tasks dpi_2,dpi_4,dpi_6,presymptomatic_2_4,all_dpi \
  --models psnet_full,psnet_no_caf,psnet_no_wavelet,psnet_no_3d_patch,psnet_learnable_cls,plain_multimodal,rgb_resnet18,spectral_1d_cnn,compact_3d_cnn,simple_multimodal \
  --seeds 157,257,357 \
  --folds 1,2,3,4,5 \
  --data-root "$HSI_DATA_ROOT" \
  --epochs 100 \
  --patience 12 \
  --batch-size 8 \
  --workers 4 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --inner-val-fraction 0.2 \
  --device cuda \
  --output "$HSI_RESULTS/runs"
```

For a one-fold smoke test before committing GPU time:

```bash
python -m reviewer_experiments.cli train \
  --task-csv "$HSI_RESULTS/tasks/dpi_2.csv" \
  --model spectral_1d_cnn \
  --seed 157 \
  --fold 1 \
  --data-root "$HSI_DATA_ROOT" \
  --epochs 2 \
  --patience 2 \
  --batch-size 8 \
  --device cuda \
  --output "$HSI_RESULTS/smoke_test"
```

3. Aggregate out-of-fold predictions by plant and compute CIs and confusion counts:

```bash
python -m reviewer_experiments.cli evaluate \
  --inputs $(find "$HSI_RESULTS/runs" -type f -name predictions.csv | sort) \
  --bootstrap 5000 \
  --seed 157 \
  --output "$HSI_RESULTS/evaluation"
```

The key outputs are `metrics_summary.csv`, `metrics_by_seed.csv`, `plant_predictions.csv`, `paired_model_differences.csv`, and `metrics.json`. The all-dpi versus presymptomatic-only binary result is the direct response to the reported 97.5% claim.

4. Run within-dpi spectral statistics and red-edge analysis:

```bash
python -m reviewer_experiments.cli spectral \
  --metadata "$HSI_METADATA" \
  --data-root "$HSI_DATA_ROOT" \
  --wavelengths /absolute/path/to/wavelengths.csv \
  --output "$HSI_RESULTS/spectral"
```

5. Run grouped-band occlusion on every outer-fold checkpoint. Example for one fold:

```bash
python -m reviewer_experiments.cli occlusion \
  --checkpoint "$HSI_RESULTS/runs/psnet_full/dpi_2/seed_157/fold_1/best.pt" \
  --task-csv "$HSI_RESULTS/tasks/dpi_2.csv" \
  --data-root "$HSI_DATA_ROOT" \
  --group-size 10 \
  --batch-size 8 \
  --device cuda \
  --output "$HSI_RESULTS/occlusion/psnet_full/dpi_2/seed_157/fold_1"
```

Repeat occlusion for every fold/seed used in the attribution figure and aggregate the resulting `band_occlusion.csv` files by wavelength group.

6. Report model size and measured computational cost on the actual GPU:

```bash
python -m reviewer_experiments.cli params --bands 313 > "$HSI_RESULTS/parameter_counts.csv"

python -m reviewer_experiments.cli profile \
  --models psnet_full,psnet_no_caf,psnet_no_wavelet,psnet_no_3d_patch,psnet_learnable_cls,plain_multimodal,rgb_resnet18,spectral_1d_cnn,compact_3d_cnn,simple_multimodal \
  --bands 313 \
  --height 132 \
  --width 135 \
  --batch-size 1 \
  --warmup 20 \
  --repeats 100 \
  --device cuda \
  --output "$HSI_RESULTS/computational_costs.csv"
```

Replace `313 × 132 × 135` with the final usable-band tensor shape if noisy bands are removed.

## Interpretation guardrails

- Report these as within-run cross-validated estimates, never as an external test or evidence of deployable generalization.
- Do not infer hardware properties such as wavelength calibration error, SNR, sensor linearity, registration accuracy, or repeatability from classification data. Those require calibration measurements. If they were not measured, state that transparently as a limitation.
- `plain_multimodal` is an architectural comparator using the same RGB/HSI widths and fusion head, but parameter counts will not be numerically identical after modules are removed. Report the generated parameter table and, if strict equality is required, adjust `--dim` before the definitive run.
- Keep every hyperparameter-search budget identical across comparison models and document any model-specific stability change.

## Tests

```bash
python -m unittest discover -s reviewer_experiments/tests -v
```
