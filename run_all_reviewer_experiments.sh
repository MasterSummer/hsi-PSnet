#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_ROOT}/split_4re}"
RGB_ROOT="${RGB_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/reviewer_results}"
WAVELENGTHS_CSV="${WAVELENGTHS_CSV:-${SPLIT_DIR}/wavelengths.csv}"
DEVICE="${DEVICE:-cuda}"
TASKS="${TASKS:-dpi_2,dpi_4,dpi_6,presymptomatic_2_4,all_dpi}"
MODELS="${MODELS:-psnet_full,psnet_no_caf,psnet_no_wavelet,psnet_no_3d_patch,psnet_learnable_cls,plain_multimodal,rgb_resnet18,rgb_resnet34,hsi_transformer,spectral_1d_cnn,compact_3d_cnn,simple_multimodal}"
SEEDS="${SEEDS:-157,257,357}"
FOLDS="${FOLDS:-1,2,3,4,5}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
INNER_VAL_FRACTION="${INNER_VAL_FRACTION:-0.2}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-5000}"
OCCLUSION_GROUP_SIZE="${OCCLUSION_GROUP_SIZE:-10}"
DIM="${DIM:-64}"
DEPTH="${DEPTH:-5}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.5}"
RGB_PRETRAINED="${RGB_PRETRAINED:-1}"
RUN_OCCLUSION="${RUN_OCCLUSION:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"
PROFILE_WARMUP="${PROFILE_WARMUP:-20}"
PROFILE_REPEATS="${PROFILE_REPEATS:-100}"
FORCE="${FORCE:-0}"

METADATA="${OUTPUT_ROOT}/input/metadata.csv"
TASK_DIR="${OUTPUT_ROOT}/tasks"
RUN_DIR="${OUTPUT_ROOT}/runs"
EVALUATION_DIR="${OUTPUT_ROOT}/evaluation"
SPECTRAL_DIR="${OUTPUT_ROOT}/spectral"
OCCLUSION_DIR="${OUTPUT_ROOT}/occlusion"
AUDIT_DIR="${OUTPUT_ROOT}/metadata_audit"
PROFILE_CSV="${OUTPUT_ROOT}/computational_costs.csv"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/input"
LOG_FILE="${OUTPUT_ROOT}/run_all.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

on_error() {
  local exit_code=$?
  echo "FAILED at line ${BASH_LINENO[0]} (exit ${exit_code}). Re-run the same command after fixing the issue; completed folds are retained."
  exit "${exit_code}"
}
trap on_error ERR

echo "project_root=${PROJECT_ROOT}"
echo "split_dir=${SPLIT_DIR}"
echo "output_root=${OUTPUT_ROOT}"
echo "device=${DEVICE}"
echo "tasks=${TASKS}"
echo "models=${MODELS}"
echo "seeds=${SEEDS}"
echo "folds=${FOLDS}"
echo "model_shape=dim${DIM},depth${DEPTH},heads${HEADS},dropout${DROPOUT}"
echo "rgb_pretrained=${RGB_PRETRAINED}"

cd "${PROJECT_ROOT}"
command -v "${PYTHON_BIN}" >/dev/null
test -f "${SPLIT_DIR}/trainval.pt"
test -f "${SPLIT_DIR}/cleantest.pt"

"${PYTHON_BIN}" - <<'PY'
import importlib
required = ("numpy", "pandas", "PIL", "scipy", "sklearn", "torch", "torchvision")
for name in required:
    importlib.import_module(name)
print("Python dependencies: OK")
PY

if [[ "${DEVICE}" == cuda* ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
fi

INGEST_ARGS=(
  --split-dir "${SPLIT_DIR}"
  --output "${METADATA}"
)
if [[ -n "${RGB_ROOT}" ]]; then
  INGEST_ARGS+=(--rgb-root "${RGB_ROOT}")
fi
"${PYTHON_BIN}" -m reviewer_experiments.cli ingest-split4re "${INGEST_ARGS[@]}"

"${PYTHON_BIN}" - "${METADATA%.csv}.summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if not summary["multimodal_ready"]:
    raise SystemExit(
        f"RGB paths are missing for {summary['missing_rgb']} samples. "
        "Set RGB_ROOT to the server directory containing the matching RGB files."
    )
if summary["plants"] != 96:
    raise SystemExit(f"Unexpected biological plant count: {summary['plants']} (expected 96)")
if not summary["design_complete"]:
    print(
        "WARNING: the PT bundles do not contain the complete 576-pair design: "
        f"observed={summary['samples']}, missing_expected={summary['missing_expected_samples']}, "
        f"unexpected={summary['unexpected_samples']}. "
        "The analysis will use the observed samples and retain an audit trail."
    )
print(f"Dataset identity: {summary['samples']} observed samples from {summary['plants']} biological plants")
PY

"${PYTHON_BIN}" -m reviewer_experiments.cli audit \
  --metadata "${METADATA}" \
  --output "${AUDIT_DIR}"

"${PYTHON_BIN}" -m reviewer_experiments.cli prepare \
  --metadata "${METADATA}" \
  --output "${TASK_DIR}" \
  --folds 5 \
  --seed 157 \
  --require-files

read -r BANDS HEIGHT WIDTH < <("${PYTHON_BIN}" - "${METADATA}" <<'PY'
import pandas as pd
import sys
from reviewer_experiments.data import load_hsi
row = pd.read_csv(sys.argv[1]).iloc[0]
cube = load_hsi(row.hsi_path, row.get("hsi_layout", "auto"))
print(cube.shape[0], cube.shape[1], cube.shape[2])
PY
)
echo "hsi_shape=${BANDS}x${HEIGHT}x${WIDTH}"

"${PYTHON_BIN}" -m reviewer_experiments.cli params \
  --bands "${BANDS}" \
  --dim "${DIM}" \
  --depth "${DEPTH}" \
  --heads "${HEADS}" > "${OUTPUT_ROOT}/parameter_counts.csv"

MATRIX_ARGS=(
  --task-dir "${TASK_DIR}"
  --tasks "${TASKS}"
  --models "${MODELS}"
  --seeds "${SEEDS}"
  --folds "${FOLDS}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --inner-val-fraction "${INNER_VAL_FRACTION}"
  --dim "${DIM}"
  --depth "${DEPTH}"
  --heads "${HEADS}"
  --dropout "${DROPOUT}"
  --device "${DEVICE}"
  --output "${RUN_DIR}"
)
if [[ "${RGB_PRETRAINED}" == "1" ]]; then
  MATRIX_ARGS+=(--rgb-pretrained)
fi
if [[ "${FORCE}" == "1" ]]; then
  MATRIX_ARGS+=(--force)
fi
"${PYTHON_BIN}" -m reviewer_experiments.cli matrix "${MATRIX_ARGS[@]}"

PREDICTION_FILES=()
while IFS= read -r prediction_file; do
  if [[ -n "${prediction_file}" ]]; then
    PREDICTION_FILES+=("${prediction_file}")
  fi
done < "${RUN_DIR}/prediction_manifest.txt"
if [[ "${#PREDICTION_FILES[@]}" -eq 0 ]]; then
  echo "No prediction files were generated" >&2
  exit 1
fi
"${PYTHON_BIN}" -m reviewer_experiments.cli evaluate \
  --inputs "${PREDICTION_FILES[@]}" \
  --bootstrap "${BOOTSTRAP_REPEATS}" \
  --seed 157 \
  --output "${EVALUATION_DIR}"

SPECTRAL_ARGS=(
  --metadata "${METADATA}"
  --output "${SPECTRAL_DIR}"
)
if [[ -f "${WAVELENGTHS_CSV}" ]]; then
  SPECTRAL_ARGS+=(--wavelengths "${WAVELENGTHS_CSV}")
else
  echo "WARNING: ${WAVELENGTHS_CSV} is missing; physical red-edge metrics will be marked incomplete."
fi
"${PYTHON_BIN}" -m reviewer_experiments.cli spectral "${SPECTRAL_ARGS[@]}"

if [[ "${RUN_OCCLUSION}" == "1" && ",${MODELS}," == *",psnet_full,"* ]]; then
  OCCLUSION_FILES=()
  IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
  IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
  IFS=',' read -r -a FOLD_ARRAY <<< "${FOLDS}"
  for task in "${TASK_ARRAY[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
      for fold in "${FOLD_ARRAY[@]}"; do
        checkpoint="${RUN_DIR}/psnet_full/${task}/seed_${seed}/fold_${fold}/best.pt"
        destination="${OCCLUSION_DIR}/psnet_full/${task}/seed_${seed}/fold_${fold}"
        if [[ -f "${destination}/band_occlusion.csv" && "${FORCE}" != "1" ]]; then
          echo "resume: keeping ${destination}/band_occlusion.csv"
          OCCLUSION_FILES+=("${destination}/band_occlusion.csv")
          continue
        fi
        "${PYTHON_BIN}" -m reviewer_experiments.cli occlusion \
          --checkpoint "${checkpoint}" \
          --task-csv "${TASK_DIR}/${task}.csv" \
          --group-size "${OCCLUSION_GROUP_SIZE}" \
          --batch-size "${BATCH_SIZE}" \
          --workers "${WORKERS}" \
          --device "${DEVICE}" \
          --output "${destination}"
        OCCLUSION_FILES+=("${destination}/band_occlusion.csv")
      done
    done
  done

  if [[ "${#OCCLUSION_FILES[@]}" -eq 0 ]]; then
    echo "No occlusion files were generated" >&2
    exit 1
  fi
  OCCLUSION_SUMMARY_ARGS=(--inputs "${OCCLUSION_FILES[@]}" --output "${OCCLUSION_DIR}/summary")
  if [[ -f "${WAVELENGTHS_CSV}" ]]; then
    OCCLUSION_SUMMARY_ARGS+=(--wavelengths "${WAVELENGTHS_CSV}")
  fi
  "${PYTHON_BIN}" -m reviewer_experiments.cli aggregate-occlusion "${OCCLUSION_SUMMARY_ARGS[@]}"
else
  echo "Skipping band occlusion (RUN_OCCLUSION=${RUN_OCCLUSION}, psnet_full selected: ${MODELS})."
fi

if [[ "${RUN_PROFILE}" == "1" ]]; then
  "${PYTHON_BIN}" -m reviewer_experiments.cli profile \
    --models "${MODELS}" \
    --bands "${BANDS}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --batch-size 1 \
    --warmup "${PROFILE_WARMUP}" \
    --repeats "${PROFILE_REPEATS}" \
    --dim "${DIM}" \
    --depth "${DEPTH}" \
    --heads "${HEADS}" \
    --device "${DEVICE}" \
    --output "${PROFILE_CSV}"
else
  echo "Skipping model profiling (RUN_PROFILE=${RUN_PROFILE})."
fi

"${PYTHON_BIN}" -m pip freeze > "${OUTPUT_ROOT}/environment_frozen.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUTPUT_ROOT}/nvidia_smi.txt"
fi
git rev-parse HEAD > "${OUTPUT_ROOT}/git_commit.txt"

echo "COMPLETE"
echo "Main metrics: ${EVALUATION_DIR}/metrics_summary.csv"
echo "Spectral statistics: ${SPECTRAL_DIR}/within_dpi_band_statistics.csv"
echo "Occlusion summary: ${OCCLUSION_DIR}/summary/band_occlusion_summary.csv"
echo "Computational costs: ${PROFILE_CSV}"
