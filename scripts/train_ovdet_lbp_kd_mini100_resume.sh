#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
PROJECT_ROOT="${LBP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESUME_FROM="${RESUME_FROM:-${PROJECT_ROOT}/outputs/ovdet_train_kd_mini50/iter_50.pth}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/outputs/ovdet_train_kd_mini100_resume}"

export LBP_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

if [ ! -f "${RESUME_FROM}" ]; then
  echo "Checkpoint not found: ${RESUME_FROM}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}/ovdet"

"${PYTHON_EXE}" tools/train.py \
  configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_mini100.py \
  --work-dir "${WORK_DIR}" \
  --resume "${RESUME_FROM}"
