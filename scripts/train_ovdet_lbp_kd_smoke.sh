#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export LBP_PROJECT_ROOT="${LBP_PROJECT_ROOT:-$PROJECT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

cd "${LBP_PROJECT_ROOT}/ovdet"

python tools/train.py \
  configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_smoke.py \
  --work-dir "${LBP_PROJECT_ROOT}/outputs/ovdet_train_kd_smoke_quick"
