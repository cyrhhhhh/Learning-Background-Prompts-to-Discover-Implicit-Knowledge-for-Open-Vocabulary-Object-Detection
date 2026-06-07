#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export LBP_PROJECT_ROOT="${LBP_PROJECT_ROOT:-$PROJECT_ROOT}"

cd "${LBP_PROJECT_ROOT}/ovdet"

python tools/train.py \
  configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp.py \
  --work-dir "${LBP_PROJECT_ROOT}/outputs/ovdet_train_kd"

