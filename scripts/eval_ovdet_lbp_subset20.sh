#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
PROJECT_ROOT="${LBP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs/ovdet_train_kd_mini50/iter_50.pth}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/outputs/ovdet_train_kd_subset20_eval}"

export LBP_PROJECT_ROOT="${PROJECT_ROOT}"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}/ovdet"

"${PYTHON_EXE}" tools/test.py \
  configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_subset20_eval.py \
  "${CHECKPOINT}" \
  --work-dir "${WORK_DIR}"
