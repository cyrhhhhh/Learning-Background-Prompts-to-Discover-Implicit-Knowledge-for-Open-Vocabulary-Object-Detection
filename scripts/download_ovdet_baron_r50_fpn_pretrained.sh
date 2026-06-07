#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
PROJECT_ROOT="${LBP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/external/checkpoints/baron/r50_fpn_clip}"

export LBP_PROJECT_ROOT="${PROJECT_ROOT}"
mkdir -p "${OUT_DIR}"

"${PYTHON_EXE}" -m gdown "1rtPRsT5JQfraNPTRx-lpQkgvSN02qJl6" \
  -O "${OUT_DIR}/20230408_125206.log" \
  --continue --no-check-certificate

"${PYTHON_EXE}" -m gdown "1Kxdf8gXWeoMVzkIUgwPPDagZD-bwwsJO" \
  -O "${OUT_DIR}/iter_90000.pth" \
  --continue --no-check-certificate
