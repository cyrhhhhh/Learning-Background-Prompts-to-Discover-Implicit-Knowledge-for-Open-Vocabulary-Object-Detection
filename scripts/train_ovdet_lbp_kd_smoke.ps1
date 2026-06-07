param(
  [string]$PythonExe = "python"
)

$projectRoot = if ($env:LBP_PROJECT_ROOT) { $env:LBP_PROJECT_ROOT } else { Split-Path -Parent $PSScriptRoot }
$env:LBP_PROJECT_ROOT = $projectRoot
$env:PYTORCH_CUDA_ALLOC_CONF = if ($env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF } else { "max_split_size_mb:128" }

Set-Location (Join-Path $projectRoot 'ovdet')

& $PythonExe tools/train.py `
  configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_smoke.py `
  --work-dir (Join-Path $projectRoot 'outputs\ovdet_train_kd_smoke_quick')
