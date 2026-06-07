param(
  [string]$PythonExe = "python",
  [string]$ResumeFrom = "",
  [string]$WorkDir = ""
)

$projectRoot = if ($env:LBP_PROJECT_ROOT) { $env:LBP_PROJECT_ROOT } else { Split-Path -Parent $PSScriptRoot }
$env:LBP_PROJECT_ROOT = $projectRoot
$env:PYTORCH_CUDA_ALLOC_CONF = if ($env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF } else { "max_split_size_mb:128" }

if (-not $ResumeFrom) {
  $ResumeFrom = Join-Path $projectRoot 'outputs\ovdet_train_kd_mini50\iter_50.pth'
}
if (-not $WorkDir) {
  $WorkDir = Join-Path $projectRoot 'outputs\ovdet_train_kd_mini100_resume'
}
if (-not (Test-Path $ResumeFrom)) {
  throw "Checkpoint not found: $ResumeFrom"
}

Set-Location (Join-Path $projectRoot 'ovdet')

& $PythonExe tools/train.py `
  configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_mini100.py `
  --work-dir $WorkDir `
  --resume $ResumeFrom
