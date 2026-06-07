param(
  [string]$PythonExe = "python",
  [string]$Checkpoint = "",
  [string]$WorkDir = ""
)

$projectRoot = if ($env:LBP_PROJECT_ROOT) { $env:LBP_PROJECT_ROOT } else { Split-Path -Parent $PSScriptRoot }
$env:LBP_PROJECT_ROOT = $projectRoot

if (-not $Checkpoint) {
  $Checkpoint = Join-Path $projectRoot 'outputs\ovdet_train_kd_mini50\iter_50.pth'
}
if (-not $WorkDir) {
  $WorkDir = Join-Path $projectRoot 'outputs\ovdet_train_kd_subset20_eval'
}
if (-not (Test-Path $Checkpoint)) {
  throw "Checkpoint not found: $Checkpoint"
}

Set-Location (Join-Path $projectRoot 'ovdet')

& $PythonExe tools/test.py `
  configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_subset20_eval.py `
  $Checkpoint `
  --work-dir $WorkDir
