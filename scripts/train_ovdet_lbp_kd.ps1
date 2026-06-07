$projectRoot = if ($env:LBP_PROJECT_ROOT) { $env:LBP_PROJECT_ROOT } else { Split-Path -Parent $PSScriptRoot }
$env:LBP_PROJECT_ROOT = $projectRoot

Set-Location (Join-Path $projectRoot 'ovdet')

python tools/train.py `
  configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp.py `
  --work-dir (Join-Path $projectRoot 'outputs\ovdet_train_kd')
