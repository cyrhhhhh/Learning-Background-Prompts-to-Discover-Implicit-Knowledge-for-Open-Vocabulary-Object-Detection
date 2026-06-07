param(
  [string]$PythonExe = "python",
  [string]$OutDir = ""
)

$projectRoot = if ($env:LBP_PROJECT_ROOT) { $env:LBP_PROJECT_ROOT } else { Split-Path -Parent $PSScriptRoot }
$env:LBP_PROJECT_ROOT = $projectRoot

if (-not $OutDir) {
  $OutDir = Join-Path $projectRoot 'data\external\checkpoints\baron\r50_fpn_clip'
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

& $PythonExe -m gdown "1rtPRsT5JQfraNPTRx-lpQkgvSN02qJl6" `
  -O (Join-Path $OutDir '20230408_125206.log') `
  --continue --no-check-certificate

& $PythonExe -m gdown "1Kxdf8gXWeoMVzkIUgwPPDagZD-bwwsJO" `
  -O (Join-Path $OutDir 'iter_90000.pth') `
  --continue --no-check-certificate
