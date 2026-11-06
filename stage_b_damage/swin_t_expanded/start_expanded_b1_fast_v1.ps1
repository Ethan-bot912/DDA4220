$ErrorActionPreference = "Stop"

# Run from this source directory on the local project checkout.
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $projectRoot "4220project\python.exe"
$config = Join-Path $projectRoot "swin-t\config_expanded_b1_fast_v1.json"
$output = Join-Path $projectRoot "swin-t\runs\expanded_b1_fast_v1_seed42_bs16"

& $python (Join-Path $PSScriptRoot "train_b1.py") `
  --config $config `
  --output-dir $output `
  --device cuda
