param(
    [string]$Manifest = 'dataset\pilot_tier3_4events\manifest_classification.csv',
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int[]]$Seeds = @(42),
    [string]$OutputRoot = 'output\resnet18',
    [int]$MaxTrain = 0,
    [int]$MaxVal = 0
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$env:TEMP = 'D:\DDA4220\tmp'
$env:TMP = $env:TEMP
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
foreach ($seed in $Seeds) {
    foreach ($mode in @('post', 'early')) {
        $outputFile = Join-Path $OutputRoot "${mode}_seed${seed}.json"
        $checkpointFile = [System.IO.Path]::ChangeExtension($outputFile, '.pt')
        if ((Test-Path -LiteralPath $outputFile) -or (Test-Path -LiteralPath $checkpointFile)) {
            throw "Output exists: $outputFile. Choose a new OutputRoot to preserve prior runs."
        }
        & $pythonExe -m stage_b_damage.train_baseline --manifest $Manifest --project-root $projectRoot --output $outputFile --backbone resnet18 --mode $mode --epochs $Epochs --batch-size $BatchSize --lr 0.001 --crop-size 128 --context 16 --workers 0 --seed $seed --max-train $MaxTrain --max-val $MaxVal --device cuda
        if ($LASTEXITCODE -ne 0) { throw "Training failed: mode=$mode seed=$seed" }
    }
}
