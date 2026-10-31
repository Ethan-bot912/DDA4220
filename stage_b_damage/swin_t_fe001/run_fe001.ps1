param(
    [ValidateSet("validate", "estimate", "pilot", "train", "eval-val")]
    [string]$Action = "validate",
    [string]$Config = "",
    [string]$OutputDir = "",
    [string]$Checkpoint = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $repoRoot "configs\swin_t_fe001_b1.json"
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $projectPython = Join-Path $repoRoot "4220project\python.exe"
    if (Test-Path -LiteralPath $projectPython) {
        $PythonExe = $projectPython
    } else {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $PythonExe $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

switch ($Action) {
    "validate" {
        Invoke-ProjectPython ".\validate_fe001.py" "--config" $Config
    }
    "estimate" {
        Invoke-ProjectPython ".\estimate_runtime.py"
    }
    "pilot" {
        if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = "runs\pilot_fe001_gpu_bs16" }
        Invoke-ProjectPython ".\train_b1.py" "--config" $Config "--output-dir" $OutputDir `
            "--epochs" "1" "--max-train" "128" "--max-val" "64" `
            "--device" "cuda" "--batch-size" "16"
    }
    "train" {
        if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = "runs\fe001_b1_swin_t_seed42" }
        Invoke-ProjectPython ".\train_b1.py" "--config" $Config "--output-dir" $OutputDir
    }
    "eval-val" {
        if ([string]::IsNullOrWhiteSpace($Checkpoint)) { throw "-Checkpoint is required for eval-val" }
        if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = "runs\eval_val" }
        Invoke-ProjectPython ".\evaluate_b1.py" "--config" $Config "--checkpoint" $Checkpoint `
            "--split" "val" "--output-dir" $OutputDir
    }
}

