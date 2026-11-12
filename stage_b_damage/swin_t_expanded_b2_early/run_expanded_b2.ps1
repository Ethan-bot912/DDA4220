param(
    [ValidateSet("validate", "pilot", "train", "eval-val")]
    [string]$Action = "validate",
    [string]$PythonExe = "",
    [string]$Config = "..\..\configs\swin_t_expanded_b2_early_seed42_bs16.json",
    [string]$OutputDir = "..\..\runs\expanded_b2_early_seed42_bs16",
    [string]$Checkpoint = "",
    [switch]$CheckImages,
    [switch]$SampleImages,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# This is the B2-only entrypoint.  It is intentionally safe by default:
# invoking it without -Action performs validation and never creates a run.
$SwinRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $SwinRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $bundledPython = Join-Path $RepoRoot "4220project\python.exe"
    if (Test-Path -LiteralPath $bundledPython) {
        $PythonExe = $bundledPython
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            throw "Python executable not found; pass -PythonExe explicitly"
        }
        $PythonExe = $pythonCommand.Source
    }
}
if (-not (Test-Path -LiteralPath $PythonExe) -and $PythonExe -ne "python") {
    throw "Python executable not found: $PythonExe"
}

$ConfigPath = if ([System.IO.Path]::IsPathRooted($Config)) {
    [System.IO.Path]::GetFullPath($Config)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $SwinRoot $Config))
}
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "B2 config not found: $ConfigPath"
}

Push-Location $SwinRoot
try {
    switch ($Action) {
        "validate" {
            $validateArgs = @(
                ".\validate_b2.py",
                "--config", $ConfigPath
            )
            if ($CheckImages) {
                $validateArgs += "--check-images"
            }
            if ($SampleImages) {
                $validateArgs += @("--sample", "2")
            }
            & $PythonExe @validateArgs
            if ($LASTEXITCODE -ne 0) {
                throw "B2 validation failed with exit code $LASTEXITCODE"
            }
        }
        "pilot" {
            # Pilot is explicit and isolated from the formal output directory.
            $pilotOutput = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "runs\pilot_expanded_b2_early_seed42_bs16"))
            $pilotArgs = @(
                ".\train_b1.py",
                "--config", $ConfigPath,
                "--output-dir", $pilotOutput,
                "--device", "cuda",
                "--batch-size", "16",
                "--epochs", "1",
                "--max-train", "128",
                "--max-val", "64"
            )
            & $PythonExe @pilotArgs
            if ($LASTEXITCODE -ne 0) {
                throw "B2 pilot failed with exit code $LASTEXITCODE"
            }
        }
        "train" {
            $resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
                [System.IO.Path]::GetFullPath($OutputDir)
            } else {
                [System.IO.Path]::GetFullPath((Join-Path $SwinRoot $OutputDir))
            }
            $trainArgs = @(
                ".\train_b1.py",
                "--config", $ConfigPath,
                "--output-dir", $resolvedOutput,
                "--device", "cuda",
                "--batch-size", "16"
            )
            if ($Resume) {
                $trainArgs += "--resume"
            }
            & $PythonExe @trainArgs
            if ($LASTEXITCODE -ne 0) {
                throw "B2 training failed with exit code $LASTEXITCODE"
            }
        }
        "eval-val" {
            if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
                $Checkpoint = [System.IO.Path]::GetFullPath((Join-Path $SwinRoot (Join-Path $OutputDir "best.pt")))
            } else {
                $Checkpoint = if ([System.IO.Path]::IsPathRooted($Checkpoint)) {
                    [System.IO.Path]::GetFullPath($Checkpoint)
                } else {
                    [System.IO.Path]::GetFullPath((Join-Path $SwinRoot $Checkpoint))
                }
            }
            $evalOutput = [System.IO.Path]::GetFullPath((Join-Path $SwinRoot (Join-Path $OutputDir "eval_val")))
            $evalArgs = @(
                ".\evaluate_b1.py",
                "--config", $ConfigPath,
                "--checkpoint", $Checkpoint,
                "--split", "val",
                "--output-dir", $evalOutput,
                "--device", "cuda"
            )
            & $PythonExe @evalArgs
            if ($LASTEXITCODE -ne 0) {
                throw "B2 val evaluation failed with exit code $LASTEXITCODE"
            }
        }
    }
}
finally {
    Pop-Location
}
