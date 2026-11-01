param(
    [string]$PythonExe = "",
    [string]$OutputDir = "runs\expanded_b1_swin_t_seed42_bs16",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SwinRoot = $PSScriptRoot
$Workspace = (Resolve-Path (Join-Path $SwinRoot "..")).Path
$DataRoot = Join-Path $Workspace "dataset\expanded_7events"
$ManifestSource = Join-Path $Workspace "stage_b_damage\primary 4 dataset\github_full_expanded_1638a3e\dataset\expanded_7events\manifest_classification.csv"
$Config = Join-Path $SwinRoot "config_expanded_swin_t.json"
$SummaryPath = Join-Path $DataRoot "extraction_summary.json"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $Workspace "4220project\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $ManifestSource)) {
    throw "Canonical expanded manifest not found: $ManifestSource"
}

New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null
Write-Output ("Waiting for expanded extraction: {0}" -f $SummaryPath)
while (-not (Test-Path -LiteralPath $SummaryPath)) {
    Start-Sleep -Seconds 30
}

$summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
if ([int]$summary.files_written -lt 9948) {
    throw ("Extraction summary reports only {0} files; refusing to train." -f $summary.files_written)
}
$requiredEvents = @(
    "hurricane-harvey", "hurricane-michael", "hurricane-matthew",
    "joplin-tornado", "lower-puna-volcano", "moore-tornado",
    "tuscaloosa-tornado"
)
$actualEvents = @($summary.events | ForEach-Object { [string]$_ })
foreach ($event in $requiredEvents) {
    if ($actualEvents -notcontains $event) {
        throw "Extraction summary is missing event: $event"
    }
}

# Preserve the frozen manifest bytes so the configured SHA256 remains stable.
Copy-Item -LiteralPath $ManifestSource -Destination (Join-Path $DataRoot "manifest_classification.csv") -Force
$descriptorRoot = Split-Path -Parent $ManifestSource
foreach ($name in @("audit.json", "data_lock.json", "event_split.json", "extraction_summary.json", "geotiff_audit.json", "validation_images.csv")) {
    $source = Join-Path $descriptorRoot $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $DataRoot $name) -Force
    }
}

Push-Location $SwinRoot
try {
    & $PythonExe .\validate_fe001.py --config .\config_expanded_swin_t.json --check-images
    if ($LASTEXITCODE -ne 0) {
        throw "Expanded manifest/image validation failed with exit code $LASTEXITCODE"
    }

    $resolvedOutput = Resolve-Path (Join-Path $SwinRoot $OutputDir) -ErrorAction SilentlyContinue
    if ($resolvedOutput) {
        $outputPath = $resolvedOutput.Path
    } else {
        $outputPath = [System.IO.Path]::GetFullPath((Join-Path $SwinRoot $OutputDir))
    }
    $trainArgs = @(
        ".\train_b1.py",
        "--config", ".\config_expanded_swin_t.json",
        "--output-dir", $outputPath,
        "--device", "cuda",
        "--batch-size", "16"
    )
    if ($Resume) { $trainArgs += "--resume" }
    & $PythonExe @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Expanded Swin-T training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
