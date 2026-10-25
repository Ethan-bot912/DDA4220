# Exercise the real chain's process/status decisions with a fake runner.
# No Python, CUDA, real datasets, or model training is used.
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$source = Get-Content (Join-Path $repo 'docs/within_event_mm/run_overnight_chain.ps1') -Raw
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('dda4220-chain-' + [guid]::NewGuid().ToString('N'))
$powershell = (Get-Command powershell.exe -ErrorAction Stop).Source
New-Item -ItemType Directory $testRoot | Out-Null

$bootstrap = @'
param([string]$FixtureRoot, [string]$SubjectPath)
$ErrorActionPreference = 'Stop'
$scenario = Get-Content (Join-Path $FixtureRoot 'scenario.json') -Raw | ConvertFrom-Json
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle,
          [switch]$PassThru, $RedirectStandardOutput, $RedirectStandardError)
    $dir = if ($ArgumentList -contains 'post') { 'b1' } else { 'b2' }
    Add-Content (Join-Path $FixtureRoot 'launched.txt') $dir -Encoding ascii
    if ($scenario.throw_on -eq $dir) { throw 'Simulated process creation failure' }
    $state = $scenario.$dir
    $out = Join-Path $FixtureRoot "output/within_event_mm/$dir"
    New-Item -ItemType Directory -Force $out | Out-Null
    if ($state.kind -eq 'malformed') {
        Set-Content (Join-Path $out 'status.json') '{invalid' -Encoding ascii
    } elseif ($state.kind -ne 'missing') {
        $state | Select-Object status, completed_epochs | ConvertTo-Json |
            Set-Content (Join-Path $out 'status.json') -Encoding ascii
    }
    $process = [pscustomobject]@{ Id = 9001; Handle = 1; ExitCode = $state.exit_code }
    $process | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value {}
    return $process
}
& $SubjectPath
exit $LASTEXITCODE
'@

function State($status, $epochs, $code = 0, $kind = 'valid') {
    return @{ status = $status; completed_epochs = $epochs; exit_code = $code; kind = $kind }
}
$complete = State 'complete' 20
$cases = @(
    @{ name = 'both_complete'; b1 = $complete; b2 = $complete; code = 0; status = 'complete'; launched = 'b1,b2' },
    @{ name = 'b1_paused_blocks_b2'; b1 = (State 'paused' 7); b2 = $complete; code = 2; status = 'paused'; launched = 'b1' },
    @{ name = 'b2_paused_is_not_complete'; b1 = $complete; b2 = (State 'paused' 9); code = 2; status = 'paused'; launched = 'b1,b2' },
    @{ name = 'failed_b1_blocks_b2_even_with_complete_status'; b1 = (State 'complete' 20 1); b2 = $complete; code = 1; status = 'failed'; launched = 'b1' },
    @{ name = 'failed_b2'; b1 = $complete; b2 = (State 'running' 8 1); code = 1; status = 'failed'; launched = 'b1,b2' },
    @{ name = 'incomplete_epoch_count_blocks_b2'; b1 = (State 'complete' 19); b2 = $complete; code = 1; status = 'failed'; launched = 'b1' },
    @{ name = 'missing_b1_status_blocks_b2'; b1 = (State '' 0 0 'missing'); b2 = $complete; code = 1; status = 'failed'; launched = 'b1' },
    @{ name = 'malformed_b2_status'; b1 = $complete; b2 = (State '' 0 0 'malformed'); code = 1; status = 'failed'; launched = 'b1,b2' },
    @{ name = 'unexpected_b2_state'; b1 = $complete; b2 = (State 'running' 9); code = 1; status = 'failed'; launched = 'b1,b2' },
    @{ name = 'process_creation_failure_blocks_b2'; b1 = $complete; b2 = $complete; throw_on = 'b1'; code = 1; status = 'failed'; launched = 'b1' }
)

try {
    $bootstrapPath = Join-Path $testRoot 'bootstrap.ps1'
    Set-Content -LiteralPath $bootstrapPath -Value $bootstrap -Encoding ascii
    foreach ($case in $cases) {
        $caseRoot = Join-Path $testRoot $case.name
        New-Item -ItemType Directory -Force (Join-Path $caseRoot '.venv/Scripts') | Out-Null
        New-Item -ItemType Directory (Join-Path $caseRoot 'configs') | Out-Null
        Set-Content (Join-Path $caseRoot '.venv/Scripts/python.exe') '' -Encoding ascii
        Set-Content (Join-Path $caseRoot 'configs/within_event_matthew_moore_seed42.json') '{"epochs":20}' -Encoding ascii
        $case | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $caseRoot 'scenario.json') -Encoding ascii
        $subjectPath = Join-Path $caseRoot 'subject.ps1'
        $rootAssignment = '$root = ''' + $caseRoot.Replace("'", "''") + "'"
        $subject = $source.Replace('$root = ''D:\DDA4220\DDA4220''', $rootAssignment)
        if ($subject -eq $source) { throw 'The test did not isolate the chain root' }
        $utf8Bom = New-Object System.Text.UTF8Encoding($true)
        [IO.File]::WriteAllText($subjectPath, $subject, $utf8Bom)
        $childLog = Join-Path $caseRoot 'child.log'
        # Windows PowerShell treats redirected native stderr as ErrorRecord;
        # these cases intentionally exercise failures, so inspect the exit code.
        $savedPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $powershell -NoProfile -ExecutionPolicy Bypass -File $bootstrapPath -FixtureRoot $caseRoot -SubjectPath $subjectPath *> $childLog
            $code = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $savedPreference
        }
        $actual = Get-Content (Join-Path $caseRoot 'output/within_event_mm/chain_status.json') -Raw | ConvertFrom-Json
        $launched = (Get-Content (Join-Path $caseRoot 'launched.txt')) -join ','
        if ($code -ne $case.code -or $actual.status -ne $case.status -or $launched -ne $case.launched) {
            Get-Content $childLog | Write-Output
            throw "$($case.name): exit=$code status=$($actual.status) launched=$launched"
        }
        Write-Output "PASS $($case.name)"
    }
    Write-Output "All $($cases.Count) chain checks passed (fake runner; no training)."
} finally {
    $resolved = [IO.Path]::GetFullPath($testRoot)
    $parent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($parent, [StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path $resolved -Leaf) -notlike 'dda4220-chain-*') {
        throw "Refusing cleanup outside the test directory: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
