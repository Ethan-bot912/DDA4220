# Overnight chain for the within-event experiment (protocol v3, step 8).
# B1 (post) first; B2 (early) starts only after all 20 B1 epochs complete.
# Exit codes: 0 = both complete, 2 = session paused, 1 = failure.
# Launch from repo root (see CODEX_OVERNIGHT.md section 5, S7). ASCII only on purpose.
$ErrorActionPreference = 'Stop'
$root = 'D:\DDA4220\DDA4220'
Set-Location $root
$cfg = 'configs/within_event_matthew_moore_seed42.json'
$base = 'output/within_event_mm'
New-Item -ItemType Directory -Force $base | Out-Null
$chain = Join-Path $base 'chain.log'
$chainStatus = Join-Path $root "$base/chain_status.json"
$models = [ordered]@{}
$activeModel = $null
function Note($text) { Add-Content -Path $chain -Value ((Get-Date -Format o) + ' ' + $text) -Encoding ascii }

function Set-ChainStatus($status, $message) {
    Note "$status $message"
    $snapshot = [ordered]@{
        status = $status
        message = $message
        chain_pid = $PID
        active_model = $activeModel
        updated_local = (Get-Date -Format o)
        models = $models
    }
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText("$chainStatus.partial", ($snapshot | ConvertTo-Json -Depth 6), $utf8)
    Move-Item -LiteralPath "$chainStatus.partial" -Destination $chainStatus -Force
}

function Run-Model($mode, $dir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $argv = @('-u', '-m', 'tools.run_expanded_session', '--config', $cfg, '--mode', $mode,
              '--output-dir', "$base/$dir", '--session-hours', '8')
    Note "start $dir mode=$mode"
    $p = Start-Process -FilePath $py -ArgumentList $argv -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$base/${dir}_runner_$stamp.stdout.log" -RedirectStandardError "$base/${dir}_runner_$stamp.stderr.log"
    $null = $p.Handle   # keep the handle so ExitCode is available after exit
    Note "$dir runner pid=$($p.Id)"
    $p.WaitForExit()
    if ($null -eq $p.ExitCode) { throw "$dir runner exit code is unavailable" }
    $status = 'missing'
    $completed = $null
    if (Test-Path "$base/$dir/status.json") {
        $state = Get-Content "$base/$dir/status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        $status = $state.status
        $completed = $state.completed_epochs
    }
    Note "end $dir exit=$($p.ExitCode) status=$status completed_epochs=$completed"
    return [pscustomobject]@{
        exit_code = $p.ExitCode
        status = $status
        completed_epochs = $completed
        runner_pid = $p.Id
    }
}

try {
    $py = (Resolve-Path '.venv\Scripts\python.exe').Path
    $epochs = (Get-Content $cfg -Raw -Encoding UTF8 | ConvertFrom-Json).epochs
    if ($epochs -ne 20) { throw 'Protocol v3 requires exactly 20 epochs' }
    Set-ChainStatus 'running' 'chain started'
    foreach ($model in @(@{ mode = 'post'; dir = 'b1' }, @{ mode = 'early'; dir = 'b2' })) {
        $activeModel = $model.dir
        Set-ChainStatus 'running' "starting $activeModel"
        $result = Run-Model $model.mode $activeModel
        $models[$activeModel] = $result
        if ($result.exit_code -ne 0) { throw "$activeModel runner exited $($result.exit_code)" }
        if ($result.status -eq 'paused' -and $null -ne $result.completed_epochs -and
            $result.completed_epochs -ge 1 -and $result.completed_epochs -lt $epochs) {
            $message = "$activeModel paused at $($result.completed_epochs)/$epochs; resume explicitly"
            if ($activeModel -eq 'b1') { $message += '; B2 not started' }
            Set-ChainStatus 'paused' $message
            exit 2
        }
        if ($result.status -ne 'complete' -or $result.completed_epochs -ne $epochs) {
            throw "$activeModel did not report a valid completion: status=$($result.status), epochs=$($result.completed_epochs)"
        }
    }
    $activeModel = $null
    Set-ChainStatus 'complete' 'both models completed 20/20 epochs'
    exit 0
} catch {
    Set-ChainStatus 'failed' $_.Exception.Message
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
