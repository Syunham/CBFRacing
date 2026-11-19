param([string]$RunName = 'open_yard_rgbd_50hz')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid run name' }
$dataset = Join-Path 'captures' $RunName
$run = Join-Path 'reports' $RunName
$log = Join-Path 'logs' "$RunName.log"
$status = Join-Path 'logs' "$RunName.status.json"
$manifest = Join-Path $dataset 'manifest.json'
$source = 'assets/flight/trajectories/baseline'
$placement = 'reports\split_s_open_yard\placement.json'
$simExe = (Get-ChildItem simulator\AirSim360_Pro_FactoryExteriorInteriorBundle -Recurse -Filter '*-Win64-Shipping.exe').FullName

function Write-State([string]$phase, [string]$detail) {
    @{ phase=$phase; detail=$detail; pid=$PID; updated=(Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath "$status.tmp" -Encoding utf8
    [IO.File]::Move((Convert-Path "$status.tmp"), (Join-Path $PSScriptRoot $status), $true)
}

function Stop-OwnedSimulator([int]$idToStop) {
    $p = Get-Process -Id $idToStop -ErrorAction SilentlyContinue
    if ($p -and $p.Path -eq $simExe) {
        Stop-Process -Id $idToStop -ErrorAction Stop
        $p.WaitForExit(15000) | Out-Null
    }
}

try {
    if (Test-Path -LiteralPath $run) { throw "Training directory already exists: $run" }
    for ($attempt=1; -not (Test-Path -LiteralPath $manifest); $attempt++) {
        if (Get-Process -Name 'AirSim360-Win64-Shipping' -ErrorAction SilentlyContinue) {
            throw 'An existing simulator is using the project scene or RPC port'
        }
        Write-State 'capture' "Attempt $attempt"
        & .\start-simulator.ps1
        $ownedId = [int](Get-Content logs/simulator.pid)
        try {
            $connected = $false
            for ($i=0; $i -lt 40; $i++) {
                $tcp = [Net.Sockets.TcpClient]::new()
                try { $tcp.Connect('127.0.0.1',41451); $connected=$true } catch {} finally { $tcp.Dispose() }
                if ($connected) { break }
                Start-Sleep -Seconds 1
            }
            if (-not $connected) { throw 'Simulator did not open its RPC port' }
            & .\.venv\python.exe -u -m scripts.simulation.collect_risk_flights --source $source --output $dataset --placement $placement *>> $log
            $code = $LASTEXITCODE
        } finally {
            Stop-OwnedSimulator $ownedId
        }
        if ($code -ne 0 -and $attempt -ge 4) { throw "Capture stopped after $attempt attempts; see $log" }
    }
    Write-State 'training' 'CUDA; 30 epochs'
    & .\.conda-envs\risk-training\python.exe -u -m scripts.training.train_rgbd_field --manifest $manifest --output $run --device cuda --epochs 30 *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Training failed; see $log" }
    if (!(Test-Path "$run/best.pt") -or !(Test-Path "$run/last.pt") -or @(Get-Content "$run/history.jsonl").Count -ne 30) {
        throw 'Checkpoint or epoch history missing'
    }
    Write-State 'complete' '30 epochs; best.pt and last.pt saved'
} catch {
    Write-State 'failed' $_.Exception.Message
    throw
}
