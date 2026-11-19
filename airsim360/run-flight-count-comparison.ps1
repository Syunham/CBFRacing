param([int]$ExistingCollector, [int]$ExistingSimulator)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$exp='reports/flight_count_comparison'
$capture='captures/open_yard_rgbd_expansion'
$manifest=Join-Path $capture 'manifest.json'
$simExe=(Get-ChildItem simulator/AirSim360_Pro_FactoryExteriorInteriorBundle -Recurse -Filter '*-Win64-Shipping.exe').FullName
function Set-Phase([string]$phase) {
    @{phase=$phase; updated=(Get-Date).ToString('o'); pid=$PID} | ConvertTo-Json | Set-Content "$exp/status.tmp"
    [IO.File]::Move((Convert-Path "$exp/status.tmp"),(Join-Path $PSScriptRoot "$exp/status.json"),$true)
}
function Release-Simulator([int]$owned) {
    $process=Get-Process -Id $owned -ErrorAction SilentlyContinue
    if ($process -and $process.Path -eq $simExe -and [int](Get-Content logs/simulator.pid) -eq $owned) {
        Stop-Process -Id $owned
        $process.WaitForExit(15000) | Out-Null
    }
}
try {
    Set-Phase 'capture'
    if ($ExistingCollector) {
        $collector=Get-Process -Id $ExistingCollector -ErrorAction SilentlyContinue
        if ($collector -and $collector.Path -eq (Resolve-Path '.venv/python.exe').Path) { $collector.WaitForExit() }
        Release-Simulator $ExistingSimulator
    }
    $previousCount=-1
    $stalls=0
    while (!(Test-Path $manifest)) {
        $count=@(Get-ChildItem "$capture/*/frames/*.npz" -File).Count
        if ($count -eq $previousCount) {$stalls++} else {$stalls=0}
        if ($stalls -ge 4) {throw "Capture repeatedly stopped at frame count $count"}
        $previousCount=$count
        if (Get-Process -Name AirSim360-Win64-Shipping -ErrorAction SilentlyContinue) {throw 'Another simulator is active'}
        & .\start-simulator.ps1
        $owned=[int](Get-Content logs/simulator.pid)
        try {
            Start-Sleep -Seconds 6
            & .\.venv\python.exe -u -m scripts.simulation.collect_risk_flights --output $capture --placement reports/split_s_open_yard/placement.json --flight-list "$exp/capture_plan.json" *>> logs/flight_count_capture.log
        } finally {Release-Simulator $owned}
    }
    foreach ($n in 4,8) {
        if (Test-Path "$exp/run_$n") {
            if ((Test-Path "$exp/run_$n/last.pt") -and @(Get-Content "$exp/run_$n/history.jsonl").Count -eq 30) {continue}
            throw "Incomplete existing run $n; inspect before resuming"
        }
        Set-Phase "training_$n"
        & .\.conda-envs\risk-training\python.exe -u -m scripts.training.train_rgbd_field --manifest "$exp/manifest_$n.json" --output "$exp/run_$n" --device cuda --epochs 30 *>> logs/flight_count_training.log
        if ($LASTEXITCODE -ne 0) {throw "Training $n failed"}
    }
    Set-Phase 'evaluation'
    & .\.conda-envs\risk-training\python.exe -u -m scripts.evaluation.evaluate_flight_count --experiment $exp --baseline reports/open_yard_rgbd_50hz *>> logs/flight_count_evaluation.log
    if ($LASTEXITCODE -ne 0) {throw 'Held-out evaluation failed'}
    Set-Phase 'complete'
} catch {
    Set-Phase "failed: $($_.Exception.Message)"
    throw
}
