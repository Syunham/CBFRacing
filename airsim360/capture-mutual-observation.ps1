$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
$ownedPid = $null
try {
    & .\start-simulator.ps1
    $ownedPid = [int](Get-Content -LiteralPath 'logs\simulator.pid')
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $connection = [System.Net.Sockets.TcpClient]::new()
        try { $connection.Connect('127.0.0.1',41451); $ready = $true }
        catch { Start-Sleep -Seconds 1 }
        finally { $connection.Dispose() }
        if ($ready) { break }
    }
    if (-not $ready) { throw 'Simulator RPC did not become ready' }
    & .\.venv\python.exe -m scripts.simulation.capture_mutual_observation
    if ($LASTEXITCODE -ne 0) { throw 'Mutual observation capture failed' }
} finally {
    if ($ownedPid) {
        $taskProcess = Get-Process -Id $ownedPid -ErrorAction SilentlyContinue
        if ($taskProcess -and $taskProcess.ProcessName -eq 'AirSim360-Win64-Shipping') {
            Stop-Process -Id $ownedPid -ErrorAction Continue
        }
    }
    Pop-Location
}
