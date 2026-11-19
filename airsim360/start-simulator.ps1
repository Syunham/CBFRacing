param([switch]$Offscreen = $true,
      [ValidateSet('FactoryExteriorInteriorBundle','DekogonGym')]
      [string]$Scene = 'FactoryExteriorInteriorBundle')
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$sceneRoot = Join-Path $projectRoot "simulator\AirSim360_Pro_$Scene"
$pidFile = Join-Path $projectRoot 'logs\simulator.pid'
if (Test-Path -LiteralPath $pidFile) {
    $existingPid = [int](Get-Content -LiteralPath $pidFile)
    $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existingProcess -and $existingProcess.ProcessName -eq 'AirSim360-Win64-Shipping') {
        if (-not $existingProcess.Path.StartsWith("$sceneRoot\", [StringComparison]::OrdinalIgnoreCase)) {
            throw "Another scene is running on the project RPC port."
        }
        Write-Output "AirSim360 is already running: PID $existingPid."
        return
    }
}
$executables = @(Get-ChildItem -LiteralPath $sceneRoot -Filter '*-Win64-Shipping.exe' -Recurse)
if ($executables.Count -ne 1) {
    throw "Expected one simulator shipping executable, found $($executables.Count)."
}
$simulatorExe = $executables[0]
$settingsPath = Join-Path $projectRoot 'settings.json'
$logPath = Join-Path $projectRoot 'logs\simulator.log'
$arguments = @('-windowed', '-ResX=1280', '-ResY=720', '-nosound', '-unattended',
    ('-settings="{0}"' -f $settingsPath), ('-abslog="{0}"' -f $logPath))
if ($Offscreen) { $arguments += '-RenderOffscreen' }
$simulatorProcess = Start-Process -FilePath $simulatorExe.FullName -ArgumentList $arguments -WorkingDirectory $simulatorExe.DirectoryName -WindowStyle Hidden -PassThru
$simulatorProcess.Id | Set-Content -LiteralPath (Join-Path $projectRoot 'logs\simulator.pid')
Write-Output "AirSim360 launched: PID $($simulatorProcess.Id). RPC 127.0.0.1:41451. Log: $logPath"
