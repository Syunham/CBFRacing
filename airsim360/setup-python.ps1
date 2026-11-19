$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:CONDA_PKGS_DIRS = Join-Path $PSScriptRoot '.conda-pkgs'
    if (-not (Test-Path -LiteralPath '.venv\python.exe')) {
        conda create --prefix .\.venv python=3.10 pip -y --override-channels -c conda-forge
        if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
    }
    & .\.venv\python.exe -m pip install -r requirements.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed' }
    & .\.venv\python.exe -m pip install -e .\upstream\AirSim360-main\software\Python_API_Env\PythonClient --no-deps --no-build-isolation
    if ($LASTEXITCODE -ne 0) { throw 'AirSim360 client installation failed' }
} finally {
    Pop-Location
}
