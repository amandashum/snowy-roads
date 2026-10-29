$ErrorActionPreference = 'Stop'

# Resolve paths from this script so setup also works from another directory
$projectRoot = $PSScriptRoot
$environmentPath = Join-Path $projectRoot '.venv'
$pythonPath = Join-Path $environmentPath 'Scripts\python.exe'
$requirementsPath = Join-Path $projectRoot 'requirements.txt'

if (-not (Test-Path -LiteralPath $environmentPath)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Install Python 3.13 with the Windows Python launcher (py), then run setup.ps1 again.'
    }

    & py -3.13 -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create .venv. Check that Python 3.13 is installed.'
    }
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The existing .venv is incomplete or incompatible. Rename it, then run setup.ps1 again.'
}

# Preserve existing environments rather than silently replacing them
& $pythonPath -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) and sys.prefix != sys.base_prefix and not __import__("site").ENABLE_USER_SITE else 1)'
if ($LASTEXITCODE -ne 0) {
    throw 'Setup requires an isolated Python 3.13 environment. Rename the existing .venv, then run setup.ps1 again.'
}

& $pythonPath -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw 'Dependency installation failed. Check the pip error above, then run setup.ps1 again.'
}

& $pythonPath -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'Dependency conflicts were found. Check the pip output above.'
}

Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -c 'from inference import load_checkpoint; load_checkpoint("models/classifier.pt"); print("Checkpoint loaded successfully on CPU.")'
    if ($LASTEXITCODE -ne 0) {
        throw 'Model verification failed. Check the error above and ensure models/classifier.pt is present.'
    }
}
finally {
    Pop-Location
}

Write-Host 'Setup complete. From the project folder, start the app with:'
Write-Host '.venv\Scripts\python.exe -m streamlit run app.py'
