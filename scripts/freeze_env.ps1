# Records the exact package versions of the active virtual environment.
# Run it after a successful install, then commit requirements.lock.txt.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
python -m pip freeze --exclude-editable | Out-File -Encoding utf8 (Join-Path $root 'requirements.lock.txt')
Write-Host "Wrote requirements.lock.txt"
