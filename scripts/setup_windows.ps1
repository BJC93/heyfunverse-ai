# HeyFunverse AI: Windows setup (PowerShell).
#   .\scripts\setup_windows.ps1            # runtime packages
#   .\scripts\setup_windows.ps1 -Dev       # also installs test tools
#   .\scripts\setup_windows.ps1 -Lock      # install exact versions from requirements.lock.txt
# Downloads Python packages only. It never downloads AI models.
param(
    [string]$VenvName = '.venv',
    [switch]$Dev,
    [switch]$Lock
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "== Checking Python"
$candidates = @(@('py', '-3.13'), @('py', '-3'), @('python'))
$exe = $null
$extra = @()
foreach ($c in $candidates) {
    $tryExe = $c[0]
    $tryExtra = @()
    if ($c.Length -gt 1) { $tryExtra = $c[1..($c.Length - 1)] }
    try {
        $version = & $tryExe @tryExtra --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $version) {
            $exe = $tryExe; $extra = $tryExtra
            Write-Host "Using $tryExe $tryExtra ($version)"
            break
        }
    } catch { }
}
if (-not $exe) { throw 'Python 3.10+ not found. Install Python 3.13 from python.org and re-run.' }

if (-not (Test-Path $VenvName)) {
    Write-Host "== Creating virtual environment $VenvName"
    & $exe @extra -m venv $VenvName
}

$venvPython = Join-Path $VenvName 'Scripts\python.exe'
& $venvPython -m pip install --upgrade pip

if ($Lock) {
    if (-not (Test-Path 'requirements.lock.txt')) { throw 'requirements.lock.txt not found. Run scripts\freeze_env.ps1 on a working machine first.' }
    & $venvPython -m pip install -r requirements.lock.txt
} elseif ($Dev) {
    & $venvPython -m pip install -r requirements-dev.txt
} else {
    & $venvPython -m pip install -r requirements.txt
}

Write-Host "== Environment check"
& $venvPython ai-engine\check_env.py
Write-Host ""
Write-Host "Activate with:  $VenvName\Scripts\Activate.ps1"
