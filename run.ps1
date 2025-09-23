<#
.SYNOPSIS
    Helper script to bootstrap the local dev environment and launch the AI trader bot.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$venvPath = Join-Path -Path (Get-Location) -ChildPath '.venv'
if (-not (Test-Path -Path $venvPath)) {
    Write-Host 'Creating virtual environment (.venv)...'
    python -m venv $venvPath
}

$activateScript = Join-Path -Path $venvPath -ChildPath 'Scripts\Activate.ps1'
if (-not (Test-Path -Path $activateScript)) {
    throw "Virtual environment activation script not found at $activateScript"
}

. $activateScript

Write-Host 'Installing dependencies from requirements.txt...'
pip install --upgrade pip
pip install -r requirements.txt

Write-Host 'Validating core dependencies...'
python -c "import river, ccxt; print(f'river {river.__version__} | ccxt {ccxt.__version__}')"

Write-Host 'Starting AI Trader bot...'
python -m ai_trader.main
