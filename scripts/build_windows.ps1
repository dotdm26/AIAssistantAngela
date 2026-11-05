$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
Set-Location $repositoryRoot

python -m pip install -r requirements.txt -r requirements-build.txt
python -m PyInstaller --clean --noconfirm packaging/windows/angela.spec

$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $iscc) {
    throw "Inno Setup Compiler (ISCC.exe) is required. Install it and rerun this script."
}

& $iscc.Source packaging/windows/angela.iss
Write-Host "Installer created under dist/installer"
