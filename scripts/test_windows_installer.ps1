param(
    [Parameter(Mandatory = $true)]
    [string]$InstallerPath,
    [string]$UpgradeInstallerPath
)

$ErrorActionPreference = "Stop"
$installDirectory = Join-Path $env:LOCALAPPDATA "Programs\Angela"
$dataDirectory = Join-Path $env:LOCALAPPDATA "Angela"
$sentinel = Join-Path $dataDirectory "clean-machine-test.txt"

if (Test-Path $installDirectory) {
    throw "Install directory already exists. Run this script on a clean VM or remove the previous test installation."
}

Start-Process -FilePath (Resolve-Path $InstallerPath) -ArgumentList @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    "/CLOSEAPPLICATIONS"
) -Wait -PassThru | Out-Null

$gui = Join-Path $installDirectory "Angela.exe"
$worker = Join-Path $installDirectory "Angela-worker.exe"
if (-not (Test-Path $gui) -or -not (Test-Path $worker)) {
    throw "The installer did not install both Angela executables."
}

$version = & $gui --version
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($version)) {
    throw "The installed GUI executable did not pass its version smoke test."
}

New-Item -ItemType Directory -Force -Path $dataDirectory | Out-Null
Set-Content -Path $sentinel -Value "preserve user data" -Encoding UTF8
$backupPath = Join-Path $env:TEMP "angela-clean-machine-backup.zip"
& $worker --backup $backupPath
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $backupPath)) {
    throw "Backup smoke test failed."
}

if ($UpgradeInstallerPath) {
    Start-Process -FilePath (Resolve-Path $UpgradeInstallerPath) -ArgumentList @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/CLOSEAPPLICATIONS"
    ) -Wait -PassThru | Out-Null
    if (-not (Test-Path $sentinel)) {
        throw "Upgrade removed existing user data."
    }
}

$uninstaller = Join-Path $installDirectory "unins000.exe"
Start-Process -FilePath $uninstaller -ArgumentList @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART"
) -Wait -PassThru | Out-Null

if (Test-Path $installDirectory) {
    throw "Uninstallation left application files behind."
}
if (-not (Test-Path $sentinel)) {
    throw "Uninstallation removed user data that should have been preserved."
}

Write-Host "Windows installer clean-machine checks passed for $version"
