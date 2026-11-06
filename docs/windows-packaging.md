# Windows packaging and clean-machine testing

The Windows release is a per-user installer. It installs application files
under `%LOCALAPPDATA%\Programs\Angela` and leaves user data under
`%LOCALAPPDATA%\Angela` so upgrades and uninstallations do not erase user
conversations or credentials.

## Build

Run these commands from a native Windows PowerShell checkout. The build is
not expected to run from WSL because PyInstaller, Inno Setup, and the Windows
credential backend are native Windows components.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\build_windows.ps1
```

The script creates:

```text
dist\Angela\Angela.exe
dist\Angela\Angela-worker.exe
dist\installer\Angela-Setup-0.1.0.exe
```

The GUI executable supervises `Angela-worker.exe`. The worker is a separate
console executable so its stdout can be captured by the GUI without opening a
console window for the GUI itself.

## Backup and restore

Stop Angela before restoring an archive. Backups contain non-secret
configuration and a consistent SQLite snapshot; API keys, Discord tokens, and
OAuth credentials remain in the operating-system credential store and are not
exported.

```powershell
Angela-worker.exe --backup "$env:USERPROFILE\Desktop\angela-backup.zip"
Angela-worker.exe --restore "$env:USERPROFILE\Desktop\angela-backup.zip" --force
```

The restore command creates `pre-restore-backup.zip` in the target data
directory before replacing existing config or database files.

## Clean-machine test matrix

Use fresh Windows 10 and Windows 11 virtual machines or snapshots.

1. Install the EXE with the default per-user path.
2. Launch Angela and confirm the first-run setup appears.
3. Enter credentials and verify they are stored by Windows Credential Manager.
4. Select SQLite, start the worker, and confirm Discord reaches `ready`.
5. Send a Discord message and confirm a response is produced.
6. Create a backup, remove the application, reinstall it, and restore the backup.
7. Install a newer build over the old build and confirm config, credentials, and
   conversations remain available.
8. Uninstall and confirm application files and shortcuts are removed while
   `%LOCALAPPDATA%\Angela` remains.
9. Reinstall and confirm the preserved data is detected.
10. Test the optional user-data deletion flow separately, only after making a
    backup.

For repeatable smoke checks after installation:

```powershell
& "$env:LOCALAPPDATA\Programs\Angela\Angela.exe" --version
```

On a clean VM, the install/upgrade/backup/uninstall smoke test can be run
with:

```powershell
.\scripts\test_windows_installer.ps1 `
  -InstallerPath .\dist\installer\Angela-Setup-0.1.0.exe `
  -UpgradeInstallerPath .\dist\installer\Angela-Setup-0.1.1.exe
```

The script verifies that application files install, the worker can create a
backup, upgrades preserve a user-data sentinel, and uninstallation removes
application files without deleting `%LOCALAPPDATA%\Angela`.

## Optional compact ONNX profile

The default profile keeps the existing Sentence Transformers backend for
compatibility. To create a smaller CPU-only embedding artifact, install the
export dependencies and run:

```powershell
python -m pip install -r requirements-onnx-build.txt
python scripts\export_onnx_model.py `
  --output models\all-MiniLM-L6-v2
```

Then configure the local runtime with:

```dotenv
EMBEDDING_BACKEND="onnx"
EMBEDDING_MODEL_PATH="models/all-MiniLM-L6-v2"
```

The exporter defaults to `sentence-transformers/all-MiniLM-L6-v2`, a compact
architecture supported by Optimum's ONNX exporter. Nomic Embed v1.5 uses the
custom `nomic_bert` architecture and cannot be exported by this generic path;
continue using it through the default Sentence Transformers backend unless a
compatible pre-converted ONNX artifact is supplied.

The exporter writes an INT8 `model.int8.onnx` file and a SHA-256 manifest.
Benchmark its retrieval quality and startup/memory usage against the default
backend before making it the installer default. Existing conversation vectors
must be rebuilt when changing embedding models.
