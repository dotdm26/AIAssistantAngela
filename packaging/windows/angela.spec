"""PyInstaller spec for the GUI and console worker executables."""

from pathlib import Path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ


project_root = Path(SPECPATH).resolve().parents[1]
collect_data = []
collect_binaries = []
collect_hidden_imports = [
    "src.desktop.app",
    "src.desktop.setup_wizard",
    "src.runtime.backup",
    "src.runtime.cli",
    "src.runtime.config",
    "src.runtime.discord_adapter",
    "src.runtime.migrations",
    "src.runtime.observability",
    "src.runtime.runtime",
    "src.runtime.secrets",
    "src.runtime.worker",
    "src.storage.factory",
    "src.storage.sqlite_store",
    "keyring.backends.Windows",
    "pystray._win32",
]


analysis = Analysis(
    [str(project_root / "src" / "launcher.py")],
    pathex=[str(project_root)],
    binaries=collect_binaries,
    datas=collect_data,
    hiddenimports=collect_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

python_archive = PYZ(analysis.pure)
gui_executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Angela",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
worker_executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Angela-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

COLLECT(
    gui_executable,
    worker_executable,
    analysis.binaries,
    analysis.datas,
    analysis.zipfiles,
    name="Angela",
)
