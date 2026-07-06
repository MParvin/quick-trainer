# PyInstaller spec for the quick-trainer CLI.
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = collect_submodules("quick_trainer") + [
    "typer",
    "click",
    "pydantic",
    "pydantic_settings",
    "yaml",
]

a = Analysis(
    ["quick_trainer/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="quick-trainer",
    debug=False,
    strip=False,
    upx=True,
    console=True,
)
