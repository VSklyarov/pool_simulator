# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Pool Simulator.

Build with:
    pyinstaller pool_simulator.spec

Output goes to dist/PoolSimulator/
"""

import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "processed_data"), "processed_data"),
        (str(ROOT / "assets"), "assets"),
    ],
    hiddenimports=[
        "PySide6.QtSvg",
        "PySide6.QtSvgWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        # Prevent bloated bundles when building from a "kitchen sink" Python env:
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "pandas",
        "numba",
        "jupyter",
        "IPython",
        "notebook",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PoolSimulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(ROOT / "assets" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PoolSimulator",
)
