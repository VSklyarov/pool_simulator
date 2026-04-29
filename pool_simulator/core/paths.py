"""Centralized path resolution for both dev and PyInstaller frozen modes.

In dev mode, paths resolve relative to the project root.
In frozen mode (PyInstaller), bundled read-only data lives under sys._MEIPASS,
while user-writable directories (profiles) live next to the executable.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_bundle_dir() -> Path:
    """Root for bundled read-only resources (processed_data, etc.)."""
    if _is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent.parent


def get_app_dir() -> Path:
    """Root for user-writable data (profiles, configs).

    In frozen mode this is the directory containing the .exe.
    In dev mode it's the project root.
    """
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    return get_bundle_dir() / "processed_data"


def get_profiles_dir() -> Path:
    return get_app_dir() / "profiles"
