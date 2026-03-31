"""Unified MNE raw-file loading helpers for recording formats."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import mne

SUPPORTED_RECORDING_EXTENSIONS: tuple[str, ...] = (
    ".edf",
    ".edf+",
    ".bdf",
    ".fif",
    ".fif.gz",
    ".vhdr",
    ".set",
)


def normalize_recording_extension(path: str | Path) -> str:
    """Return a normalized lowercase extension identifier."""
    name = Path(path).name.lower()
    if name.endswith(".fif.gz"):
        return ".fif.gz"
    if name.endswith(".edf+"):
        return ".edf+"
    return Path(name).suffix.lower()


def is_supported_recording_file(path: str | Path) -> bool:
    ext = normalize_recording_extension(path)
    return ext in SUPPORTED_RECORDING_EXTENSIONS


def recording_extensions_for_scan() -> tuple[str, ...]:
    """Return extensions usable for recursive file scans."""
    return SUPPORTED_RECORDING_EXTENSIONS


def recording_filetypes_for_dialog() -> list[tuple[str, str]]:
    """Return Tk-compatible filetype filters for recording files."""
    pattern = " ".join(f"*{ext}" for ext in SUPPORTED_RECORDING_EXTENSIONS)
    return [
        ("Fichiers enregistrement", pattern),
        ("Tous les fichiers", "*.*"),
    ]


def open_raw_file(path: str | Path, *, preload: bool = False, verbose: str | bool = "ERROR"):
    """Open a recording with a reader selected from file extension."""
    reader = _select_reader(path)
    return reader(str(path), preload=preload, verbose=verbose)


def _select_reader(path: str | Path) -> Callable:
    ext = normalize_recording_extension(path)
    if ext in {".edf", ".edf+"}:
        return mne.io.read_raw_edf
    if ext == ".bdf":
        return mne.io.read_raw_bdf
    if ext in {".fif", ".fif.gz"}:
        return mne.io.read_raw_fif
    if ext == ".vhdr":
        return mne.io.read_raw_brainvision
    if ext == ".set":
        return mne.io.read_raw_eeglab
    raise ValueError(
        f"Unsupported recording format: {Path(path).suffix}. "
        f"Supported: {', '.join(SUPPORTED_RECORDING_EXTENSIONS)}"
    )


def iter_supported_files(root: Path, files: Iterable[str]) -> list[str]:
    """Filter a list of filenames into supported recording full paths."""
    matches: list[str] = []
    supported = set(SUPPORTED_RECORDING_EXTENSIONS)
    for file in files:
        ext = normalize_recording_extension(file)
        if ext in supported:
            matches.append(str(root / file))
    return matches
