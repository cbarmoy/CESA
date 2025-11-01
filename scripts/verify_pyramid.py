#!/usr/bin/env python
"""
Verify integrity of a CESA multiscale (Zarr) pyramid.

Usage:
  python scripts/verify_pyramid.py "C:\Users\...\sample\S043_Ap_EDF+"
  python scripts/verify_pyramid.py "C:\Users\...\sample\S043_Ap_EDF+\_ms"
  python scripts/verify_pyramid.py  # uses default below
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    # Allow running from project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
except Exception:
    pass

from core.store import open_multiscale


def normalize_ms_path(user_input: Optional[str]) -> Path:
    if user_input:
        p = Path(user_input)
    else:
        # Default for convenience (change as needed)
        p = Path(r"C:\Users\comeb\Documents\SupBiotech\Stage\BT4\Code\CESA 1.0\sample\S043_Ap_EDF+")

    if p.is_file() and p.suffix.lower() == ".edf":
        return p.with_suffix("") / "_ms"

    if p.is_dir():
        # direct Zarr?
        if (p / ".zattrs").exists() and (p / "levels").exists():
            return p
        # has child _ms?
        if (p / "_ms").exists():
            return p / "_ms"
        # assume this is EDF base dir name
        return p / "_ms"

    # Non-existing path: assume it's EDF base; point to _ms
    if p.suffix.lower() == ".edf":
        return p.with_suffix("") / "_ms"
    return p / "_ms"


def exists_level1(ms_root: Path) -> bool:
    return (ms_root / "levels" / "lvl1").exists()


def main() -> int:
    user_arg = sys.argv[1] if len(sys.argv) > 1 else None
    ms_path = normalize_ms_path(user_arg)

    print(f"🔎 Checking multiscale path: {ms_path}", flush=True)
    print(f"   exists={ms_path.exists()}  .zattrs={(ms_path / '.zattrs').exists()}  levels={(ms_path / 'levels').exists()}  lvl1={(ms_path / 'levels' / 'lvl1').exists()}", flush=True)

    if not ms_path.exists():
        print("❌ Path does not exist.", flush=True)
        return 2

    if not (ms_path / ".zattrs").exists() or not (ms_path / "levels").exists():
        print("❌ Not a valid Zarr multiscale folder (missing .zattrs/levels).", flush=True)
        return 3

    # Open store and validate metadata/datasets
    try:
        store = open_multiscale(ms_path)
    except Exception as e:
        print(f"❌ Failed to open_multiscale: {e}", flush=True)
        return 4

    meta = store.metadata
    print("📄 Metadata:", flush=True)
    print(f"   fs={meta.sampling_frequency} Hz  channels={len(meta.channel_names)}  total_samples={meta.total_samples:,}", flush=True)
    print(f"   levels={list(meta.level_bin_sizes)}", flush=True)
    if 1 in meta.level_bin_sizes:
        print("✅ Level 1 declared in metadata.", flush=True)
    else:
        print("❌ Level 1 is NOT declared in metadata.", flush=True)

    # Check level descriptors and basic shapes
    ok = True
    try:
        levels = store.available_levels()
        for b in levels:
            lvl = store.get_level(b)
            arr = lvl.dataset
            if arr.ndim != 3 or arr.shape[2] != 2:
                print(f"❌ lvl{b}: invalid shape {arr.shape} (expected (channels, bins, 2))", flush=True)
                ok = False
            if arr.shape[0] != len(meta.channel_names):
                print(f"❌ lvl{b}: channel dimension {arr.shape[0]} != {len(meta.channel_names)}", flush=True)
                ok = False
            # Quick sample read (first 5 bins) to ensure data is readable
            try:
                _ = arr.oindex[0, 0 : min(5, arr.shape[1]), :]
            except Exception as er:
                print(f"❌ lvl{b}: failed to read sample window: {er}", flush=True)
                ok = False
    except Exception as e:
        print(f"❌ Failed while iterating levels: {e}", flush=True)
        return 5

    if 1 not in levels or not exists_level1(ms_path):
        print("❌ Level 1 missing (either not declared or dataset absent on disk).", flush=True)
        ok = False

    if ok:
        dur_min = (meta.total_samples / max(meta.sampling_frequency, 1.0)) / 60.0
        print(f"✅ Multiscale store looks OK. Duration≈{dur_min:.1f} min, levels={len(levels)}", flush=True)
        return 0
    else:
        print("⚠️  Multiscale store has issues. See messages above.", flush=True)
        return 6


if __name__ == "__main__":
    sys.exit(main())


