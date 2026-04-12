"""Persistence for EDF import profiles (channel config, types, gains, aliases, filters).

Profiles are stored as JSON files under ``config/import_profiles/``.
A profile captures the **complete** channel configuration chosen in the
import wizard — display settings *and* filter pipelines — so it can be
reapplied to the same file or any file with a matching channel layout.

Matching heuristic
------------------
A profile matches a recording when **every original channel name** listed
in the profile exists in the recording's channel list (subset match).
Profiles are ranked by how many channels they cover so the most specific
one wins.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parents[2] / "config" / "import_profiles"


def _sanitise_filename(name: str) -> str:
    """Turn a human name into a safe filename slug."""
    slug = re.sub(r"[^\w\s-]", "", name.strip().lower())
    slug = re.sub(r"[\s]+", "_", slug)
    return slug or "profile"


# ------------------------------------------------------------------
# Data shape stored in each JSON file
# ------------------------------------------------------------------
# {
#   "profile_name": "PSG standard IRBA",
#   "global_filter_enabled": true,
#   "channels": [
#       {
#           "name": "D1",
#           "alias": "F3-M2",
#           "signal_type": "eeg",
#           "gain": 150.0,
#           "selected": true,
#           "filter_pipeline": { "enabled": true, "filters": [...] }
#       },
#       ...
#   ]
# }


class ImportProfileStore:
    """Read / write import profiles from ``config/import_profiles/``."""

    def __init__(self, profiles_dir: Optional[Path] = None) -> None:
        self._dir = profiles_dir or _PROFILES_DIR

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        profile_name: str,
        channels: List[Dict[str, Any]],
        *,
        global_filter_enabled: bool = True,
    ) -> Path:
        """Persist a profile and return the written file path."""
        self._dir.mkdir(parents=True, exist_ok=True)
        slug = _sanitise_filename(profile_name)
        path = self._dir / f"{slug}.json"

        counter = 1
        while path.exists():
            existing = self._read_json(path)
            if existing and existing.get("profile_name") == profile_name:
                break
            counter += 1
            path = self._dir / f"{slug}_{counter}.json"

        data = {
            "profile_name": profile_name,
            "global_filter_enabled": global_filter_enabled,
            "channels": channels,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[PROFILE] Saved '%s' -> %s (%d channels)", profile_name, path.name, len(channels))
        return path

    # ------------------------------------------------------------------
    # List / load
    # ------------------------------------------------------------------

    def list_profiles(self) -> List[str]:
        """Return sorted list of available profile names."""
        names: List[str] = []
        if not self._dir.is_dir():
            return names
        for p in sorted(self._dir.glob("*.json")):
            data = self._read_json(p)
            if data and "profile_name" in data:
                names.append(data["profile_name"])
        return names

    def load(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Load the full profile dict for *profile_name*.

        Returns ``{"channels": [...], "global_filter_enabled": bool}``
        or ``None`` if not found.
        """
        if not self._dir.is_dir():
            return None
        for p in self._dir.glob("*.json"):
            data = self._read_json(p)
            if data and data.get("profile_name") == profile_name:
                return {
                    "channels": data.get("channels", []),
                    "global_filter_enabled": data.get("global_filter_enabled", True),
                }
        return None

    def delete(self, profile_name: str) -> bool:
        """Delete a profile by name. Returns True if found and deleted."""
        if not self._dir.is_dir():
            return False
        for p in self._dir.glob("*.json"):
            data = self._read_json(p)
            if data and data.get("profile_name") == profile_name:
                p.unlink()
                logger.info("[PROFILE] Deleted '%s'", profile_name)
                return True
        return False

    # ------------------------------------------------------------------
    # Auto-match
    # ------------------------------------------------------------------

    def find_matching(self, channel_names: List[str]) -> Optional[str]:
        """Return the best matching profile name for the given channels.

        A profile matches when all its channel names are present in
        *channel_names*.  The profile covering the most channels wins.
        Returns ``None`` if no profile matches.
        """
        if not self._dir.is_dir():
            return None

        ch_set = set(channel_names)
        best_name: Optional[str] = None
        best_score = 0

        for p in self._dir.glob("*.json"):
            data = self._read_json(p)
            if not data or "channels" not in data:
                continue
            profile_chs = {c["name"] for c in data["channels"] if "name" in c}
            if profile_chs and profile_chs.issubset(ch_set):
                score = len(profile_chs)
                if score > best_score:
                    best_score = score
                    best_name = data.get("profile_name")

        return best_name

    # ------------------------------------------------------------------
    # Helpers to convert between ChannelInfo and dicts
    # ------------------------------------------------------------------

    @staticmethod
    def channels_to_dicts(
        channels,
        filter_pipelines: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert ``ChannelInfo`` objects to serialisable dicts.

        *filter_pipelines* is ``{channel_display_name: FilterPipeline}``
        from the controller.  Each pipeline's ``to_dict()`` is stored so
        that the full filter configuration is persisted.
        """
        pipes = filter_pipelines or {}
        result: List[Dict[str, Any]] = []
        for ch in channels:
            d: Dict[str, Any] = {
                "name": ch.name,
                "alias": ch.alias,
                "signal_type": ch.signal_type,
                "gain": ch.gain,
                "selected": ch.selected,
            }
            # Try display name first (alias), then original name
            display = ch.alias if ch.alias else ch.name
            pipe = pipes.get(display) or pipes.get(ch.name)
            if pipe is not None:
                try:
                    d["filter_pipeline"] = pipe.to_dict()
                except Exception:
                    pass
            elif ch.filter_pipeline_dict is not None:
                d["filter_pipeline"] = ch.filter_pipeline_dict
            result.append(d)
        return result

    @staticmethod
    def apply_profile_to_channels(channels, profile_dicts: List[Dict[str, Any]]) -> int:
        """Apply profile settings to a list of ``ChannelInfo`` in place.

        Returns number of channels updated.
        """
        lookup = {d["name"]: d for d in profile_dicts if "name" in d}
        updated = 0
        for ch in channels:
            if ch.name in lookup:
                d = lookup[ch.name]
                ch.alias = d.get("alias", "")
                ch.signal_type = d.get("signal_type", ch.signal_type)
                ch.gain = d.get("gain", ch.gain)
                ch.selected = d.get("selected", ch.selected)
                ch.filter_pipeline_dict = d.get("filter_pipeline")
                updated += 1
        return updated

    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
