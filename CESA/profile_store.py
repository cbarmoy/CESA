"""Persistence layer for CESA global display/processing profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from CESA.profile_schema import DisplayProcessingProfile, build_default_profile


class ProfileStore:
    """JSON-backed profile store in user home directory."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or (Path.home() / ".cesa_profiles")
        self.profiles_dir = self.root_dir / "profiles"
        self.index_path = self.root_dir / "index.json"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({"last_profile": "default"})

    def list_profiles(self) -> List[str]:
        names: List[str] = []
        for fp in self.profiles_dir.glob("*.json"):
            names.append(fp.stem)
        return sorted(set(names))

    def load_profile(self, name: str) -> DisplayProcessingProfile:
        path = self._profile_path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return DisplayProcessingProfile.from_dict(data)

    def save_profile(self, profile: DisplayProcessingProfile) -> None:
        profile.touch()
        path = self._profile_path(profile.name)
        path.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")

    def delete_profile(self, name: str) -> None:
        if name == "default":
            return
        path = self._profile_path(name)
        if path.exists():
            path.unlink()
        if self.get_last_profile_name() == name:
            self.set_last_profile_name("default")

    def ensure_default_profile(self) -> DisplayProcessingProfile:
        try:
            return self.load_profile("default")
        except Exception:
            profile = build_default_profile()
            self.save_profile(profile)
            return profile

    def get_last_profile_name(self) -> str:
        data = self._read_index()
        return str(data.get("last_profile", "default"))

    def set_last_profile_name(self, name: str) -> None:
        data = self._read_index()
        data["last_profile"] = str(name)
        self._write_index(data)

    def load_last_or_default(self) -> DisplayProcessingProfile:
        self.ensure_default_profile()
        last_name = self.get_last_profile_name()
        try:
            return self.load_profile(last_name)
        except Exception:
            return self.load_profile("default")

    def duplicate_profile(self, source_name: str, target_name: str) -> DisplayProcessingProfile:
        profile = self.load_profile(source_name)
        profile.name = str(target_name)
        self.save_profile(profile)
        return profile

    def rename_profile(self, old_name: str, new_name: str) -> DisplayProcessingProfile:
        if old_name == "default":
            raise ValueError("Cannot rename default profile")
        profile = self.load_profile(old_name)
        profile.name = str(new_name)
        self.save_profile(profile)
        old_path = self._profile_path(old_name)
        if old_path.exists():
            old_path.unlink()
        if self.get_last_profile_name() == old_name:
            self.set_last_profile_name(new_name)
        return profile

    def _profile_path(self, name: str) -> Path:
        safe = "".join(ch for ch in str(name).strip() if ch.isalnum() or ch in ("-", "_")).strip()
        if not safe:
            safe = "profile"
        return self.profiles_dir / f"{safe}.json"

    def _read_index(self) -> Dict[str, str]:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"last_profile": "default"}

    def _write_index(self, data: Dict[str, str]) -> None:
        self.index_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

