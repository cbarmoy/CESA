"""Schema objects for global display/processing profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


PROFILE_SCHEMA_VERSION = "1.0"


@dataclass
class SignalSection:
    """One signal section rendered in PSG view."""

    key: str
    label: str
    ratio: float
    signal_type: str = "eeg"
    enabled: bool = True
    color_palette: List[str] = field(default_factory=list)


@dataclass
class DisplayProcessingProfile:
    """Global profile for channel mapping, layout and processing."""

    name: str
    version: str = PROFILE_SCHEMA_VERSION
    last_used_at: str = ""
    hypnogram_ratio: float = 1.0
    events_ratio: float = 0.6
    signal_sections: List[SignalSection] = field(default_factory=list)
    channel_mappings: Dict[str, str] = field(default_factory=dict)  # channel -> section key
    ignored_channels: List[str] = field(default_factory=list)
    baseline_enabled: bool = True
    baseline_window_duration: float = 30.0
    filter_enabled: bool = True
    filter_order: int = 4
    filter_low: Optional[float] = None
    filter_high: Optional[float] = None
    filter_type: str = "butterworth"
    filter_window: str = "hamming"
    channel_filter_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_used_at = datetime.utcnow().isoformat(timespec="seconds")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "last_used_at": self.last_used_at,
            "hypnogram_ratio": float(self.hypnogram_ratio),
            "events_ratio": float(self.events_ratio),
            "signal_sections": [asdict(section) for section in self.signal_sections],
            "channel_mappings": dict(self.channel_mappings),
            "ignored_channels": list(self.ignored_channels),
            "baseline_enabled": bool(self.baseline_enabled),
            "baseline_window_duration": float(self.baseline_window_duration),
            "filter_enabled": bool(self.filter_enabled),
            "filter_order": int(self.filter_order),
            "filter_low": self.filter_low,
            "filter_high": self.filter_high,
            "filter_type": str(self.filter_type),
            "filter_window": str(self.filter_window),
            "channel_filter_params": {
                str(ch): dict(cfg) for ch, cfg in self.channel_filter_params.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisplayProcessingProfile":
        sections_raw = data.get("signal_sections", []) or []
        sections = [
            SignalSection(
                key=str(item.get("key", "")).strip(),
                label=str(item.get("label", "")).strip() or str(item.get("key", "")).strip(),
                ratio=float(item.get("ratio", 1.0)),
                signal_type=str(item.get("signal_type", "eeg")).strip().lower() or "eeg",
                enabled=bool(item.get("enabled", True)),
                color_palette=[
                    str(c).strip()
                    for c in (item.get("color_palette", []) or [])
                    if str(c).strip()
                ],
            )
            for item in sections_raw
            if str(item.get("key", "")).strip()
        ]
        return cls(
            name=str(data.get("name", "default")),
            version=str(data.get("version", PROFILE_SCHEMA_VERSION)),
            last_used_at=str(data.get("last_used_at", "")),
            hypnogram_ratio=float(data.get("hypnogram_ratio", 1.0)),
            events_ratio=float(data.get("events_ratio", 0.6)),
            signal_sections=sections,
            channel_mappings={str(k): str(v) for k, v in (data.get("channel_mappings", {}) or {}).items()},
            ignored_channels=[str(x) for x in (data.get("ignored_channels", []) or [])],
            baseline_enabled=bool(data.get("baseline_enabled", True)),
            baseline_window_duration=float(data.get("baseline_window_duration", 30.0)),
            filter_enabled=bool(data.get("filter_enabled", True)),
            filter_order=int(data.get("filter_order", 4)),
            filter_low=float(data["filter_low"]) if data.get("filter_low") is not None else None,
            filter_high=float(data["filter_high"]) if data.get("filter_high") is not None else None,
            filter_type=str(data.get("filter_type", "butterworth")),
            filter_window=str(data.get("filter_window", "hamming")),
            channel_filter_params={
                str(k): dict(v) for k, v in (data.get("channel_filter_params", {}) or {}).items()
            },
        )


def build_default_profile() -> DisplayProcessingProfile:
    return DisplayProcessingProfile(
        name="default",
        signal_sections=[
            SignalSection(
                key="eeg",
                label="EEG",
                ratio=5.0,
                signal_type="eeg",
                color_palette=["#3B82F6", "#22C55E", "#8B5CF6", "#06B6D4"],
            ),
            SignalSection(
                key="eog",
                label="EOG",
                ratio=1.2,
                signal_type="eog",
                color_palette=["#06B6D4", "#0EA5E9"],
            ),
            SignalSection(
                key="emg",
                label="EMG",
                ratio=1.2,
                signal_type="emg",
                color_palette=["#F59E0B", "#D97706"],
            ),
            SignalSection(
                key="ecg",
                label="ECG",
                ratio=1.2,
                signal_type="ecg",
                color_palette=["#EF4444", "#DC2626"],
            ),
        ],
        baseline_enabled=True,
        baseline_window_duration=30.0,
        filter_enabled=True,
        filter_order=4,
    )

