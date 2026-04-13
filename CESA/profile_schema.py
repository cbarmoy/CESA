"""Schema objects for global display/processing profiles.

Extends the original flat ``channel_filter_params`` with the richer
``channel_filter_pipelines`` dict that serialises full
``FilterPipeline`` objects from ``CESA.filter_engine``.  Old profiles
are transparently upgraded on load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


PROFILE_SCHEMA_VERSION = "1.0"


@dataclass
class ScalingConfig:
    """Scaling / gain configuration for the EEG viewer."""

    enabled: bool = False
    mode: str = "manual"  # "manual" | "auto"
    global_gain: float = 1.0
    spacing_uv: float = 150.0
    per_channel_gains: Dict[str, float] = field(default_factory=dict)
    per_type_gains: Dict[str, float] = field(default_factory=lambda: {
        "eeg": 1.0, "eog": 1.0, "emg": 1.0, "ecg": 1.0,
    })
    clipping_enabled: bool = False
    clip_value_uv: float = 500.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "global_gain": self.global_gain,
            "spacing_uv": self.spacing_uv,
            "per_channel_gains": dict(self.per_channel_gains),
            "per_type_gains": dict(self.per_type_gains),
            "clipping_enabled": self.clipping_enabled,
            "clip_value_uv": self.clip_value_uv,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScalingConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            mode=str(data.get("mode", "manual")),
            global_gain=float(data.get("global_gain", 1.0)),
            spacing_uv=float(data.get("spacing_uv", 150.0)),
            per_channel_gains={
                str(k): float(v)
                for k, v in (data.get("per_channel_gains") or {}).items()
            },
            per_type_gains={
                str(k): float(v)
                for k, v in (data.get("per_type_gains") or {
                    "eeg": 1.0, "eog": 1.0, "emg": 1.0, "ecg": 1.0,
                }).items()
            },
            clipping_enabled=bool(data.get("clipping_enabled", False)),
            clip_value_uv=float(data.get("clip_value_uv", 500.0)),
        )


LAYOUT_PRESETS: Dict[str, Dict[str, float]] = {
    "standard": {"eeg": 1.0, "eog": 1.0, "emg": 1.0, "ecg": 1.0},
    "clinical_eeg": {"eeg": 1.5, "eog": 1.0, "emg": 0.5, "ecg": 0.4},
    "compact": {"eeg": 1.0, "eog": 0.7, "emg": 0.3, "ecg": 0.3},
}


@dataclass
class LayoutConfig:
    """Per-type vertical spacing, signal centering, and visual guide configuration."""

    enabled: bool = False
    mode: str = "standard"  # "standard" | "clinical_eeg" | "compact" | "auto" | "custom"
    per_channel_offset: Dict[str, float] = field(default_factory=dict)
    per_type_spacing_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "eeg": 1.0, "eog": 1.0, "emg": 1.0, "ecg": 1.0,
    })
    center_signal: bool = False

    # Visual guide flags
    show_baselines: bool = False
    show_amplitude_scale: bool = False
    show_grid_fine: bool = False
    show_artifact_highlight: bool = False
    artifact_threshold_uv: float = 500.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "per_channel_offset": dict(self.per_channel_offset),
            "per_type_spacing_multiplier": dict(self.per_type_spacing_multiplier),
            "center_signal": self.center_signal,
            "show_baselines": self.show_baselines,
            "show_amplitude_scale": self.show_amplitude_scale,
            "show_grid_fine": self.show_grid_fine,
            "show_artifact_highlight": self.show_artifact_highlight,
            "artifact_threshold_uv": self.artifact_threshold_uv,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayoutConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            mode=str(data.get("mode", "standard")),
            per_channel_offset={
                str(k): float(v)
                for k, v in (data.get("per_channel_offset") or {}).items()
            },
            per_type_spacing_multiplier={
                str(k): float(v)
                for k, v in (data.get("per_type_spacing_multiplier") or {
                    "eeg": 1.0, "eog": 1.0, "emg": 1.0, "ecg": 1.0,
                }).items()
            },
            center_signal=bool(data.get("center_signal", False)),
            show_baselines=bool(data.get("show_baselines", False)),
            show_amplitude_scale=bool(data.get("show_amplitude_scale", False)),
            show_grid_fine=bool(data.get("show_grid_fine", False)),
            show_artifact_highlight=bool(data.get("show_artifact_highlight", False)),
            artifact_threshold_uv=float(data.get("artifact_threshold_uv", 500.0)),
        )

    def apply_preset(self, preset_name: str) -> None:
        """Apply a built-in preset by name, updating multipliers and mode."""
        mults = LAYOUT_PRESETS.get(preset_name)
        if mults is None:
            return
        self.mode = preset_name
        self.per_type_spacing_multiplier = dict(mults)
        if preset_name != "standard":
            self.center_signal = True


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
    channel_filter_pipelines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)

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
            "channel_filter_pipelines": {
                str(ch): dict(cfg) for ch, cfg in self.channel_filter_pipelines.items()
            },
            "scaling": self.scaling.to_dict(),
            "layout": self.layout.to_dict(),
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
            channel_filter_pipelines={
                str(k): dict(v) for k, v in (data.get("channel_filter_pipelines", {}) or {}).items()
            },
            scaling=ScalingConfig.from_dict(data.get("scaling", {})),
            layout=LayoutConfig.from_dict(data.get("layout", {})),
        )


def _migrate_legacy_params_to_pipelines(
    params: Dict[str, Dict[str, Any]],
    order: int = 4,
    filter_type: str = "butterworth",
) -> Dict[str, Dict[str, Any]]:
    """Convert flat ``channel_filter_params`` to serialised ``FilterPipeline`` dicts.

    Used when loading an old profile that lacks ``channel_filter_pipelines``.
    """
    try:
        from CESA.filter_engine import pipeline_from_legacy_params
    except Exception:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for ch, cfg in params.items():
        lo = float(cfg.get("low", 0.0))
        hi = float(cfg.get("high", 0.0))
        enabled = bool(cfg.get("enabled", True))
        pipe = pipeline_from_legacy_params(
            low=lo, high=hi, order=order,
            filter_type=filter_type, enabled=enabled,
        )
        result[ch] = pipe.to_dict()
    return result


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

