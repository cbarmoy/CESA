"""Schema objects for global display/processing profiles.

Extends the original flat ``channel_filter_params`` with the richer
``channel_filter_pipelines`` dict that serialises full
``FilterPipeline`` objects from ``CESA.filter_engine``.  Old profiles
are transparently upgraded on load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


PROFILE_SCHEMA_VERSION = "1.0"


@dataclass
class ScalingConfig:
    """Scaling / gain configuration for the EEG viewer."""

    enabled: bool = False
    mode: str = "manual"  # "manual" | "auto"
    global_gain: float = 1.0
    spacing_uv: float = 150.0
    per_channel_gains: Dict[str, float] = field(default_factory=dict)
    # ECG is ~1–5 mV vs EEG ~50–100 µV → default attenuation ~20× so traces
    # stay readable next to EEG without overflowing neighbouring lanes.
    per_type_gains: Dict[str, float] = field(default_factory=lambda: {
        "eeg": 1.0, "eog": 1.0, "emg": 0.5, "ecg": 0.05,
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
        defaults = {"eeg": 1.0, "eog": 1.0, "emg": 0.5, "ecg": 0.05}
        raw_type = dict(data.get("per_type_gains") or defaults)
        # Upgrade legacy profiles that still ship ecg/emg at 1.0 (unusable
        # next to EEG on a µV scale).
        if float(raw_type.get("ecg", 1.0)) >= 0.99:
            raw_type["ecg"] = defaults["ecg"]
        if float(raw_type.get("emg", 1.0)) >= 0.99:
            raw_type["emg"] = defaults["emg"]
        return cls(
            enabled=bool(data.get("enabled", False)),
            mode=str(data.get("mode", "manual")),
            global_gain=float(data.get("global_gain", 1.0)),
            spacing_uv=float(data.get("spacing_uv", 150.0)),
            per_channel_gains={
                str(k): float(v)
                for k, v in (data.get("per_channel_gains") or {}).items()
            },
            per_type_gains={str(k): float(v) for k, v in raw_type.items()},
            clipping_enabled=bool(data.get("clipping_enabled", False)),
            clip_value_uv=float(data.get("clip_value_uv", 500.0)),
        )


LAYOUT_PRESETS: Dict[str, Dict[str, float]] = {
    # Lane-height multipliers only (amplitude is handled by per_type_gains).
    # Keep ECG/EMG lanes reasonably tall — shrinking them causes overflow.
    "standard": {"eeg": 1.0, "eog": 1.0, "emg": 0.8, "ecg": 0.8},
    "clinical_eeg": {"eeg": 1.5, "eog": 1.0, "emg": 0.6, "ecg": 0.7},
    "compact": {"eeg": 1.0, "eog": 0.7, "emg": 0.5, "ecg": 0.6},
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
            per_type_spacing_multiplier=_sanitize_layout_multipliers(
                data.get("per_type_spacing_multiplier")
            ),
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


def _sanitize_layout_multipliers(raw: Any) -> Dict[str, float]:
    """Load layout multipliers; bump legacy ECG/EMG lanes that were too small."""
    defaults = {"eeg": 1.0, "eog": 1.0, "emg": 0.8, "ecg": 0.8}
    src = raw or defaults
    out: Dict[str, float] = {}
    for k, v in src.items():
        key = str(k)
        val = float(v)
        if key in ("ecg", "emg") and val < 0.5:
            val = 0.7
        out[key] = val
    for k, v in defaults.items():
        out.setdefault(k, v)
    return out


@dataclass
class SignalSection:
    """One signal section rendered in PSG view."""

    key: str
    label: str
    ratio: float
    signal_type: str = "eeg"
    enabled: bool = True
    color_palette: List[str] = field(default_factory=list)


SPECTROGRAM_FREQ_PRESETS: Dict[str, Tuple[float, float]] = {
    "wideband": (0.5, 30.0),
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (11.0, 16.0),
    "beta": (16.0, 30.0),
    "wideband_45": (0.5, 45.0),
}


@dataclass
class SpectrogramConfig:
    """Wavelet (Morlet) spectrogram bars, aligned with the hypnogram axis.

    Two independent bands share this configuration:
    - "night": full recording, below the hypnogram (computed once in the
      background, shares its height with the hypnogram).
    - "window": visible EEG window only, below the EEG trace (recomputed
      at high resolution on navigation).
    """

    enabled_night: bool = False
    enabled_window: bool = False
    night_height_ratio: float = 0.4  # 0..1 share of the hypno+night height budget
    window_height_px: int = 90
    channel: str = ""  # "" -> auto-resolve (D11 > C3-M1/M2 > first EEG)
    fmin: float = 0.5
    fmax: float = 30.0
    n_freqs: int = 40
    freq_log: bool = False
    colormap: str = "viridis"
    scale: str = "db"  # "db" | "linear" | "zscore"
    use_filtered: bool = False
    onef_correction: bool = False
    opacity: float = 1.0
    overlay_hypnogram: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled_night": bool(self.enabled_night),
            "enabled_window": bool(self.enabled_window),
            "night_height_ratio": float(self.night_height_ratio),
            "window_height_px": int(self.window_height_px),
            "channel": str(self.channel),
            "fmin": float(self.fmin),
            "fmax": float(self.fmax),
            "n_freqs": int(self.n_freqs),
            "freq_log": bool(self.freq_log),
            "colormap": str(self.colormap),
            "scale": str(self.scale),
            "use_filtered": bool(self.use_filtered),
            "onef_correction": bool(self.onef_correction),
            "opacity": float(self.opacity),
            "overlay_hypnogram": bool(self.overlay_hypnogram),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectrogramConfig":
        if not data:
            return cls()
        return cls(
            enabled_night=bool(data.get("enabled_night", False)),
            enabled_window=bool(data.get("enabled_window", False)),
            night_height_ratio=float(data.get("night_height_ratio", 0.4)),
            window_height_px=int(data.get("window_height_px", 90)),
            channel=str(data.get("channel", "") or ""),
            fmin=float(data.get("fmin", 0.5)),
            fmax=float(data.get("fmax", 30.0)),
            n_freqs=int(data.get("n_freqs", 40)),
            freq_log=bool(data.get("freq_log", False)),
            colormap=str(data.get("colormap", "viridis") or "viridis"),
            scale=str(data.get("scale", "db") or "db"),
            use_filtered=bool(data.get("use_filtered", False)),
            onef_correction=bool(data.get("onef_correction", False)),
            opacity=float(data.get("opacity", 1.0)),
            overlay_hypnogram=bool(data.get("overlay_hypnogram", False)),
        )


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
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)

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
            "spectrogram": self.spectrogram.to_dict(),
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
            spectrogram=SpectrogramConfig.from_dict(data.get("spectrogram", {})),
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

