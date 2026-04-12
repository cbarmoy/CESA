"""Composable signal-filter engine for CESA.

Provides typed filter nodes (bandpass, highpass, lowpass, notch, smoothing),
a ``FilterPipeline`` that chains them, a ``PresetLibrary`` for
save / load of named presets to JSON, physiological-range warnings,
and an audit-logging facility for traceability.

Every filter is a dataclass with:
* ``apply(data, sfreq) -> ndarray``   -- run the filter
* ``validate(sfreq) -> list[str]``    -- return human-readable errors (empty = OK)
* ``physiological_warnings(channel_type, sfreq) -> list[str]`` -- clinical range warnings
* ``to_dict() / from_dict()``         -- JSON-safe (de)serialisation
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

logger = logging.getLogger(__name__)
_audit_logger = logging.getLogger("cesa.filter_audit")

# ---------------------------------------------------------------------------
# Physiological plausible ranges per channel type
# Values based on AASM / clinical PSG standards.
# Structure: {channel_type: {param_name: (min, max, unit, note)}}
# ---------------------------------------------------------------------------

PHYSIOLOGICAL_RANGES: Dict[str, Dict[str, Tuple[float, float, str, str]]] = {
    "eeg": {
        "bp_low_hz":  (0.1, 1.0, "Hz", "AASM recommends 0.3 Hz high-pass for EEG"),
        "bp_high_hz": (15.0, 70.0, "Hz", "Standard PSG EEG: 35 Hz; research: up to 70 Hz"),
        "hp_cutoff_hz": (0.1, 1.0, "Hz", "Too high removes slow waves critical for N3"),
        "lp_cutoff_hz": (15.0, 70.0, "Hz", "Below 15 Hz removes spindles (11-16 Hz)"),
        "notch_hz":   (45.0, 65.0, "Hz", "Mains: 50 Hz (EU) or 60 Hz (US)"),
        "order":      (2, 6, "", "High orders risk ringing and phase distortion"),
    },
    "eog": {
        "bp_low_hz":  (0.1, 1.0, "Hz", "AASM recommends 0.3 Hz for EOG"),
        "bp_high_hz": (5.0, 35.0, "Hz", "Slow eye movements are < 5 Hz; REM bursts < 30 Hz"),
        "hp_cutoff_hz": (0.1, 1.0, "Hz", "Too high removes slow eye movements"),
        "lp_cutoff_hz": (5.0, 35.0, "Hz", "Standard EOG: 10 Hz; wide-band: 35 Hz"),
        "notch_hz":   (45.0, 65.0, "Hz", "Mains rejection"),
        "order":      (1, 4, "", "Low order preferred for EOG"),
    },
    "emg": {
        "bp_low_hz":  (5.0, 20.0, "Hz", "AASM chin EMG: 10 Hz high-pass"),
        "bp_high_hz": (70.0, 200.0, "Hz", "AASM chin EMG: 100 Hz low-pass"),
        "hp_cutoff_hz": (5.0, 20.0, "Hz", "Below 10 Hz captures movement artefacts"),
        "lp_cutoff_hz": (70.0, 200.0, "Hz", "Above 100 Hz for surface EMG"),
        "notch_hz":   (45.0, 65.0, "Hz", "Mains rejection"),
        "order":      (2, 6, "", "Moderate order for EMG"),
    },
    "ecg": {
        "bp_low_hz":  (0.1, 1.0, "Hz", "Standard ECG: 0.3 Hz"),
        "bp_high_hz": (40.0, 150.0, "Hz", "Standard ECG: 70 Hz; diagnostic: 150 Hz"),
        "hp_cutoff_hz": (0.1, 1.0, "Hz", "Keep baseline wander removal gentle"),
        "lp_cutoff_hz": (40.0, 150.0, "Hz", "Below 40 Hz distorts QRS complex"),
        "notch_hz":   (45.0, 65.0, "Hz", "Mains rejection"),
        "order":      (2, 6, "", "Moderate order for ECG"),
    },
}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@dataclass
class FilterAuditEntry:
    """Single auditable event for filter parameter changes."""
    timestamp: str
    channel: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)


class FilterAuditLog:
    """Append-only log of filter configuration changes for traceability.

    Each entry records a timestamp, channel, action description, and details
    dict.  The log can be exported to JSON for inclusion in analysis reports.
    """

    def __init__(self) -> None:
        self._entries: List[FilterAuditEntry] = []

    def record(self, channel: str, action: str, **details: Any) -> None:
        entry = FilterAuditEntry(
            timestamp=_dt.datetime.now().isoformat(timespec="milliseconds"),
            channel=channel,
            action=action,
            details=details,
        )
        self._entries.append(entry)
        _audit_logger.info(
            "[%s] %s | %s | %s",
            entry.timestamp, channel, action,
            json.dumps(details, default=str),
        )

    @property
    def entries(self) -> List[FilterAuditEntry]:
        return list(self._entries)

    def to_list(self) -> List[Dict[str, Any]]:
        return [
            {"timestamp": e.timestamp, "channel": e.channel,
             "action": e.action, "details": e.details}
            for e in self._entries
        ]

    def export_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_list(), fh, indent=2, ensure_ascii=False)

    def clear(self) -> None:
        self._entries.clear()

# Registry used by ``filter_from_dict`` to look up concrete classes by name.
_FILTER_REGISTRY: Dict[str, Type["BaseFilter"]] = {}


def _register(cls: Type["BaseFilter"]) -> Type["BaseFilter"]:
    _FILTER_REGISTRY[cls.__name__] = cls
    return cls


# ---------------------------------------------------------------------------
# Helper – IIR coefficient builder
# ---------------------------------------------------------------------------

def _iir_coeffs(
    order: int,
    wn: Any,
    btype: str,
    filter_type: str = "butterworth",
    output: str = "sos",
):
    """Build IIR filter coefficients (SOS by default)."""
    from scipy.signal import butter, cheby1, cheby2, ellip

    ft = (filter_type or "butterworth").strip().lower()
    if ft == "cheby1":
        return cheby1(order, rp=0.5, Wn=wn, btype=btype, output=output)
    if ft == "cheby2":
        return cheby2(order, rs=40, Wn=wn, btype=btype, output=output)
    if ft == "ellip":
        return ellip(order, rp=0.5, rs=40, Wn=wn, btype=btype, output=output)
    return butter(order, wn, btype=btype, output=output)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

@dataclass
class BaseFilter(ABC):
    """Abstract base for all filter nodes."""

    enabled: bool = True

    # -- public API ----------------------------------------------------------

    @abstractmethod
    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Apply this filter to *data* sampled at *sfreq* Hz."""

    @abstractmethod
    def validate(self, sfreq: float = 0.0) -> List[str]:
        """Return a list of human-readable validation errors (empty = OK)."""

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict.  Subclasses should call super()."""
        d: Dict[str, Any] = {"type": type(self).__name__, "enabled": self.enabled}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseFilter":
        """Dispatch to the correct concrete class via the registry."""
        return filter_from_dict(d)

    # -- physiological plausibility ------------------------------------------

    def physiological_warnings(
        self, channel_type: str = "generic", sfreq: float = 0.0,
    ) -> List[str]:
        """Return warnings when params fall outside clinical norms.

        Override in subclasses to provide filter-specific checks.
        """
        return []

    # -- helpers used by subclasses -----------------------------------------

    @staticmethod
    def _safe(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _check_range(
        value: float, param_key: str, channel_type: str,
    ) -> Optional[str]:
        """Check *value* against PHYSIOLOGICAL_RANGES and return a warning or None."""
        ranges = PHYSIOLOGICAL_RANGES.get(channel_type.lower(), {})
        entry = ranges.get(param_key)
        if entry is None:
            return None
        lo, hi, unit, note = entry
        if value < lo:
            return f"{param_key}={value} {unit} < {lo} {unit} ({note})"
        if value > hi:
            return f"{param_key}={value} {unit} > {hi} {unit} ({note})"
        return None


# ---------------------------------------------------------------------------
# Concrete filters
# ---------------------------------------------------------------------------

@_register
@dataclass
class BandpassFilter(BaseFilter):
    """IIR band-pass filter."""

    low_hz: float = 0.3
    high_hz: float = 35.0
    order: int = 4
    filter_type: str = "butterworth"
    causal: bool = False

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        if not self.enabled:
            return data
        from scipy.signal import sosfilt, sosfiltfilt

        errs = self.validate(sfreq)
        if errs:
            return data
        safe = self._safe(data)
        if safe.size == 0:
            return safe
        nyq = sfreq / 2.0
        wn = [self.low_hz / nyq, self.high_hz / nyq]
        sos = _iir_coeffs(self.order, wn, "band", self.filter_type)
        fn = sosfilt if self.causal else sosfiltfilt
        try:
            return fn(sos, safe).astype(np.float64)
        except ValueError:
            return safe

    def validate(self, sfreq: float = 0.0) -> List[str]:
        errs: List[str] = []
        if self.low_hz <= 0:
            errs.append(f"low_hz must be > 0 (got {self.low_hz})")
        if self.high_hz <= 0:
            errs.append(f"high_hz must be > 0 (got {self.high_hz})")
        if self.low_hz >= self.high_hz:
            errs.append(f"low_hz ({self.low_hz}) must be < high_hz ({self.high_hz})")
        if self.order < 1 or self.order > 12:
            errs.append(f"order must be 1-12 (got {self.order})")
        if sfreq > 0:
            nyq = sfreq / 2.0
            if self.high_hz >= nyq:
                errs.append(f"high_hz ({self.high_hz}) must be < Nyquist ({nyq})")
            if self.low_hz >= nyq:
                errs.append(f"low_hz ({self.low_hz}) must be < Nyquist ({nyq})")
        return errs

    def physiological_warnings(self, channel_type: str = "generic", sfreq: float = 0.0) -> List[str]:
        w: List[str] = []
        msg = self._check_range(self.low_hz, "bp_low_hz", channel_type)
        if msg:
            w.append(msg)
        msg = self._check_range(self.high_hz, "bp_high_hz", channel_type)
        if msg:
            w.append(msg)
        msg = self._check_range(self.order, "order", channel_type)
        if msg:
            w.append(msg)
        return w

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(low_hz=self.low_hz, high_hz=self.high_hz, order=self.order,
                 filter_type=self.filter_type, causal=self.causal)
        return d


@_register
@dataclass
class HighpassFilter(BaseFilter):
    """IIR high-pass filter."""

    cutoff_hz: float = 0.3
    order: int = 4
    filter_type: str = "butterworth"
    causal: bool = False

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        if not self.enabled:
            return data
        from scipy.signal import sosfilt, sosfiltfilt

        if self.validate(sfreq):
            return data
        safe = self._safe(data)
        if safe.size == 0:
            return safe
        nyq = sfreq / 2.0
        wn = self.cutoff_hz / nyq
        sos = _iir_coeffs(self.order, wn, "highpass", self.filter_type)
        fn = sosfilt if self.causal else sosfiltfilt
        try:
            return fn(sos, safe).astype(np.float64)
        except ValueError:
            return safe

    def validate(self, sfreq: float = 0.0) -> List[str]:
        errs: List[str] = []
        if self.cutoff_hz <= 0:
            errs.append(f"cutoff_hz must be > 0 (got {self.cutoff_hz})")
        if self.order < 1 or self.order > 12:
            errs.append(f"order must be 1-12 (got {self.order})")
        if sfreq > 0 and self.cutoff_hz >= sfreq / 2.0:
            errs.append(f"cutoff_hz ({self.cutoff_hz}) must be < Nyquist ({sfreq / 2.0})")
        return errs

    def physiological_warnings(self, channel_type: str = "generic", sfreq: float = 0.0) -> List[str]:
        w: List[str] = []
        msg = self._check_range(self.cutoff_hz, "hp_cutoff_hz", channel_type)
        if msg:
            w.append(msg)
        msg = self._check_range(self.order, "order", channel_type)
        if msg:
            w.append(msg)
        return w

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(cutoff_hz=self.cutoff_hz, order=self.order,
                 filter_type=self.filter_type, causal=self.causal)
        return d


@_register
@dataclass
class LowpassFilter(BaseFilter):
    """IIR low-pass filter."""

    cutoff_hz: float = 35.0
    order: int = 4
    filter_type: str = "butterworth"
    causal: bool = False

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        if not self.enabled:
            return data
        from scipy.signal import sosfilt, sosfiltfilt

        if self.validate(sfreq):
            return data
        safe = self._safe(data)
        if safe.size == 0:
            return safe
        nyq = sfreq / 2.0
        wn = self.cutoff_hz / nyq
        sos = _iir_coeffs(self.order, wn, "lowpass", self.filter_type)
        fn = sosfilt if self.causal else sosfiltfilt
        try:
            return fn(sos, safe).astype(np.float64)
        except ValueError:
            return safe

    def validate(self, sfreq: float = 0.0) -> List[str]:
        errs: List[str] = []
        if self.cutoff_hz <= 0:
            errs.append(f"cutoff_hz must be > 0 (got {self.cutoff_hz})")
        if self.order < 1 or self.order > 12:
            errs.append(f"order must be 1-12 (got {self.order})")
        if sfreq > 0 and self.cutoff_hz >= sfreq / 2.0:
            errs.append(f"cutoff_hz ({self.cutoff_hz}) must be < Nyquist ({sfreq / 2.0})")
        return errs

    def physiological_warnings(self, channel_type: str = "generic", sfreq: float = 0.0) -> List[str]:
        w: List[str] = []
        msg = self._check_range(self.cutoff_hz, "lp_cutoff_hz", channel_type)
        if msg:
            w.append(msg)
        msg = self._check_range(self.order, "order", channel_type)
        if msg:
            w.append(msg)
        return w

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(cutoff_hz=self.cutoff_hz, order=self.order,
                 filter_type=self.filter_type, causal=self.causal)
        return d


@_register
@dataclass
class NotchFilter(BaseFilter):
    """IIR notch (band-stop) filter with optional harmonics."""

    freq_hz: float = 50.0
    quality_factor: float = 30.0
    harmonics: int = 1

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        if not self.enabled:
            return data
        from scipy.signal import iirnotch, filtfilt

        if self.validate(sfreq):
            return data
        y = self._safe(data)
        if y.size == 0:
            return y
        nyq = sfreq / 2.0
        for h in range(1, self.harmonics + 1):
            f0 = self.freq_hz * h
            if f0 >= nyq:
                break
            b, a = iirnotch(f0, self.quality_factor, sfreq)
            try:
                y = filtfilt(b, a, y)
            except ValueError:
                break
        return y.astype(np.float64)

    def validate(self, sfreq: float = 0.0) -> List[str]:
        errs: List[str] = []
        if self.freq_hz <= 0:
            errs.append(f"freq_hz must be > 0 (got {self.freq_hz})")
        if self.quality_factor <= 0:
            errs.append(f"quality_factor must be > 0 (got {self.quality_factor})")
        if self.harmonics < 1:
            errs.append(f"harmonics must be >= 1 (got {self.harmonics})")
        if sfreq > 0 and self.freq_hz >= sfreq / 2.0:
            errs.append(f"freq_hz ({self.freq_hz}) must be < Nyquist ({sfreq / 2.0})")
        return errs

    def physiological_warnings(self, channel_type: str = "generic", sfreq: float = 0.0) -> List[str]:
        w: List[str] = []
        msg = self._check_range(self.freq_hz, "notch_hz", channel_type)
        if msg:
            w.append(msg)
        return w

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(freq_hz=self.freq_hz, quality_factor=self.quality_factor,
                 harmonics=self.harmonics)
        return d


@_register
@dataclass
class SmoothingFilter(BaseFilter):
    """Non-IIR smoothing (moving average, Savitzky-Golay, or Gaussian)."""

    method: str = "savgol"
    window_size: int = 11
    poly_order: int = 3

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        if not self.enabled:
            return data
        if self.validate(sfreq):
            return data
        y = self._safe(data)
        n = len(y)
        ws = self.window_size if self.window_size % 2 == 1 else self.window_size + 1
        if ws > n:
            ws = n if n % 2 == 1 else max(n - 1, 1)
        if ws < 3:
            return y

        m = self.method.strip().lower()
        if m == "savgol":
            from scipy.signal import savgol_filter
            po = min(self.poly_order, ws - 1)
            return savgol_filter(y, ws, po).astype(np.float64)
        if m == "gaussian":
            from scipy.ndimage import gaussian_filter1d
            sigma = ws / 6.0
            return gaussian_filter1d(y, sigma).astype(np.float64)
        # moving_average (default)
        kernel = np.ones(ws) / ws
        return np.convolve(y, kernel, mode="same").astype(np.float64)

    def validate(self, sfreq: float = 0.0) -> List[str]:
        errs: List[str] = []
        if self.window_size < 3:
            errs.append(f"window_size must be >= 3 (got {self.window_size})")
        m = self.method.strip().lower()
        if m not in ("moving_average", "savgol", "gaussian"):
            errs.append(f"Unknown smoothing method: {self.method!r}")
        if m == "savgol" and self.poly_order < 0:
            errs.append(f"poly_order must be >= 0 (got {self.poly_order})")
        return errs

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(method=self.method, window_size=self.window_size,
                 poly_order=self.poly_order)
        return d


# ---------------------------------------------------------------------------
# De-serialisation dispatcher
# ---------------------------------------------------------------------------

def filter_from_dict(d: Dict[str, Any]) -> BaseFilter:
    """Reconstruct a filter from its ``to_dict()`` output."""
    type_name = d.get("type", "")
    cls = _FILTER_REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown filter type: {type_name!r}")
    enabled = d.get("enabled", True)
    if cls is BandpassFilter:
        return BandpassFilter(
            enabled=enabled, low_hz=float(d.get("low_hz", 0.3)),
            high_hz=float(d.get("high_hz", 35.0)),
            order=int(d.get("order", 4)),
            filter_type=str(d.get("filter_type", "butterworth")),
            causal=bool(d.get("causal", False)),
        )
    if cls is HighpassFilter:
        return HighpassFilter(
            enabled=enabled, cutoff_hz=float(d.get("cutoff_hz", 0.3)),
            order=int(d.get("order", 4)),
            filter_type=str(d.get("filter_type", "butterworth")),
            causal=bool(d.get("causal", False)),
        )
    if cls is LowpassFilter:
        return LowpassFilter(
            enabled=enabled, cutoff_hz=float(d.get("cutoff_hz", 35.0)),
            order=int(d.get("order", 4)),
            filter_type=str(d.get("filter_type", "butterworth")),
            causal=bool(d.get("causal", False)),
        )
    if cls is NotchFilter:
        return NotchFilter(
            enabled=enabled, freq_hz=float(d.get("freq_hz", 50.0)),
            quality_factor=float(d.get("quality_factor", 30.0)),
            harmonics=int(d.get("harmonics", 1)),
        )
    if cls is SmoothingFilter:
        return SmoothingFilter(
            enabled=enabled, method=str(d.get("method", "savgol")),
            window_size=int(d.get("window_size", 11)),
            poly_order=int(d.get("poly_order", 3)),
        )
    raise ValueError(f"No deserializer for filter type: {type_name!r}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class FilterPipeline:
    """Ordered chain of filters applied sequentially."""

    filters: List[BaseFilter] = field(default_factory=list)
    enabled: bool = True

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Apply every enabled filter in order."""
        if not self.enabled or not self.filters:
            return data
        y = np.asarray(data, dtype=np.float64)
        for f in self.filters:
            if f.enabled:
                y = f.apply(y, sfreq)
        return y

    def validate(self, sfreq: float = 0.0) -> List[str]:
        all_errs: List[str] = []
        for i, f in enumerate(self.filters):
            for msg in f.validate(sfreq):
                all_errs.append(f"[{i}] {type(f).__name__}: {msg}")
        return all_errs

    def physiological_warnings(
        self, channel_type: str = "generic", sfreq: float = 0.0,
    ) -> List[str]:
        """Collect physiological-range warnings from every filter in the chain."""
        warns: List[str] = []
        for i, f in enumerate(self.filters):
            if not f.enabled:
                continue
            for msg in f.physiological_warnings(channel_type, sfreq):
                warns.append(f"[{i}] {type(f).__name__}: {msg}")
        return warns

    def add(self, f: BaseFilter) -> "FilterPipeline":
        self.filters.append(f)
        return self

    def remove(self, index: int) -> "FilterPipeline":
        if 0 <= index < len(self.filters):
            self.filters.pop(index)
        return self

    def move(self, from_idx: int, to_idx: int) -> "FilterPipeline":
        if 0 <= from_idx < len(self.filters) and 0 <= to_idx < len(self.filters):
            f = self.filters.pop(from_idx)
            self.filters.insert(to_idx, f)
        return self

    def deep_copy(self) -> "FilterPipeline":
        return FilterPipeline(
            filters=[copy.deepcopy(f) for f in self.filters],
            enabled=self.enabled,
        )

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "filters": [f.to_dict() for f in self.filters],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilterPipeline":
        filters = [filter_from_dict(fd) for fd in d.get("filters", [])]
        return cls(filters=filters, enabled=d.get("enabled", True))

    # -- frequency response --------------------------------------------------

    def frequency_response(
        self, sfreq: float, n_points: int = 512
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the combined magnitude frequency response (in dB).

        Returns ``(freqs, magnitude_db)`` arrays of length *n_points*.
        """
        from scipy.signal import sosfreqz, iirnotch, freqz

        freqs = np.linspace(0, sfreq / 2.0, n_points)
        combined_h = np.ones(n_points, dtype=complex)
        worN = freqs / (sfreq / 2.0) * np.pi

        for f in self.filters:
            if not f.enabled:
                continue
            try:
                if isinstance(f, (BandpassFilter, HighpassFilter, LowpassFilter)):
                    nyq = sfreq / 2.0
                    if isinstance(f, BandpassFilter):
                        wn = [f.low_hz / nyq, f.high_hz / nyq]
                        btype = "band"
                    elif isinstance(f, HighpassFilter):
                        wn = f.cutoff_hz / nyq
                        btype = "highpass"
                    else:
                        wn = f.cutoff_hz / nyq
                        btype = "lowpass"
                    sos = _iir_coeffs(f.order, wn, btype, f.filter_type)
                    _, h = sosfreqz(sos, worN=worN)
                    if not f.causal:
                        h = h * np.conj(h)
                    combined_h *= h
                elif isinstance(f, NotchFilter):
                    for harm in range(1, f.harmonics + 1):
                        f0 = f.freq_hz * harm
                        if f0 >= sfreq / 2.0:
                            break
                        b, a = iirnotch(f0, f.quality_factor, sfreq)
                        _, h = freqz(b, a, worN=worN)
                        combined_h *= h
            except Exception:
                pass

        mag_db = 20 * np.log10(np.maximum(np.abs(combined_h), 1e-12))
        return freqs, mag_db


# ---------------------------------------------------------------------------
# Preset / library
# ---------------------------------------------------------------------------

@dataclass
class FilterPreset:
    """A named filter pipeline configuration."""

    name: str
    description: str = ""
    channel_type: str = "generic"
    pipeline: FilterPipeline = field(default_factory=FilterPipeline)
    builtin: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "channel_type": self.channel_type,
            "pipeline": self.pipeline.to_dict(),
            "builtin": self.builtin,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilterPreset":
        return cls(
            name=str(d.get("name", "Unnamed")),
            description=str(d.get("description", "")),
            channel_type=str(d.get("channel_type", "generic")),
            pipeline=FilterPipeline.from_dict(d.get("pipeline", {})),
            builtin=bool(d.get("builtin", False)),
        )


class PresetLibrary:
    """Manage a collection of ``FilterPreset`` objects.

    Supports two backing files:

    * **builtin_path** -- read-only factory presets shipped with CESA.
    * **user_path** -- read/write file for user-created presets (defaults
      to ``<config_dir>/../user_presets.json`` next to the built-in file).

    Import / export helpers allow sharing presets between users or machines.
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        user_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._presets: Dict[str, FilterPreset] = {}
        self._builtin_path: Optional[Path] = Path(path) if path else None
        if user_path is not None:
            self._user_path: Optional[Path] = Path(user_path)
        elif self._builtin_path is not None:
            self._user_path = self._builtin_path.parent / "user_presets.json"
        else:
            self._user_path = None

        if self._builtin_path and self._builtin_path.exists():
            self.load(self._builtin_path)
        if self._user_path and self._user_path.exists():
            self._load_user(self._user_path)

    # -- accessors -----------------------------------------------------------

    @property
    def builtin_path(self) -> Optional[Path]:
        return self._builtin_path

    @property
    def user_path(self) -> Optional[Path]:
        return self._user_path

    def list_names(self, channel_type: Optional[str] = None) -> List[str]:
        if channel_type:
            ct = channel_type.strip().lower()
            return [
                n for n, p in self._presets.items()
                if p.channel_type.strip().lower() in (ct, "generic")
            ]
        return list(self._presets.keys())

    def list_user_names(self) -> List[str]:
        """Return names of user (non-builtin) presets only."""
        return [n for n, p in self._presets.items() if not p.builtin]

    def list_builtin_names(self) -> List[str]:
        """Return names of built-in presets only."""
        return [n for n, p in self._presets.items() if p.builtin]

    def get(self, name: str) -> Optional[FilterPreset]:
        return self._presets.get(name)

    def add(self, preset: FilterPreset, *, overwrite: bool = False) -> None:
        if preset.name in self._presets and not overwrite:
            raise ValueError(f"Preset {preset.name!r} already exists")
        self._presets[preset.name] = preset

    def remove(self, name: str) -> None:
        p = self._presets.get(name)
        if p and p.builtin:
            raise ValueError(f"Cannot remove built-in preset {name!r}")
        self._presets.pop(name, None)

    def rename(self, old_name: str, new_name: str) -> None:
        p = self._presets.get(old_name)
        if p is None:
            raise ValueError(f"Preset {old_name!r} not found")
        if p.builtin:
            raise ValueError(f"Cannot rename built-in preset {old_name!r}")
        if new_name in self._presets:
            raise ValueError(f"Preset {new_name!r} already exists")
        self._presets.pop(old_name)
        p.name = new_name
        self._presets[new_name] = p

    # -- persistence ---------------------------------------------------------

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data.get("presets", []):
            preset = FilterPreset.from_dict(entry)
            self._presets[preset.name] = preset

    def _load_user(self, path: Path) -> None:
        """Load user presets, marking them as non-builtin."""
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data.get("presets", []):
            preset = FilterPreset.from_dict(entry)
            preset.builtin = False
            self._presets[preset.name] = preset

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save **all** presets (builtin + user) to *path* or the builtin_path."""
        target = Path(path) if path else self._builtin_path
        if target is None:
            raise ValueError("No path specified for saving presets")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {"presets": [p.to_dict() for p in self._presets.values()]}
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def save_user(self, path: Optional[Union[str, Path]] = None) -> None:
        """Persist only non-builtin presets to the user-presets file."""
        target = Path(path) if path else self._user_path
        if target is None:
            raise ValueError("No user_path specified for saving user presets")
        target.parent.mkdir(parents=True, exist_ok=True)
        user_presets = [p.to_dict() for p in self._presets.values() if not p.builtin]
        data = {"presets": user_presets}
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # -- import / export -----------------------------------------------------

    def export_presets(
        self, path: Union[str, Path], names: Optional[List[str]] = None,
    ) -> int:
        """Export selected (or all) presets to a standalone JSON file.

        Returns the number of presets exported.
        """
        path = Path(path)
        if names:
            items = [self._presets[n].to_dict() for n in names if n in self._presets]
        else:
            items = [p.to_dict() for p in self._presets.values()]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"presets": items}, fh, indent=2, ensure_ascii=False)
        return len(items)

    def import_presets(
        self, path: Union[str, Path], *, overwrite: bool = False,
    ) -> int:
        """Import presets from a JSON file.  Imported presets are non-builtin.

        Returns the number of presets imported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preset file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        count = 0
        for entry in data.get("presets", []):
            preset = FilterPreset.from_dict(entry)
            preset.builtin = False
            if preset.name in self._presets and not overwrite:
                continue
            self._presets[preset.name] = preset
            count += 1
        return count


# ---------------------------------------------------------------------------
# Undo / Redo manager
# ---------------------------------------------------------------------------

class UndoManager:
    """Stack-based undo/redo for pipeline-per-channel snapshots.

    Each snapshot is a full ``{channel: pipeline.to_dict()}`` dict.  The manager
    keeps two bounded stacks (undo/redo) and exposes ``save_state``,
    ``undo``, and ``redo``.
    """

    def __init__(self, max_depth: int = 50) -> None:
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []
        self._max_depth = max_depth

    @staticmethod
    def _snapshot(pipelines: Dict[str, "FilterPipeline"]) -> Dict[str, Any]:
        return {ch: p.to_dict() for ch, p in pipelines.items()}

    @staticmethod
    def _restore(snap: Dict[str, Any]) -> Dict[str, "FilterPipeline"]:
        return {ch: FilterPipeline.from_dict(d) for ch, d in snap.items()}

    def save_state(self, pipelines: Dict[str, "FilterPipeline"]) -> None:
        """Push the current state onto the undo stack and clear redo."""
        self._undo_stack.append(self._snapshot(pipelines))
        if len(self._undo_stack) > self._max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(
        self, current_pipelines: Dict[str, "FilterPipeline"],
    ) -> Optional[Dict[str, "FilterPipeline"]]:
        """Return the previous state (or *None* if nothing to undo)."""
        if not self._undo_stack:
            return None
        self._redo_stack.append(self._snapshot(current_pipelines))
        return self._restore(self._undo_stack.pop())

    def redo(
        self, current_pipelines: Dict[str, "FilterPipeline"],
    ) -> Optional[Dict[str, "FilterPipeline"]]:
        """Return the next state (or *None* if nothing to redo)."""
        if not self._redo_stack:
            return None
        self._undo_stack.append(self._snapshot(current_pipelines))
        return self._restore(self._redo_stack.pop())

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_depth(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_depth(self) -> int:
        return len(self._redo_stack)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()


# ---------------------------------------------------------------------------
# Adaptive filter suggester
# ---------------------------------------------------------------------------

# Mapping: (channel_type, context) -> preset name
_ADAPTIVE_RULES: Dict[Tuple[str, str], str] = {
    ("eeg", "nrem"):   "EEG Standard PSG",
    ("eeg", "rem"):    "EEG Recherche",
    ("eeg", "scoring"): "EEG Sleep Scoring AASM",
    ("eeg", "default"): "EEG Standard PSG",
    ("eog", "default"): "EOG Standard",
    ("eog", "rem"):    "EOG Large Bande",
    ("emg", "default"): "EMG Menton",
    ("emg", "scoring"): "EMG Sleep Scoring AASM",
    ("ecg", "default"): "ECG Standard",
}


@dataclass
class FilterSuggestion:
    """A single suggestion from the adaptive engine."""
    preset_name: str
    reason: str
    confidence: float = 1.0
    channel_type: str = "generic"
    context: str = "default"


class AdaptiveFilterSuggester:
    """Suggests optimal filter presets based on signal characteristics.

    Uses a combination of:
    * Channel type (EEG/EOG/EMG/ECG)
    * Sleep stage context (NREM / REM / Wake / scoring)
    * Dominant spectral content (alpha, delta, high-frequency)

    All suggestions are recorded in the audit log when accepted.
    """

    def __init__(
        self,
        preset_library: PresetLibrary,
        audit_log: Optional[FilterAuditLog] = None,
    ) -> None:
        self._lib = preset_library
        self._audit = audit_log or FilterAuditLog()

    def suggest_for_channel(
        self,
        channel_type: str,
        context: str = "default",
        signal_snippet: Optional[np.ndarray] = None,
        sfreq: float = 256.0,
    ) -> List[FilterSuggestion]:
        """Return ranked suggestions for a given channel and context."""
        ct = channel_type.strip().lower()
        ctx = context.strip().lower()
        suggestions: List[FilterSuggestion] = []

        key = (ct, ctx)
        if key in _ADAPTIVE_RULES:
            pname = _ADAPTIVE_RULES[key]
            if self._lib.get(pname):
                suggestions.append(FilterSuggestion(
                    preset_name=pname,
                    reason=f"Standard preset for {ct.upper()} in {ctx} context",
                    confidence=0.9,
                    channel_type=ct,
                    context=ctx,
                ))

        default_key = (ct, "default")
        if default_key in _ADAPTIVE_RULES and default_key != key:
            pname = _ADAPTIVE_RULES[default_key]
            if self._lib.get(pname) and pname not in [s.preset_name for s in suggestions]:
                suggestions.append(FilterSuggestion(
                    preset_name=pname,
                    reason=f"Default preset for {ct.upper()}",
                    confidence=0.7,
                    channel_type=ct,
                    context="default",
                ))

        if signal_snippet is not None and len(signal_snippet) > 0 and ct == "eeg":
            spectral = self._analyze_spectrum(signal_snippet, sfreq)
            if spectral:
                suggestions.extend(spectral)

        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions

    def _analyze_spectrum(
        self, signal: np.ndarray, sfreq: float,
    ) -> List[FilterSuggestion]:
        """Detect dominant frequency content and suggest accordingly."""
        extras: List[FilterSuggestion] = []
        try:
            n = len(signal)
            freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
            psd = np.abs(np.fft.rfft(signal)) ** 2

            delta_mask = (freqs >= 0.5) & (freqs <= 4)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            total = psd.sum() + 1e-12

            delta_ratio = psd[delta_mask].sum() / total
            alpha_ratio = psd[alpha_mask].sum() / total
            beta_ratio = psd[beta_mask].sum() / total

            if delta_ratio > 0.5:
                pname = "EEG Standard PSG"
                if self._lib.get(pname):
                    extras.append(FilterSuggestion(
                        preset_name=pname,
                        reason=f"Delta-dominant signal ({delta_ratio:.0%}) -- standard PSG bandpass preserves slow waves",
                        confidence=0.85,
                        channel_type="eeg",
                        context="nrem",
                    ))
            elif alpha_ratio > 0.3:
                pname = "EEG Recherche"
                if self._lib.get(pname):
                    extras.append(FilterSuggestion(
                        preset_name=pname,
                        reason=f"Alpha-dominant signal ({alpha_ratio:.0%}) -- wide-band preserves alpha peak",
                        confidence=0.8,
                        channel_type="eeg",
                        context="wake",
                    ))
            elif beta_ratio > 0.3:
                pname = "EEG Recherche"
                if self._lib.get(pname):
                    extras.append(FilterSuggestion(
                        preset_name=pname,
                        reason=f"Beta-dominant signal ({beta_ratio:.0%}) -- wide-band for beta activity",
                        confidence=0.75,
                        channel_type="eeg",
                        context="wake",
                    ))

            mains_50 = (freqs >= 49) & (freqs <= 51)
            mains_60 = (freqs >= 59) & (freqs <= 61)
            if psd[mains_50].sum() / total > 0.05:
                pname = "Notch 50 Hz seul"
                if self._lib.get(pname):
                    extras.append(FilterSuggestion(
                        preset_name=pname,
                        reason="50 Hz mains contamination detected",
                        confidence=0.95,
                        channel_type="generic",
                        context="artifact",
                    ))
            if psd[mains_60].sum() / total > 0.05:
                pname = "Notch 60 Hz seul"
                if self._lib.get(pname):
                    extras.append(FilterSuggestion(
                        preset_name=pname,
                        reason="60 Hz mains contamination detected",
                        confidence=0.95,
                        channel_type="generic",
                        context="artifact",
                    ))
        except Exception:
            pass
        return extras

    def accept_suggestion(
        self,
        suggestion: FilterSuggestion,
        channel: str,
        pipelines: Dict[str, "FilterPipeline"],
    ) -> bool:
        """Apply a suggestion and record it in the audit log."""
        preset = self._lib.get(suggestion.preset_name)
        if preset is None:
            return False
        pipelines[channel] = preset.pipeline.deep_copy()
        self._audit.record(
            channel, "accept_suggestion",
            preset_name=suggestion.preset_name,
            reason=suggestion.reason,
            confidence=suggestion.confidence,
        )
        return True


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def pipeline_from_legacy_params(
    low: float = 0.0,
    high: float = 0.0,
    order: int = 4,
    filter_type: str = "butterworth",
    notch_hz: float = 0.0,
    enabled: bool = True,
) -> FilterPipeline:
    """Build a ``FilterPipeline`` from flat legacy parameters.

    This bridges the old ``channel_filter_params`` dict format (used by
    ``DisplayProcessingProfile``) to the new composable engine.
    """
    filters: List[BaseFilter] = []
    lo = float(low) if low else 0.0
    hi = float(high) if high else 0.0

    if lo > 0 and hi > 0:
        filters.append(BandpassFilter(low_hz=lo, high_hz=hi, order=order,
                                      filter_type=filter_type))
    elif lo > 0:
        filters.append(HighpassFilter(cutoff_hz=lo, order=order,
                                      filter_type=filter_type))
    elif hi > 0:
        filters.append(LowpassFilter(cutoff_hz=hi, order=order,
                                     filter_type=filter_type))

    if notch_hz > 0:
        filters.append(NotchFilter(freq_hz=notch_hz))

    return FilterPipeline(filters=filters, enabled=enabled)


# ---------------------------------------------------------------------------
# Favorite presets
# ---------------------------------------------------------------------------

class FavoritePresets:
    """Manage a set of starred / favorite preset names persisted as JSON.

    Favorites are stored independently from the preset library itself so they
    survive import/export of preset files and user profile changes.
    """

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self._path = Path(path) if path else Path("config/favorite_presets.json")
        self._names: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._names = set(data)
            except Exception:
                logger.warning("Could not load favorite presets from %s", self._path)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(sorted(self._names), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def names(self) -> set[str]:
        return set(self._names)

    def is_favorite(self, name: str) -> bool:
        return name in self._names

    def toggle(self, name: str) -> bool:
        """Toggle favorite status. Returns ``True`` if *name* is now a favorite."""
        if name in self._names:
            self._names.discard(name)
            self.save()
            return False
        self._names.add(name)
        self.save()
        return True

    def add(self, name: str) -> None:
        self._names.add(name)
        self.save()

    def remove(self, name: str) -> None:
        self._names.discard(name)
        self.save()

    def clear(self) -> None:
        self._names.clear()
        self.save()

    def to_list(self) -> List[str]:
        return sorted(self._names)


# ---------------------------------------------------------------------------
# Channel annotations
# ---------------------------------------------------------------------------

@dataclass
class ChannelAnnotation:
    """Per-channel text annotation (note, tag, comment).

    Annotations are displayed in the filter dialog and included in reports.
    """
    channel: str
    text: str
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    author: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "text": self.text,
            "timestamp": self.timestamp,
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChannelAnnotation":
        return cls(
            channel=d["channel"],
            text=d.get("text", ""),
            timestamp=d.get("timestamp", ""),
            author=d.get("author", ""),
        )


class ChannelAnnotationStore:
    """Persistent store for per-channel annotations (JSON-backed)."""

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self._path = Path(path) if path else Path("config/channel_annotations.json")
        self._data: Dict[str, ChannelAnnotation] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for ch, d in raw.items():
                    self._data[ch] = ChannelAnnotation.from_dict(d)
            except Exception:
                logger.warning("Could not load channel annotations from %s", self._path)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {ch: ann.to_dict() for ch, ann in self._data.items()}
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get(self, channel: str) -> Optional[ChannelAnnotation]:
        return self._data.get(channel)

    def get_text(self, channel: str) -> str:
        ann = self._data.get(channel)
        return ann.text if ann else ""

    def set(self, channel: str, text: str, author: str = "") -> ChannelAnnotation:
        ann = ChannelAnnotation(channel=channel, text=text, author=author)
        self._data[channel] = ann
        self.save()
        return ann

    def delete(self, channel: str) -> None:
        self._data.pop(channel, None)
        self.save()

    def all(self) -> Dict[str, ChannelAnnotation]:
        return dict(self._data)

    def as_text_dict(self) -> Dict[str, str]:
        return {ch: ann.text for ch, ann in self._data.items()}

    def to_list(self) -> List[Dict[str, Any]]:
        return [ann.to_dict() for ann in self._data.values()]

    def clear(self) -> None:
        self._data.clear()
        self.save()
