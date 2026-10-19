"""Night-level signal quality heuristics (no PyPREP dependency).

Flags flat / NaN / extreme-amplitude channels, estimates line-noise ratio
(50/60 Hz) and SNR (0.5–30 Hz vs >=50 Hz).
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.signal import welch

logger = logging.getLogger(__name__)

# Non-EEG auxiliaries often mis-typed as EEG because of electrode prefixes (C3 Impédance).
_EXCLUDE_NAME_TOKENS = (
    "imp",
    "impedance",
    "spo2",
    "sao2",
    "pleth",
    "status",
    "marker",
    "event",
    "battery",
    "oximet",
)


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def should_exclude_channel_name(name: str) -> bool:
    """True for impedance / SpO2 / aux channels that must not enter EEG QC."""
    low = _strip_accents(str(name)).lower()
    return any(tok in low for tok in _EXCLUDE_NAME_TOKENS)


@dataclass
class ChannelQcScore:
    name: str
    channel_type: str = "eeg"
    snr_db: float = 0.0
    line_noise_ratio: float = 0.0
    is_flat: bool = False
    has_nan: bool = False
    extreme_amp: bool = False
    p99_9_uv: float = 0.0
    frac_hi: float = 0.0
    reasons: List[str] = field(default_factory=list)

    @property
    def is_bad(self) -> bool:
        return bool(self.reasons)


@dataclass
class NightQcReport:
    per_channel: List[ChannelQcScore] = field(default_factory=list)
    bad_channels: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_channels: int = 0
    thresholds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_channels": self.n_channels,
            "bad_channels": list(self.bad_channels),
            "warnings": list(self.warnings),
            "thresholds": dict(self.thresholds),
            "per_channel": [
                {
                    "name": c.name,
                    "channel_type": c.channel_type,
                    "snr_db": c.snr_db,
                    "line_noise_ratio": c.line_noise_ratio,
                    "is_flat": c.is_flat,
                    "has_nan": c.has_nan,
                    "extreme_amp": c.extreme_amp,
                    "p99_9_uv": c.p99_9_uv,
                    "frac_hi": c.frac_hi,
                    "reasons": list(c.reasons),
                    "is_bad": c.is_bad,
                }
                for c in self.per_channel
            ],
        }


def _channel_snr_and_line(
    data: np.ndarray,
    sfreq: float,
    *,
    line_hz: float = 50.0,
    epoch_len: float = 30.0,
) -> tuple[float, float]:
    x = np.asarray(data, dtype=float).ravel()
    if x.size < 8 or sfreq <= 0:
        return 0.0, 0.0
    nperseg = min(len(x), max(16, int(sfreq * epoch_len)))
    try:
        freqs, psd = welch(x, fs=sfreq, nperseg=nperseg)
    except Exception:
        return 0.0, 0.0
    sig_mask = (freqs >= 0.5) & (freqs <= 30.0)
    noise_mask = freqs >= 50.0
    sig_power = float(np.mean(psd[sig_mask])) if sig_mask.any() else 1e-12
    noise_power = float(np.mean(psd[noise_mask])) if noise_mask.any() else 1e-12
    snr_db = 10.0 * np.log10(max(sig_power, 1e-12) / max(noise_power, 1e-12))

    band = (freqs >= line_hz - 1.0) & (freqs <= line_hz + 1.0)
    line_p = float(np.mean(psd[band])) if band.any() else 0.0
    ratio = line_p / max(sig_power, 1e-12)
    return float(snr_db), float(ratio)


def compute_night_qc(
    derivations: Dict[str, np.ndarray],
    sfreq: float,
    channel_types: Optional[Dict[str, str]] = None,
    *,
    include_types: Sequence[str] = ("eeg",),
    flat_std_uv: float = 0.5,
    extreme_amp_uv: float = 1500.0,
    extreme_frac_min: float = 0.001,
    snr_warn_db: float = 5.0,
    line_noise_warn: float = 2.0,
    line_hz: float = 50.0,
    max_samples: int = 600_000,
) -> NightQcReport:
    """Compute per-channel QC scores for a night recording (arrays in µV).

    Extreme amplitude uses robust stats: flag only if the 99.9th percentile of
    |x| exceeds *extreme_amp_uv* **and** at least *extreme_frac_min* of samples
    exceed that threshold (avoids single ADC-clip spikes ~3276 µV).
    """
    types = channel_types or {}
    report = NightQcReport(
        thresholds={
            "flat_std_uv": float(flat_std_uv),
            "extreme_amp_uv": float(extreme_amp_uv),
            "extreme_frac_min": float(extreme_frac_min),
            "snr_warn_db": float(snr_warn_db),
            "line_noise_warn": float(line_noise_warn),
            "line_hz": float(line_hz),
        }
    )
    names = list(derivations.keys())
    selected = [
        n for n in names
        if (not should_exclude_channel_name(n))
        and types.get(n, "eeg") in include_types
    ]
    # Do NOT fall back to all channels (would reintroduce Imp/SpO2).
    if not selected:
        report.n_channels = 0
        report.warnings.append(
            "Aucun canal EEG eligible pour le QC "
            "(apres exclusion Impedance/SpO2/aux)."
        )
        return report

    report.n_channels = len(selected)

    for name in selected:
        ch_type = types.get(name, "eeg")
        arr = np.asarray(derivations[name], dtype=float).ravel()
        if arr.size > max_samples:
            idx = np.linspace(0, arr.size - 1, max_samples).astype(int)
            arr = arr[idx]
        score = ChannelQcScore(name=name, channel_type=ch_type)
        if arr.size == 0:
            score.reasons.append("empty")
            report.per_channel.append(score)
            continue
        if not np.all(np.isfinite(arr)):
            score.has_nan = True
            score.reasons.append("NaN/Inf")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        std = float(np.std(arr))
        if std < flat_std_uv:
            score.is_flat = True
            score.reasons.append(f"flat (std={std:.3f} uV)")

        abs_x = np.abs(arr)
        p99_9 = float(np.percentile(abs_x, 99.9)) if abs_x.size else 0.0
        thr = float(extreme_amp_uv)
        frac_hi = float(np.mean(abs_x > thr)) if abs_x.size else 0.0
        score.p99_9_uv = p99_9
        score.frac_hi = frac_hi
        if p99_9 > thr and frac_hi >= float(extreme_frac_min):
            score.extreme_amp = True
            score.reasons.append(
                f"extreme amp (p99.9={p99_9:.0f} uV, frac={100.0 * frac_hi:.2f}%)"
            )

        snr, line_r = _channel_snr_and_line(arr, float(sfreq), line_hz=line_hz)
        score.snr_db = snr
        score.line_noise_ratio = line_r
        if snr < snr_warn_db:
            score.reasons.append(f"SNR bas ({snr:.1f} dB)")
        if line_r > line_noise_warn:
            score.reasons.append(f"bruit {line_hz:.0f} Hz (ratio={line_r:.2f})")
        report.per_channel.append(score)

    report.bad_channels = [c.name for c in report.per_channel if c.is_bad]
    if report.bad_channels:
        report.warnings.append(
            f"{len(report.bad_channels)} canal(aux) suspect(s): "
            + ", ".join(report.bad_channels[:8])
            + ("…" if len(report.bad_channels) > 8 else "")
        )
    flat_n = sum(1 for c in report.per_channel if c.is_flat)
    if flat_n:
        report.warnings.append(f"{flat_n} canal(aux) plat(s)")
    line_n = sum(
        1 for c in report.per_channel
        if c.line_noise_ratio > line_noise_warn
    )
    if line_n:
        report.warnings.append(f"{line_n} canal(aux) avec bruit secteur eleve")
    return report
