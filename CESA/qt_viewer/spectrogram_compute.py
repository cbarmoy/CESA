"""Morlet wavelet time-frequency computation for the hypnogram spectrogram bars.

Two use cases share the same core routine:

- **Night** band: the whole recording, heavily decimated so the Morlet
  transform stays fast (a few seconds) and the result is cached until the
  channel / frequency band / scale changes.
- **Window** band: only the currently visible time range, higher time
  resolution, recomputed on navigation.

Both return ``(power, freqs, times)`` where ``power`` is ``(n_freqs, n_times)``
already converted to the requested display scale (dB / linear / z-score).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class SpectrogramResult:
    """Time-frequency power matrix ready for display."""

    power: np.ndarray  # (n_freqs, n_times), already scaled
    freqs: np.ndarray  # (n_freqs,)
    times: np.ndarray  # (n_times,) seconds from recording start
    vmin: float
    vmax: float


def _filter_band(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """Best-effort band-pass filter (falls back to the raw signal)."""
    try:
        from scipy.signal import butter, filtfilt

        nyq = fs / 2.0
        lo = max(0.1, float(low)) / nyq
        hi = min(float(high), nyq - 0.5) / nyq
        if 0.0 < lo < hi < 1.0:
            b, a = butter(4, [lo, hi], btype="band")
            return filtfilt(b, a, x)
    except Exception:
        logger.debug("Spectrogram band-pass filter failed", exc_info=True)
    return x


def _decimate_to(x: np.ndarray, fs: float, target_fs: float) -> Tuple[np.ndarray, float]:
    """Decimate *x* so its effective rate is close to *target_fs* (>= 1 Hz)."""
    if fs <= target_fs:
        return x, fs
    factor = max(1, int(round(fs / target_fs)))
    if factor <= 1:
        return x, fs
    try:
        from scipy.signal import decimate

        # decimate() misbehaves for very large factors; chain smaller steps.
        y = x
        remaining = factor
        while remaining > 10:
            y = decimate(y, 10, zero_phase=True)
            remaining = remaining // 10
        if remaining > 1:
            y = decimate(y, remaining, zero_phase=True)
        return y, fs / factor
    except Exception:
        logger.debug("Spectrogram decimation failed, falling back to slicing", exc_info=True)
        return x[::factor], fs / factor


def _scale_power(power: np.ndarray, scale: str) -> Tuple[np.ndarray, float, float]:
    """Convert raw Morlet power to the requested display scale.

    Returns ``(scaled, vmin, vmax)`` with vmin/vmax picked from the 5th/99th
    percentile so the colour map is robust to outliers.
    """
    power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

    if scale == "zscore":
        mu = power.mean(axis=1, keepdims=True)
        sigma = power.std(axis=1, keepdims=True)
        sigma[sigma < _EPS] = _EPS
        out = (power - mu) / sigma
        finite = out[np.isfinite(out)]
        if finite.size == 0:
            return out, -3.0, 3.0
        vmin, vmax = np.percentile(finite, [2, 98])
        return out, float(vmin), float(vmax)

    if scale == "linear":
        finite = power[np.isfinite(power)]
        if finite.size == 0:
            return power, 0.0, 1.0
        vmin, vmax = np.percentile(finite, [5, 99])
        return power, float(vmin), float(vmax)

    # default: dB
    out = 10.0 * np.log10(power + _EPS)
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return out, -120.0, -60.0
    vmin, vmax = np.percentile(finite, [5, 99])
    return out, float(vmin), float(vmax)


def compute_tfr(
    signal: np.ndarray,
    fs: float,
    *,
    fmin: float = 0.5,
    fmax: float = 30.0,
    n_freqs: int = 40,
    freq_log: bool = False,
    scale: str = "db",
    use_filtered: bool = False,
    target_fs: Optional[float] = None,
    target_cols: Optional[int] = None,
    time_offset_s: float = 0.0,
    onef_correction: bool = False,
) -> SpectrogramResult:
    """Compute a Morlet wavelet power spectrogram for *signal*.

    Parameters
    ----------
    signal, fs :
        Raw 1-D samples and their sampling rate (Hz).
    fmin, fmax, n_freqs, freq_log :
        Frequency axis definition.
    scale :
        ``"db"`` | ``"linear"`` | ``"zscore"``.
    use_filtered :
        Band-pass the signal to ``[fmin, fmax]`` before the transform.
    target_fs :
        Decimate the signal to approximately this rate before the Morlet
        transform (speeds up long/night computations). ``None`` disables it.
    target_cols :
        If set, additionally decimate the *output* time axis via MNE's
        ``decim`` so the resulting image has roughly this many columns.
    time_offset_s :
        Added to the returned time axis (e.g. to align a window slice with
        the full-night axis).
    onef_correction :
        Multiply power by ``freq`` (flattens the 1/f trend, highlights
        fast oscillations such as spindles).
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 8:
        raise ValueError("Signal trop court pour un spectrogramme.")

    work_fs = float(fs)
    if target_fs and target_fs > 0:
        x, work_fs = _decimate_to(x, work_fs, float(target_fs))

    if use_filtered:
        x = _filter_band(x, work_fs, fmin, fmax)

    fmax_eff = min(float(fmax), work_fs / 2.0 - 0.5)
    fmin_eff = max(0.1, float(fmin))
    if fmax_eff <= fmin_eff:
        fmax_eff = fmin_eff + 1.0

    if freq_log:
        freqs = np.geomspace(fmin_eff, fmax_eff, max(4, int(n_freqs)))
    else:
        freqs = np.linspace(fmin_eff, fmax_eff, max(4, int(n_freqs)))

    decim = 1
    if target_cols and target_cols > 0:
        decim = max(1, int(x.size / max(1, target_cols)))

    from mne.time_frequency import tfr_array_morlet

    n_cycles = np.maximum(freqs / 2.0, 3.0)
    arr = x[np.newaxis, np.newaxis, :]
    power = tfr_array_morlet(
        arr, sfreq=work_fs, freqs=freqs, n_cycles=n_cycles,
        output="power", decim=decim,
    )[0, 0]

    times = time_offset_s + np.arange(power.shape[1]) * (decim / work_fs)

    if onef_correction:
        power = power * freqs[:, np.newaxis]

    scaled, vmin, vmax = _scale_power(power, scale)
    return SpectrogramResult(power=scaled, freqs=freqs, times=times, vmin=vmin, vmax=vmax)


# -- Frequency-band presets (Hz) -----------------------------------------

FREQ_PRESETS = {
    "Large bande (0.5-30 Hz)": (0.5, 30.0),
    "Delta (0.5-4 Hz)": (0.5, 4.0),
    "Theta (4-8 Hz)": (4.0, 8.0),
    "Alpha (8-12 Hz)": (8.0, 12.0),
    "Sigma / fuseaux (11-16 Hz)": (11.0, 16.0),
    "Beta (16-30 Hz)": (16.0, 30.0),
    "Large bande (0.5-45 Hz)": (0.5, 45.0),
}
