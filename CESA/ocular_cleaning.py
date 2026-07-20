"""EOG → EEG ocular artifact cleaning via linear regression.

Non-destructive helpers: operate on numpy arrays only; never write files.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]

try:
    import mne
except ImportError:  # pragma: no cover
    mne = None  # type: ignore[assignment]

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _bandpass(signal: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    """Band-pass a 1-D signal; fall back to demean if mne is unavailable."""
    x = np.asarray(signal, dtype=float).ravel()
    lo, hi = float(band[0]), float(band[1])
    if mne is not None and x.size >= 8 and sfreq > 0 and hi > lo:
        try:
            nyq = 0.5 * float(sfreq)
            h_freq = min(hi, nyq * 0.99)
            l_freq = max(lo, 0.0)
            if h_freq <= l_freq:
                return x - np.mean(x)
            return mne.filter.filter_data(
                x, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose="ERROR"
            )
        except Exception:
            logger.debug("EOG bandpass failed; using demean", exc_info=True)
    return x - np.mean(x)


def _align_length(*arrays: np.ndarray) -> List[np.ndarray]:
    n = min(a.size for a in arrays)
    return [np.asarray(a, dtype=float).ravel()[:n] for a in arrays]


def regress_eog_from_eeg(
    eeg: np.ndarray,
    eog_refs: Sequence[np.ndarray],
    sfreq: float,
    eog_band: Tuple[float, float] = (0.1, 10.0),
) -> np.ndarray:
    """Remove EOG contribution from a single EEG channel via least squares.

    Parameters
    ----------
    eeg :
        1-D EEG samples.
    eog_refs :
        One or more EOG reference channels (same sampling rate).
    sfreq :
        Sampling frequency (Hz).
    eog_band :
        Band used to condition EOG regressors before fitting.

    Returns
    -------
    cleaned :
        EEG residual after subtracting the linear EOG projection.
    """
    y = np.asarray(eeg, dtype=float).ravel()
    if not eog_refs or y.size < 8:
        return y.copy()

    refs = [_bandpass(r, sfreq, eog_band) for r in eog_refs]
    aligned = _align_length(y, *refs)
    y_a = aligned[0]
    X = np.column_stack(aligned[1:])
    # Add intercept so DC offset is not forced into betas
    X_design = np.column_stack([X, np.ones(len(y_a))])
    try:
        beta, *_ = np.linalg.lstsq(X_design, y_a, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("EOG regression lstsq failed; returning original EEG")
        return y.copy()

    cleaned = y_a - X_design @ beta
    out = y.copy()
    out[: cleaned.size] = cleaned
    return out


def clean_channels(
    signals: Dict[str, Signal],
    eeg_names: Iterable[str],
    eog_names: Iterable[str],
    sfreq: Optional[float] = None,
    eog_band: Tuple[float, float] = (0.1, 10.0),
) -> Dict[str, np.ndarray]:
    """Clean selected EEG channels using selected EOG references.

    Returns a dict of ``{eeg_name: cleaned_array}`` only for EEG channels
    that exist in ``signals``. EOG / EMG / ECG channels are never returned.
    """
    eeg_list = [n for n in eeg_names if n in signals]
    eog_list = [n for n in eog_names if n in signals]
    if not eeg_list or not eog_list:
        return {}

    if sfreq is None:
        sfreq = float(signals[eeg_list[0]][1])

    eog_arrays = [np.asarray(signals[n][0], dtype=float) for n in eog_list]
    cleaned: Dict[str, np.ndarray] = {}
    for name in eeg_list:
        data, fs = signals[name]
        fs_use = float(fs) if fs else float(sfreq)
        cleaned[name] = regress_eog_from_eeg(
            np.asarray(data, dtype=float),
            eog_arrays,
            fs_use,
            eog_band=eog_band,
        )
    return cleaned


def bandpower(
    x: np.ndarray,
    sfreq: float,
    band: Tuple[float, float] = (0.3, 4.0),
) -> float:
    """Welch band power for a 1-D signal."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 8 or sfreq <= 0:
        return float("nan")
    try:
        from scipy.signal import welch

        nperseg = min(len(x), max(64, int(sfreq)))
        freqs, psd = welch(x, fs=sfreq, nperseg=nperseg)
        lo, hi = band
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any() or _trapz is None:
            return float(np.var(x))
        return float(_trapz(psd[mask], freqs[mask]))
    except Exception:
        return float(np.var(x))


def ocular_bandpower_ratio(
    before: np.ndarray,
    after: np.ndarray,
    sfreq: float,
    band: Tuple[float, float] = (0.3, 4.0),
) -> float:
    """Relative change in ocular-band power: ``(after - before) / before``.

    Negative values mean power decreased (desired). Returns 0.0 if before
    power is non-positive / non-finite.
    """
    p0 = bandpower(before, sfreq, band)
    p1 = bandpower(after, sfreq, band)
    if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
        return 0.0
    return float((p1 - p0) / p0)


def slice_signal(
    data: np.ndarray,
    sfreq: float,
    t0: float,
    duration: float,
) -> np.ndarray:
    """Extract ``[t0, t0+duration]`` samples from a 1-D array."""
    x = np.asarray(data, dtype=float).ravel()
    if sfreq <= 0:
        return x.copy()
    i0 = max(0, int(round(t0 * sfreq)))
    i1 = min(x.size, int(round((t0 + duration) * sfreq)))
    if i1 <= i0:
        return np.asarray([], dtype=float)
    return x[i0:i1].copy()
