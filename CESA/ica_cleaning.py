"""Non-destructive MNE ICA cleaning helpers (in-memory only).

Input arrays are expected in µV (CESA derivations). Temporary RawArray uses
Volts for MNE convention; cleaned outputs are returned in µV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import welch

logger = logging.getLogger(__name__)


@dataclass
class IcaFitResult:
    """Opaque ICA fit handle + channel order used for fitting."""

    ica: object
    raw: object
    ch_names: List[str]
    sfreq: float
    explained_var: List[float] = field(default_factory=list)
    n_components: int = 0
    sources: Optional[np.ndarray] = None  # (n_ic, n_times) cached


def _require_mne():
    try:
        import mne
        from mne.preprocessing import ICA
        return mne, ICA
    except ImportError as exc:
        raise ImportError(
            "MNE est requis pour le nettoyage ICA (pip install mne)."
        ) from exc


def fit_ica(
    data_uv: Dict[str, np.ndarray],
    sfreq: float,
    *,
    n_components: Optional[int] = None,
    method: str = "fastica",
    random_state: int = 42,
    max_samples: int = 300_000,
) -> IcaFitResult:
    """Fit ICA on EEG channels (µV dict). Returns fit handle."""
    mne, ICA = _require_mne()
    if not data_uv:
        raise ValueError("Aucun canal EEG pour ICA.")
    ch_names = list(data_uv.keys())
    arrays = []
    n_min = None
    for name in ch_names:
        arr = np.asarray(data_uv[name], dtype=np.float64).ravel()
        n_min = len(arr) if n_min is None else min(n_min, len(arr))
        arrays.append(arr)
    n = int(n_min or 0)
    if n < int(sfreq) * 2:
        raise ValueError("Signal trop court pour ICA.")
    if n > max_samples:
        # Decimate uniformly for fit speed
        idx = np.linspace(0, n - 1, max_samples).astype(int)
        data_v = np.vstack([a[idx] for a in arrays]) / 1e6
        # For apply we keep full-length Raw separately — store full Volts raw
        full_v = np.vstack([a[:n] for a in arrays]) / 1e6
    else:
        data_v = np.vstack([a[:n] for a in arrays]) / 1e6
        full_v = data_v

    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types="eeg")
    raw_fit = mne.io.RawArray(data_v, info, verbose=False)
    raw_full = mne.io.RawArray(full_v, info, verbose=False)

    n_ch = len(ch_names)
    if n_components is None:
        n_components = min(n_ch, 15)
    n_components = max(1, min(int(n_components), n_ch))

    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw_fit, verbose=False)

    explained = _per_component_var_pct(ica, raw_fit, n_components)

    return IcaFitResult(
        ica=ica,
        raw=raw_full,
        ch_names=ch_names,
        sfreq=float(sfreq),
        explained_var=explained,
        n_components=n_components,
    )


def _per_component_var_pct(ica, raw, n_components: int) -> List[float]:
    """Percent variance accounted for per IC (MNE pvaf-style).

    Newer MNE dropped ``pca_explained_variance_ratio_``; use
    ``get_explained_variance_ratio(inst, components=i)`` when available,
    else fall back to relative variance of ICA sources.
    """
    explained: List[float] = []
    # Preferred: MNE >= 1.2 API
    getter = getattr(ica, "get_explained_variance_ratio", None)
    if callable(getter):
        try:
            for i in range(int(n_components)):
                ratios = getter(raw, components=i)
                if isinstance(ratios, dict) and ratios:
                    explained.append(100.0 * float(sum(ratios.values())))
                else:
                    explained.append(100.0 * float(ratios))
            if explained and any(v > 0 for v in explained):
                return explained
        except Exception as exc:
            logger.debug("get_explained_variance_ratio failed: %s", exc)
            explained = []

    # Legacy attribute (older MNE)
    ratios = getattr(ica, "pca_explained_variance_ratio_", None)
    if ratios is not None:
        try:
            return [
                float(x) * 100.0
                for x in np.asarray(ratios).ravel()[:n_components]
            ]
        except Exception:
            pass

    # Fallback: relative source variance
    try:
        src = np.asarray(ica.get_sources(raw).get_data(), dtype=np.float64)
        vars_ = np.var(src[:n_components], axis=1)
        total = float(np.sum(vars_)) + 1e-20
        return [float(100.0 * v / total) for v in vars_]
    except Exception as exc:
        logger.debug("source-variance fallback failed: %s", exc)
        return []


def get_sources(
    fit: IcaFitResult,
    *,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    max_samples: int = 300_000,
) -> np.ndarray:
    """Return ICA sources as (n_ic, n_times).

    If *tmin*/*tmax* are set, crop to that window (no cache — for viewer
    previews). Otherwise cache a (possibly subsampled) full-night array.
    """
    if tmin is None and tmax is None and fit.sources is not None:
        return fit.sources

    raw = fit.raw
    if tmin is not None or tmax is not None:
        t0 = float(tmin) if tmin is not None else 0.0
        t1 = float(tmax) if tmax is not None else float(raw.times[-1])
        t0 = max(0.0, t0)
        t1 = max(t0 + 1.0 / max(fit.sfreq, 1.0), t1)
        try:
            raw = raw.copy().crop(tmin=t0, tmax=t1, include_tmax=False)
        except Exception:
            # Fallback: index into full sources
            full = get_sources(fit, max_samples=10**9)
            i0 = int(t0 * fit.sfreq)
            i1 = int(t1 * fit.sfreq)
            return full[:, i0:i1]

    sources = fit.ica.get_sources(raw)
    data = np.asarray(sources.get_data(), dtype=np.float64)
    if tmin is None and tmax is None:
        if data.shape[1] > max_samples:
            idx = np.linspace(0, data.shape[1] - 1, max_samples).astype(int)
            data = data[:, idx]
        fit.sources = data
    return data


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(a.size, b.size)
    if n < 8:
        return 0.0
    a = a[:n]
    b = b[:n]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def component_stats(
    fit: IcaFitResult,
    *,
    eog_uv: Optional[np.ndarray] = None,
    ecg_uv: Optional[np.ndarray] = None,
    corr_hint_thr: float = 0.3,
) -> List[Dict[str, float | str | int]]:
    """Per-IC kurtosis, variance %, |corr| with EOG/ECG, and label hint."""
    sources = get_sources(fit)
    n_ic = min(fit.n_components, sources.shape[0])
    out: List[Dict[str, float | str | int]] = []

    eog = None if eog_uv is None else np.asarray(eog_uv, dtype=float).ravel()
    ecg = None if ecg_uv is None else np.asarray(ecg_uv, dtype=float).ravel()

    for i in range(n_ic):
        src = sources[i]
        # Excess kurtosis (Fisher): normal ~ 0
        m = float(np.mean(src))
        s = float(np.std(src)) + 1e-12
        z = (src - m) / s
        kurt = float(np.mean(z ** 4) - 3.0)
        var_pct = float(fit.explained_var[i]) if i < len(fit.explained_var) else 0.0
        corr_eog = abs(_pearson(src, eog)) if eog is not None else 0.0
        corr_ecg = abs(_pearson(src, ecg)) if ecg is not None else 0.0
        hint = "other"
        if corr_eog >= corr_hint_thr and corr_eog >= corr_ecg:
            hint = "eog"
        elif corr_ecg >= corr_hint_thr:
            hint = "ecg"
        out.append({
            "index": i,
            "kurtosis": kurt,
            "var_pct": var_pct,
            "corr_eog": corr_eog,
            "corr_ecg": corr_ecg,
            "label_hint": hint,
        })
    return out


def component_spectrum(
    source: np.ndarray,
    sfreq: float,
    *,
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD of one IC source. Returns (freqs, psd)."""
    x = np.asarray(source, dtype=float).ravel()
    if x.size < 8 or sfreq <= 0:
        return np.asarray([0.0]), np.asarray([0.0])
    nps = nperseg or min(len(x), max(64, int(sfreq * 4.0)))
    freqs, psd = welch(x, fs=float(sfreq), nperseg=nps)
    return freqs, psd


def preview_exclude(
    fit: IcaFitResult,
    exclude: Sequence[int],
    *,
    channels: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Apply ICA exclusion and return cleaned channels in µV."""
    ica = fit.ica
    raw = fit.raw.copy()
    excl = sorted({int(i) for i in exclude if 0 <= int(i) < fit.n_components})
    ica.exclude = excl
    cleaned_raw = ica.apply(raw, verbose=False)
    picks = list(channels) if channels else list(fit.ch_names)
    out: Dict[str, np.ndarray] = {}
    data = cleaned_raw.get_data(picks=picks)  # Volts
    for i, name in enumerate(picks):
        if name not in fit.ch_names:
            continue
        out[name] = np.asarray(data[i], dtype=np.float64) * 1e6
    return out


def component_labels(fit: IcaFitResult, stats: Optional[Sequence[dict]] = None) -> List[str]:
    """Human-readable ICA component labels (optionally enriched with stats)."""
    labels = []
    stats_by_i = {}
    if stats:
        for st in stats:
            stats_by_i[int(st["index"])] = st
    for i in range(fit.n_components):
        st = stats_by_i.get(i)
        if st is not None:
            hint = st.get("label_hint") or "other"
            hint_txt = f" ~{hint}" if hint != "other" else ""
            labels.append(
                f"IC{i}  var={float(st['var_pct']):.0f}%  "
                f"kurt={float(st['kurtosis']):.1f}  "
                f"|r_EOG|={float(st['corr_eog']):.2f}"
                f"{hint_txt}"
            )
        elif i < len(fit.explained_var):
            labels.append(f"IC{i} ({fit.explained_var[i]:.1f}%)")
        else:
            labels.append(f"IC{i}")
    return labels
