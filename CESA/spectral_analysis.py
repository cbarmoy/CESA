# -*- coding: utf-8 -*-
"""
CESA v0.0beta1.0 - Spectral Analysis Utilities (FFT-based)
==================================================

Spectral analysis utilities for EEG Analysis Studio v0.0beta1.0
Développé pour l'Unité Neuropsychologie du Stress (IRBA)

Fournit des analyses spectrales robustes pour l'EEG :
- compute_psd_fft: Spectre de magnitude et axe fréquentiel
- compute_psd_welch: PSD via méthode Welch (µV²/Hz)
- compute_band_powers: Somme des magnitudes par bande EEG
- compute_stage_psd_fft_for_array: PSD par stades avec stats robustes
- compute_stage_psd_welch_for_array: PSD Welch par stades

Méthodes statistiques robustes:
- Médiane + SEM pour la robustesse aux valeurs aberrantes
- Égalisation optionnelle du nombre d'époques par stade
- Mappage automatique des stades de sommeil
- Export CSV pour analyses statistiques externes

Bandes EEG standards:
- Delta: 0.5-4 Hz (sommeil profond)
- Theta: 4-8 Hz (somnolence, méditation)
- Alpha: 8-12 Hz (relaxation, yeux fermés)
- Beta: 12-30 Hz (activité cognitive)
- Gamma: 30-45 Hz (traitement d'information)

Auteur: Côme Barmoy (IRBA)
Version: 0.0beta1.0
Date: 2025-09-26
"""

from typing import Dict, Tuple
import numpy as np
from scipy.signal import welch
import pandas as pd
from typing import Optional, List

EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (12.0, 30.0),
    "Gamma": (30.0, 45.0),
}

def _canonical_stage(stage: str) -> Optional[str]:
    """Map various stage labels to canonical codes: W, N1, N2, N3, R.
    Returns None for unknown/unhandled labels.
    """
    if stage is None:
        return None
    s = str(stage).strip().upper()
    mapping = {
        # Wake
        'W': 'W', 'WAKE': 'W', 'AWAKE': 'W', 'EVEIL': 'W', 'ÉVEIL': 'W', 'EVEIL ': 'W', 'EVEILL': 'W',
        # REM
        'R': 'R', 'REM': 'R', 'PARADOXAL': 'R', 'SOMMEIL PARADOXAL': 'R',
        # N1
        'N1': 'N1', 'S1': 'N1', 'NREM1': 'N1', 'STAGE1': 'N1', 'STAGE 1': 'N1',
        # N2
        'N2': 'N2', 'S2': 'N2', 'NREM2': 'N2', 'STAGE2': 'N2', 'STAGE 2': 'N2',
        # N3 (merge deep sleep labels)
        'N3': 'N3', 'S3': 'N3', 'S4': 'N3', 'NREM3': 'N3', 'NREM4': 'N3', 'STAGE3': 'N3', 'STAGE 3': 'N3', 'STAGE4': 'N3', 'STAGE 4': 'N3',
    }
    return mapping.get(s, None)

def detrend_mean(signal: np.ndarray) -> np.ndarray:
    """Remove DC offset (mean) from signal."""
    if signal.size == 0:
        return signal
    return signal - np.mean(signal)

def compute_psd_fft(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute single-sided FFT magnitude spectrum of a 1D signal.

    Args:
        signal: 1D array of samples (µV).
        fs: Sampling frequency in Hz.

    Returns:
        freqs: Frequency axis (Hz) for rFFT (length N//2+1)
        spectrum: Magnitude spectrum (absolute value of rFFT)
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")

    x = detrend_mean(signal.astype(float))
    n = x.size
    if n < 2:
        return np.array([]), np.array([])

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x))
    return freqs, spectrum

def compute_band_powers(freqs: np.ndarray, spectrum: np.ndarray, bands: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
    """
    Sum magnitudes within band limits.

    Args:
        freqs: Frequency axis (Hz)
        spectrum: Magnitude spectrum
        bands: Dict of band name -> (f_min, f_max)

    Returns:
        Dict of band name -> summed magnitude in band
    """
    if bands is None:
        bands = EEG_BANDS
    if freqs.size == 0 or spectrum.size == 0:
        return {name: 0.0 for name in bands.keys()}

    band_power: Dict[str, float] = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power[name] = float(np.sum(spectrum[mask]))
    return band_power

def compute_peak_and_centroid(freqs: np.ndarray, spectrum: np.ndarray) -> Tuple[float, float]:
    """Compute dominant frequency and spectral centroid."""
    if freqs.size == 0 or spectrum.size == 0:
        return float("nan"), float("nan")
    peak_freq = float(freqs[int(np.argmax(spectrum))])
    denom = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / denom) if denom > 0 else float("nan")
    return peak_freq, centroid

def compute_psd_welch(signal: np.ndarray, fs: float, nperseg: int = 1024, noverlap: int = 512, window: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
    """Compute PSD using Welch (µV^2/Hz)."""
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    x = detrend_mean(signal.astype(float))
    if x.size < 2:
        return np.array([]), np.array([])
    freqs, psd = welch(x, fs=fs, nperseg=min(nperseg, len(x)), noverlap=min(noverlap, max(0, len(x)//2 - 1)), window=window)
    return freqs, psd


def _apply_optional_filters(x: np.ndarray, fs: float, band: Optional[Tuple[float, float]], notch_hz: Optional[float]) -> np.ndarray:
    y = x
    try:
        if band is not None and band[0] is not None and band[1] is not None:
            from scipy.signal import butter, filtfilt
            low, high = float(band[0]), float(band[1])
            low = max(0.0, low)
            high = min(high, fs/2.0 - 1e-6)
            if high > low and high > 0.0:
                b, a = butter(4, [low/(fs/2.0), high/(fs/2.0)], btype='band')
                y = filtfilt(b, a, y)
    except Exception:
        pass
    try:
        if notch_hz is not None and notch_hz > 0.0:
            from scipy.signal import iirnotch, filtfilt
            w0 = float(notch_hz) / (fs/2.0)
            b, a = iirnotch(w0, Q=30.0)
            y = filtfilt(b, a, y)
    except Exception:
        pass
    return y


def compute_stage_psd_welch_for_array(
    signal: np.ndarray,
    fs: float,
    scoring_df: Optional[pd.DataFrame],
    epoch_len: float = 30.0,
    stages: Optional[List[str]] = None,
    fmin: float = 0.5,
    fmax: float = 30.0,
    nperseg_sec: float = 4.0,
    band_filter: Optional[Tuple[float, float]] = (0.3, 40.0),
    notch_hz: Optional[float] = 50.0,
    equalize_epochs: bool = True,
    robust_stats: bool = True,
    normalize_relative: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    SleepEEGpy-style PSD by stage using Welch.

    Returns dict: stage -> (freqs, mean_or_median_psd, sem_or_mad_sem, n_epochs)
    PSD units in µV^2/Hz.
    """
    if stages is None:
        stages = ["W", "N1", "N2", "N3", "R"]
    fs = float(fs)
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    n = len(signal)
    segs: Dict[str, List[np.ndarray]] = {s: [] for s in stages}

    if scoring_df is None or len(scoring_df) == 0:
        segs["W"].append(signal.copy())
    else:
        eplen = float(epoch_len)
        for _, row in scoring_df.iterrows():
            st_raw = row.get('stage', '')
            st = _canonical_stage(st_raw)
            if st is None or st not in segs:
                continue
            t0 = float(row.get('time', 0.0))
            i0 = int(max(0, min(n-1, round(t0 * fs))))
            i1 = int(max(i0+1, min(n, round((t0 + eplen) * fs))))
            seg = signal[i0:i1]
            if len(seg) >= int(max(8, round(nperseg_sec * fs))):
                segs[st].append(seg)

    # Equalize number of epochs per stage
    if equalize_epochs:
        counts = [len(v) for v in segs.values() if len(v) > 0]
        if counts:
            m = int(min(counts))
            for k in segs.keys():
                if len(segs[k]) > m:
                    segs[k] = segs[k][:m]

    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    eps = np.finfo(float).eps
    target_nper = int(max(8, round(nperseg_sec * fs)))

    for s in stages:
        chunks = segs.get(s, [])
        if not chunks:
            continue
        psd_rows: List[np.ndarray] = []
        freqs_ref: Optional[np.ndarray] = None
        for seg in chunks:
            x = detrend_mean(seg.astype(float))
            x = _apply_optional_filters(x, fs, band_filter, notch_hz)
            if len(x) < target_nper:
                continue
            f, p = welch(x, fs=fs, window='hann', nperseg=target_nper, noverlap=target_nper//2,
                         detrend='constant', return_onesided=True, scaling='density')
            mask = (f >= fmin) & (f <= fmax)
            f = f[mask]
            p = p[mask] * 1e12  # V^2/Hz -> µV^2/Hz
            if normalize_relative:
                denom = float(np.sum(p)) + eps
                p = p / denom
            if freqs_ref is None:
                freqs_ref = f
            psd_rows.append(p)
        if psd_rows and freqs_ref is not None:
            arr = np.vstack(psd_rows)
            if robust_stats:
                med = np.median(arr, axis=0)
                mad = np.median(np.abs(arr - med), axis=0)
                sem_vals = 1.4826 * mad / max(1, np.sqrt(arr.shape[0]))
                mean_vals = med
            else:
                mean_vals = np.mean(arr, axis=0)
                sem_vals = np.std(arr, axis=0, ddof=1) / max(1, np.sqrt(arr.shape[0]))
            results[s] = (freqs_ref, mean_vals, sem_vals, int(arr.shape[0]))
    return results


def compute_stage_psd_fft_for_array(
    signal: np.ndarray,
    fs: float,
    scoring_df: Optional[pd.DataFrame],
    epoch_len: float = 30.0,
    stages: Optional[List[str]] = None,
    fmin: float = 0.5,
    fmax: float = 45.0,
    equalize_epochs: bool = True,
    robust_stats: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    PSD par stade adaptée de Analyse_spectrale.py (FFT magnitude, DC enlevée).

    - Segmente le signal selon le scoring (ou tout en W si scoring absent)
    - Calcule le spectre rFFT (magnitude) par époque
    - Restreint aux fréquences [fmin, fmax]
    - Agrège par stade (médiane+SEM robuste via MAD, ou moyenne+SEM)

    Retourne: dict[stade] -> (freqs, mean_or_median_spectrum, sem_or_mad_sem, n_epochs)
    """
    if stages is None:
        stages = ["W", "N1", "N2", "N3", "R"]
    fs = float(fs)
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")

    n = len(signal)
    segs: Dict[str, List[np.ndarray]] = {s: [] for s in stages}

    if scoring_df is None or len(scoring_df) == 0:
        segs["W"].append(signal.copy())
    else:
        eplen = float(epoch_len)
        for _, row in scoring_df.iterrows():
            st_raw = row.get('stage', '')
            st = _canonical_stage(st_raw)
            if st is None or st not in segs:
                continue
            t0 = float(row.get('time', 0.0))
            i0 = int(max(0, min(n-1, round(t0 * fs))))
            i1 = int(max(i0+1, min(n, round((t0 + eplen) * fs))))
            seg = signal[i0:i1]
            if len(seg) >= int(max(8, round(1.0 * fs))):  # >= 1s
                segs[st].append(seg)

    # Égaliser le nombre d'époques par stade si demandé
    if equalize_epochs:
        counts = [len(v) for v in segs.values() if len(v) > 0]
        if counts:
            m = int(min(counts))
            for k in segs.keys():
                if len(segs[k]) > m:
                    segs[k] = segs[k][:m]

    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    eps = np.finfo(float).eps

    for s in stages:
        chunks = segs.get(s, [])
        if not chunks:
            continue
        rows: List[np.ndarray] = []
        freqs_ref: Optional[np.ndarray] = None
        for seg in chunks:
            x = detrend_mean(seg.astype(float))
            f, spec = compute_psd_fft(x, fs)
            if f.size == 0:
                continue
            mask = (f >= fmin) & (f <= fmax)
            f = f[mask]
            spec = spec[mask]
            if freqs_ref is None:
                freqs_ref = f
            rows.append(spec)
        if rows and freqs_ref is not None:
            arr = np.vstack(rows)
            if robust_stats:
                med = np.median(arr, axis=0)
                mad = np.median(np.abs(arr - med), axis=0)
                sem_vals = 1.4826 * mad / max(1, np.sqrt(arr.shape[0]))
                mean_vals = med
            else:
                mean_vals = np.mean(arr, axis=0)
                sem_vals = np.std(arr, axis=0, ddof=1) / max(1, np.sqrt(arr.shape[0]))
            results[s] = (freqs_ref, mean_vals, sem_vals, int(arr.shape[0]))
    return results
