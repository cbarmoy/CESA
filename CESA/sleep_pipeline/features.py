"""Feature extraction for sleep-stage classification.

Each 30-s epoch is characterised by a set of interpretable features drawn
from time-domain, frequency-domain and cross-channel statistics.  These
features are used by both the rule-based (AASM) scorer and the ML model.

Feature groups
--------------
* **Band powers** (delta 0.5-4 Hz, theta 4-8, alpha 8-12, sigma 12-16,
  beta 16-30) via Welch PSD.
* **Ratios**: delta/beta, theta/alpha, sigma/total.
* **Time-domain**: variance, RMS, zero-crossing rate, Hjorth parameters.
* **Spectral entropy**: Shannon entropy of normalised PSD -- high in
  Wake/REM (broadband), low in N3 (narrow delta peak).
* **Spindle detection** (sigma-band bursts 0.5-2 s) -- key for N2.
* **K-complex detection** (sharp delta deflections >75 uV) -- key for N2.
* **Delta amplitude fraction** (AASM N3 criterion: >75 uV slow waves).
* **EOG**: REM burst detection, slow eye movement index.
* **EMG**: tonic level, phasic burst count.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, hilbert

from .preprocessing import EpochedSignals


# ---------------------------------------------------------------------------
# Band definitions (Hz)
# ---------------------------------------------------------------------------

BANDS: Dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def _band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    """Absolute band power (uV^2) between *fmin* and *fmax*."""
    mask = (freqs >= fmin) & (freqs < fmax)
    if not mask.any():
        return 0.0
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    _trapz = getattr(np, "trapezoid", np.trapz)
    return float(_trapz(psd[mask], dx=df))


def _zero_crossing_rate(x: np.ndarray) -> float:
    """Fraction of consecutive sample-pairs that cross zero."""
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.diff(np.sign(x)) != 0)) / (len(x) - 1)


def _hjorth(x: np.ndarray) -> tuple[float, float, float]:
    """Hjorth activity, mobility, complexity."""
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = float(np.var(x))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(ddx))
    if var_x < 1e-12:
        return 0.0, 0.0, 0.0
    activity = var_x
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / max(var_dx, 1e-12)) / max(mobility, 1e-12)
    return float(activity), float(mobility), float(complexity)


def _spectral_entropy(psd: np.ndarray) -> float:
    """Shannon entropy of normalised PSD.

    High (~1) for broadband signals (Wake, REM), low for narrowband (N3).
    """
    p = psd / max(psd.sum(), 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) / max(np.log2(len(p)), 1e-12)


def _bandpass_signal(data: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass."""
    nyq = sfreq / 2.0
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 0.9999)
    if lo >= hi:
        return data
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def _detect_spindles(
    epoch: np.ndarray,
    sfreq: float,
    *,
    sigma_band: tuple = (12.0, 16.0),
    min_dur_s: float = 0.5,
    max_dur_s: float = 2.0,
    threshold_sd: float = 1.5,
) -> Dict[str, float]:
    """Detect sleep spindles as sigma-band amplitude bursts.

    AASM: spindles are 11-16 Hz oscillations lasting 0.5-2.0 s, prominent
    in N2.  We use envelope thresholding on the sigma-filtered signal.
    """
    sigma = _bandpass_signal(epoch, sfreq, sigma_band[0], sigma_band[1])
    analytic = hilbert(sigma)
    envelope = np.abs(analytic)

    threshold = float(np.mean(envelope) + threshold_sd * np.std(envelope))
    above = envelope > threshold

    count = 0
    amplitudes = []
    min_samples = int(min_dur_s * sfreq)
    max_samples = int(max_dur_s * sfreq)

    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            dur = i - start
            if min_samples <= dur <= max_samples:
                count += 1
                amplitudes.append(float(np.max(envelope[start:i])))
        else:
            i += 1

    epoch_dur_s = len(epoch) / sfreq
    return {
        "spindle_count": float(count),
        "spindle_density": count / max(epoch_dur_s / 60.0, 1e-6),  # per minute
        "spindle_mean_amplitude": float(np.mean(amplitudes)) if amplitudes else 0.0,
    }


def _detect_kcomplexes(
    epoch: np.ndarray,
    sfreq: float,
    *,
    delta_band: tuple = (0.5, 4.0),
    min_amplitude_uv: float = 75.0,
    min_dur_s: float = 0.5,
    max_dur_s: float = 1.0,
) -> float:
    """Count K-complexes as sharp negative-then-positive delta deflections.

    AASM: K-complex = well-delineated negative sharp wave immediately
    followed by a positive component, standing out from background,
    duration >= 0.5 s.
    """
    delta = _bandpass_signal(epoch, sfreq, delta_band[0], delta_band[1])
    min_samples = int(min_dur_s * sfreq)
    max_samples = int(max_dur_s * sfreq)

    count = 0
    i = 0
    n = len(delta) - 1
    while i < n:
        # Look for negative deflection followed by positive
        if delta[i] < -min_amplitude_uv / 2:
            start = i
            # Find the trough
            while i < n and delta[i] < 0:
                i += 1
            # Now look for positive peak
            peak_start = i
            while i < n and delta[i] > 0:
                i += 1
            dur = i - start
            if min_samples <= dur <= max_samples:
                ptp = float(np.max(delta[peak_start:i]) - np.min(delta[start:peak_start + 1])) if peak_start < i else 0.0
                if ptp >= min_amplitude_uv:
                    count += 1
        else:
            i += 1
    return float(count)


def _delta_amplitude_fraction(
    epoch: np.ndarray,
    sfreq: float,
    threshold_uv: float = 75.0,
) -> float:
    """Fraction of epoch where delta amplitude exceeds threshold.

    AASM N3 criterion: slow-wave activity (0.5-2 Hz) with amplitude >75 uV
    occupies >=20% of the epoch.
    """
    delta = _bandpass_signal(epoch, sfreq, 0.5, 2.0)
    return float(np.mean(np.abs(delta) > threshold_uv))


def epoch_features_eeg(
    epoch: np.ndarray,
    sfreq: float,
    *,
    nperseg: Optional[int] = None,
) -> Dict[str, float]:
    """Compute interpretable EEG features for one epoch.

    Parameters
    ----------
    epoch : 1-D array, (n_samples,)
    sfreq : sampling frequency in Hz.
    nperseg : Welch segment length (default: 4 s).

    Returns
    -------
    Dict of named features.
    """
    if nperseg is None:
        nperseg = min(int(4.0 * sfreq), len(epoch))
    nperseg = max(8, nperseg)

    freqs, psd = welch(epoch, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)

    feats: Dict[str, float] = {}

    total_power = _band_power(psd, freqs, 0.5, 30.0)
    for band_name, (fmin, fmax) in BANDS.items():
        bp = _band_power(psd, freqs, fmin, fmax)
        feats[f"power_{band_name}"] = bp
        feats[f"relpow_{band_name}"] = bp / max(total_power, 1e-12)

    feats["ratio_delta_beta"] = feats["power_delta"] / max(feats["power_beta"], 1e-12)
    feats["ratio_theta_alpha"] = feats["power_theta"] / max(feats["power_alpha"], 1e-12)
    feats["ratio_sigma_total"] = feats["power_sigma"] / max(total_power, 1e-12)

    feats["variance"] = float(np.var(epoch))
    feats["rms"] = float(np.sqrt(np.mean(epoch ** 2)))
    feats["zcr"] = _zero_crossing_rate(epoch)

    act, mob, comp = _hjorth(epoch)
    feats["hjorth_activity"] = act
    feats["hjorth_mobility"] = mob
    feats["hjorth_complexity"] = comp

    # --- Sleep-specific features ---
    feats["spectral_entropy"] = _spectral_entropy(psd)

    feats["delta_fraction_above_75uv"] = _delta_amplitude_fraction(epoch, sfreq, 75.0)

    spindle_feats = _detect_spindles(epoch, sfreq)
    feats.update(spindle_feats)

    feats["kcomplex_count"] = _detect_kcomplexes(epoch, sfreq)

    return feats


def epoch_features_eog(epoch: np.ndarray, sfreq: float) -> Dict[str, float]:
    """EOG features for REM detection and slow eye movement indexing."""
    feats: Dict[str, float] = {}
    feats["eog_variance"] = float(np.var(epoch))
    feats["eog_rms"] = float(np.sqrt(np.mean(epoch ** 2)))
    feats["eog_zcr"] = _zero_crossing_rate(epoch)
    dx = np.diff(epoch)
    feats["eog_rem_activity"] = float(np.var(dx)) if len(dx) else 0.0

    # REM burst detection: count rapid eye movements as derivative peaks
    rem_count, rem_density = _detect_rem_bursts(epoch, sfreq)
    feats["eog_rem_count"] = rem_count
    feats["eog_rem_density"] = rem_density

    # Slow eye movement index (N1 marker): low-frequency (<0.5 Hz) amplitude
    feats["eog_sem_index"] = _slow_eye_movement_index(epoch, sfreq)

    return feats


def _detect_rem_bursts(
    epoch: np.ndarray,
    sfreq: float,
    *,
    derivative_threshold_factor: float = 3.0,
    min_burst_interval_s: float = 0.3,
) -> tuple:
    """Detect rapid conjugate eye movements as derivative peaks.

    Returns (count, density_per_minute).
    """
    dx = np.diff(epoch) * sfreq  # derivative in uV/s
    abs_dx = np.abs(dx)
    threshold = float(np.mean(abs_dx) + derivative_threshold_factor * np.std(abs_dx))
    if threshold < 1e-6:
        return 0.0, 0.0

    above = abs_dx > threshold
    min_interval = int(min_burst_interval_s * sfreq)
    count = 0
    last_peak = -min_interval
    for i in range(len(above)):
        if above[i] and (i - last_peak) >= min_interval:
            count += 1
            last_peak = i

    epoch_dur_min = len(epoch) / sfreq / 60.0
    return float(count), count / max(epoch_dur_min, 1e-6)


def _slow_eye_movement_index(epoch: np.ndarray, sfreq: float) -> float:
    """Slow eye movement index: RMS of low-frequency (<0.5 Hz) EOG component.

    Elevated during N1 (drowsiness), low during REM and Wake.
    """
    nyq = sfreq / 2.0
    cutoff = 0.5 / nyq
    if cutoff >= 0.999 or cutoff <= 0:
        return 0.0
    sos = butter(3, cutoff, btype="low", output="sos")
    low_freq = sosfiltfilt(sos, epoch)
    return float(np.sqrt(np.mean(low_freq ** 2)))


def epoch_features_emg(epoch: np.ndarray, sfreq: float) -> Dict[str, float]:
    """EMG features: tonic level, phasic bursts, and basic statistics."""
    feats: Dict[str, float] = {}
    rectified = np.abs(epoch)
    feats["emg_rms"] = float(np.sqrt(np.mean(epoch ** 2)))
    feats["emg_mean_rect"] = float(np.mean(rectified))
    feats["emg_variance"] = float(np.var(epoch))

    # Tonic EMG level: 10th percentile of RMS envelope (sustained muscle tone)
    feats["emg_tonic_level"] = _emg_tonic_level(epoch, sfreq)

    # Phasic EMG bursts: transient activations (RBD marker)
    feats["emg_phasic_count"] = _emg_phasic_bursts(epoch, sfreq)

    return feats


def _emg_tonic_level(epoch: np.ndarray, sfreq: float) -> float:
    """Estimate tonic (sustained) EMG level as 10th percentile of RMS envelope.

    Low in REM (atonia), high in Wake.
    """
    win = max(int(0.25 * sfreq), 1)  # 250 ms window
    sq = epoch ** 2
    kernel = np.ones(win) / win
    rms_env = np.sqrt(np.convolve(sq, kernel, mode="same"))
    return float(np.percentile(rms_env, 10))


def _emg_phasic_bursts(
    epoch: np.ndarray,
    sfreq: float,
    *,
    threshold_factor: float = 3.0,
    min_dur_s: float = 0.1,
    max_dur_s: float = 0.5,
) -> float:
    """Count phasic EMG bursts (>3x baseline, 0.1-0.5 s duration).

    Elevated in REM without atonia (REM behaviour disorder marker).
    """
    rectified = np.abs(epoch)
    baseline = float(np.median(rectified))
    if baseline < 1e-6:
        return 0.0

    threshold = baseline * threshold_factor
    above = rectified > threshold
    min_samples = int(min_dur_s * sfreq)
    max_samples = int(max_dur_s * sfreq)

    count = 0
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            if min_samples <= (i - start) <= max_samples:
                count += 1
        else:
            i += 1
    return float(count)


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_all_features(epoched: EpochedSignals) -> List[Dict[str, float]]:
    """Extract features for every epoch.

    Returns a list of dicts (one per epoch).  Features from unavailable
    modalities are filled with 0.0.
    """
    all_feats: List[Dict[str, float]] = []
    for i in range(epoched.n_epochs):
        f: Dict[str, float] = {}
        f.update(epoch_features_eeg(epoched.eeg_epochs[i], epoched.sfreq))
        if epoched.eog_epochs is not None:
            f.update(epoch_features_eog(epoched.eog_epochs[i], epoched.sfreq))
        else:
            f.update({k: 0.0 for k in [
                "eog_variance", "eog_rms", "eog_zcr", "eog_rem_activity",
                "eog_rem_count", "eog_rem_density", "eog_sem_index",
            ]})
        if epoched.emg_epochs is not None:
            f.update(epoch_features_emg(epoched.emg_epochs[i], epoched.sfreq))
        else:
            f.update({k: 0.0 for k in [
                "emg_rms", "emg_mean_rect", "emg_variance",
                "emg_tonic_level", "emg_phasic_count",
            ]})
        f["is_artifact"] = float(epoched.rejected_mask[i]) if epoched.rejected_mask is not None else 0.0
        all_feats.append(f)
    return all_feats
