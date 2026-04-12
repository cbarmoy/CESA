"""
CESA v4.0 - Signal Filter Utilities
==================================

Backward-compatible thin wrapper around ``CESA.filter_engine``.

The public API (``apply_filter``, ``apply_baseline_correction``,
``detect_signal_type``, ``get_filter_presets``) is unchanged.  Internally
``apply_filter`` now delegates to the composable filter-engine classes so
that the same IIR implementations are shared with the new pipeline-based
filter dialog.

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 0.0beta1.0
"""

from typing import Optional
import numpy as np


def apply_filter(
    data: np.ndarray,
    *,
    sfreq: float,
    filter_order: int,
    low: Optional[float] = None,
    high: Optional[float] = None,
    filter_type: str = "butterworth",
) -> np.ndarray:
    """Apply an IIR filter to a 1D signal array.

    Behavior by convention:
    - low > 0 and high > 0  -> band-pass [low, high]
    - low > 0 and (high == 0 or None) -> high-pass at low
    - (low == 0 or None) and high > 0 -> low-pass at high
    - otherwise -> return data unchanged

    Parameters
    ----------
    data : np.ndarray
        The input 1D signal in microvolts.
    sfreq : float
        Sampling frequency in Hz.
    filter_order : int
        Filter order.
    low : float | None
        Low cutoff in Hz (0 for disabled).
    high : float | None
        High cutoff in Hz (0 for disabled).
    filter_type : str
        IIR design method: ``"butterworth"`` (default), ``"cheby1"``,
        ``"cheby2"``, or ``"ellip"``.

    Returns
    -------
    np.ndarray
        Filtered signal. If parameters are invalid, returns the input unchanged.
    """
    try:
        from CESA.filter_engine import pipeline_from_legacy_params

        lo = 0.0 if low is None else float(low)
        hi = 0.0 if high is None else float(high)

        nyquist = float(sfreq) / 2.0
        if nyquist <= 0.0:
            return data
        if lo > 0 and hi > 0 and hi >= nyquist:
            hi = nyquist - 1e-6

        pipeline = pipeline_from_legacy_params(
            low=lo, high=hi, order=int(filter_order),
            filter_type=filter_type, enabled=True,
        )
        if not pipeline.filters:
            return data
        return pipeline.apply(data, sfreq)
    except Exception:
        return data


def apply_baseline_correction(
    data: np.ndarray,
    *,
    window_duration: float = 30.0,
    sfreq: float = 200.0,
    method: str = "mean",
    kernel: str = "hann",
) -> np.ndarray:
    """Apply baseline correction (vectorized) to remove DC and slow drifts.

    Optimized for performance: avoids per-sample loops using vectorized
    convolution-based detrending. Keeps visual behavior equivalent for EEG.

    Parameters
    ----------
    data : np.ndarray
        Input 1D signal in microvolts (µV).
    window_duration : float
        Window length in seconds for local baseline estimation.
    sfreq : float
        Sampling frequency (Hz).
    method : {"mean", "median"}
        Baseline estimator across the window. "mean" uses a smoothed mean;
        "median" uses a fast approximated median (via medfilt if available).
    kernel : {"hann", "box"}
        Smoothing kernel to approximate local mean when method="mean".

    Returns
    -------
    np.ndarray
        Baseline-corrected signal (data - local_baseline).
    """
    try:
        x = np.asarray(data, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        n = x.size
        w = max(1, int(round(float(window_duration) * float(sfreq))))
        if w <= 1 or n == 0:
            # Fallback to global baseline
            baseline = np.median(x) if method == "median" else float(np.mean(x))
            return x - baseline

        # Vectorized local baseline
        if method == "median":
            # Fast median filter if SciPy available
            try:
                from scipy.signal import medfilt
                # Ensure odd kernel size for medfilt
                k = w if (w % 2 == 1) else (w + 1)
                baseline = medfilt(x, kernel_size=k)
            except Exception:
                # Approximate with mean smoothing if medfilt not available
                method = "mean"

        if method == "mean":
            # Build smoothing kernel (Hann or boxcar)
            if kernel == "hann":
                k = np.hanning(w)
                if not np.any(k):
                    k = np.ones(w)
            else:
                k = np.ones(w)
            k = k / np.sum(k)
            # Convolve with reflect padding for edge behavior akin to sliding window
            baseline = np.convolve(np.pad(x, (w//2, w-1-w//2), mode='reflect'), k, mode='valid')

        return x - baseline
    except Exception:
        return data


def detect_signal_type(channel_name: str) -> str:
    """Détecte automatiquement le type de signal selon le nom du canal.

    Cette fonction utilise des patterns de reconnaissance pour identifier
    les différents types de signaux physiologiques dans les enregistrements polysomnographiques.

    Parameters
    ----------
    channel_name : str
        Nom du canal à analyser (insensible à la casse)

    Returns
    -------
    str
        Type de signal détecté :
        - 'eeg' : Électroencéphalogramme
        - 'ecg' : Électrocardiogramme
        - 'emg' : Électromyogramme
        - 'eog' : Électrooculogramme
        - 'sas_eeg' : EEG pour l'analyse SAS (Sleep Apnea Syndrome)
        - 'sas_emg' : EMG pour l'analyse SAS
        - 'unknown' : Type non reconnu
    """
    channel_lower = channel_name.lower().strip()

    # Patterns pour EEG
    eeg_patterns = [
        'f3', 'f4', 'c3', 'c4', 'o1', 'o2',  # Positions standard
        'fp1', 'fp2', 'f7', 'f8', 't3', 't4', 't5', 't6', 'p3', 'p4',  # Autres positions
        'cz', 'pz', 'fz',  # Positions centrales
        'eeg',  # Nom générique
    ]

    # Patterns pour ECG
    ecg_patterns = [
        'ecg', 'ecg1', 'ecg2', 'ekg',  # Variantes courantes
        'heart rate', 'heartrate',  # Heart Rate (anglais)
        'fréquence cardi', 'frequence cardi', 'freq cardi',  # Heart Rate (français)
        'cardiac', 'cardiaque',  # Variantes cardiaques
    ]

    # Patterns pour EMG
    emg_patterns = [
        'emg', 'chin', 'menton',  # EMG du menton
        'emg chin', 'emg_menton',
        'submental', 'submentalis',  # EMG sous-mental
        'left leg', 'right leg',  # EMG des jambes (Left Leg, Right Leg)
        'leg',  # Pattern général pour les jambes
    ]

    # Patterns pour EOG
    eog_patterns = [
        'e1', 'e2',  # Électrodes standard EOG
        'eog', 'eog left', 'eog right', 'eog gauche', 'eog droite',
        'loc', 'roc',  # Left/Right Outer Canthus
    ]

    # Vérifier chaque type de signal
    for pattern in eeg_patterns:
        if pattern in channel_lower:
            if 'sas' in channel_lower:
                return 'sas_eeg'
            return 'eeg'

    for pattern in ecg_patterns:
        if pattern in channel_lower:
            return 'ecg'

    for pattern in emg_patterns:
        if pattern in channel_lower:
            if 'sas' in channel_lower:
                return 'sas_emg'
            return 'emg'

    for pattern in eog_patterns:
        if pattern in channel_lower:
            return 'eog'

    return 'unknown'


def get_filter_presets(signal_type: str) -> dict:
    """Retourne les paramètres de filtrage optimaux selon le type de signal.

    Parameters
    ----------
    signal_type : str
        Type de signal ('eeg', 'ecg', 'emg', 'eog', 'sas_eeg', 'sas_emg')

    Returns
    -------
    dict
        Dictionnaire contenant les paramètres :
        - low : fréquence de coupure basse (Hz)
        - high : fréquence de coupure haute (Hz)
        - amplitude : échelle d'amplitude recommandée (%)
        - enabled : activation par défaut
    """
    presets = {
        'eeg': {
            'low': 0.3,
            'high': 70.0,
            'amplitude': 100.0,
            'enabled': True,
            'description': 'EEG standard (0.3-70 Hz)'
        },
        'ecg': {
            'low': 0.3,
            'high': 70.0,
            'amplitude': 100.0,
            'enabled': True,
            'description': 'ECG standard (0.3-70 Hz)'
        },
        'emg': {
            'low': 10.0,
            'high': 0.0,  # Passe-haut seulement
            'amplitude': 25.0,  # Amplitude réduite pour éviter la saturation
            'enabled': True,
            'description': 'EMG passe-haut 10 Hz (réduit)'
        },
        'eog': {
            'low': 0.3,
            'high': 35.0,
            'amplitude': 50.0,  # Amplitude réduite
            'enabled': True,
            'description': 'EOG standard (0.3-35 Hz, réduit)'
        },
        'sas_eeg': {
            'low': 0.5,
            'high': 35.0,
            'amplitude': 100.0,
            'enabled': True,
            'description': 'EEG SAS (0.5-35 Hz)'
        },
        'sas_emg': {
            'low': 25.0,
            'high': 0.0,  # Passe-haut seulement pour SAS
            'amplitude': 25.0,
            'enabled': True,
            'description': 'EMG SAS passe-haut 25 Hz (réduit)'
        },
        'unknown': {
            'low': 0.3,
            'high': 35.0,
            'amplitude': 100.0,
            'enabled': True,
            'description': 'Paramètres par défaut'
        }
    }

    return presets.get(signal_type, presets['unknown'])


