"""Signal sanity check for the EEG channel fed to a staging model.

When a model never outputs N3, the cause is either that the slow waves are not
in the signal (wrong scaling, wrong derivation, aggressive high-pass) or that
the model failed to see them. Measuring the AASM N3 criterion directly on the
channel actually used settles the question instead of leaving it to guesswork.

AASM scores N3 when slow-wave activity (0.5-2 Hz, peak-to-peak > 75 uV)
occupies at least 20 % of the epoch, so that is what is measured here.

Pure numpy / scipy: no MNE or Qt import.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: AASM peak-to-peak threshold, in microvolts, defining a slow wave.
SLOW_WAVE_UV = 75.0

#: Plausible range for the peak-to-peak amplitude of a sleep EEG derivation,
#: in microvolts. Outside of it, a unit-scaling error is the first suspicion.
PLAUSIBLE_P2P_UV = (20.0, 1000.0)


def _bandpass(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyquist = sfreq / 2.0
    high = min(high, nyquist * 0.99)
    if low >= high:
        raise ValueError(f"Bande invalide: {low}-{high} Hz a {sfreq} Hz")
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return np.asarray(sosfiltfilt(sos, data), dtype=float)


def _slow_wave_coverage(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float = 30.0,
    unit_scale: float = 1e6,
) -> Optional[np.ndarray]:
    """Per-epoch share of 1 s blocks with slow waves above 75 uV peak-to-peak.

    Returns None when the signal is too short to measure.
    """
    signal = np.asarray(data, dtype=float).ravel() * float(unit_scale)
    signal = signal[np.isfinite(signal)]
    n_per_epoch = int(round(float(sfreq) * float(epoch_length)))
    if n_per_epoch <= 0 or signal.size < n_per_epoch:
        return None
    n_epochs = signal.size // n_per_epoch
    try:
        slow = _bandpass(signal, float(sfreq), 0.5, 2.0)
    except Exception as exc:  # pragma: no cover - depends on sfreq
        logger.debug("Slow-wave filtering failed: %s", exc)
        return None
    slow_epochs = slow[: n_epochs * n_per_epoch].reshape(n_epochs, n_per_epoch)
    per_second = max(int(round(float(sfreq))), 1)
    n_seconds = n_per_epoch // per_second
    if n_seconds <= 0:
        return None
    blocks = slow_epochs[:, : n_seconds * per_second].reshape(
        n_epochs, n_seconds, per_second
    )
    block_p2p = blocks.max(axis=2) - blocks.min(axis=2)
    return (block_p2p >= SLOW_WAVE_UV).mean(axis=1)


def n3_epoch_mask(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float = 30.0,
    unit_scale: float = 1e6,
    coverage_min: float = 0.20,
) -> Optional[np.ndarray]:
    """Boolean mask of epochs that meet the AASM N3 slow-wave criterion."""
    coverage = _slow_wave_coverage(data, sfreq, epoch_length, unit_scale)
    if coverage is None:
        return None
    return coverage >= float(coverage_min)


def promote_n3(
    stages: Sequence[str],
    mask: Sequence[bool],
    *,
    only_from: Sequence[str] = ("N2",),
) -> tuple[list, int]:
    """Rewrite *only_from* stages to N3 where *mask* is true.

    Wake and REM are left alone: slow waves can coexist with them, and the AASM
    N3 rule is applied here only to the N2/N3 boundary that models under-call.
    Returns the rewritten list and how many epochs were promoted.
    """
    allowed = {str(s).upper() for s in only_from}
    out = [str(s) for s in stages]
    n = min(len(out), len(mask))
    promoted = 0
    for i in range(n):
        if mask[i] and out[i].upper() in allowed:
            out[i] = "N3"
            promoted += 1
    return out, promoted


def eeg_signal_stats(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float = 30.0,
    unit_scale: float = 1e6,
) -> Optional[Dict[str, Any]]:
    """Amplitude statistics of one EEG channel, in microvolts.

    ``data`` is expected in volts, as MNE stores it; ``unit_scale`` converts to
    microvolts. Returns None when the signal is too short to measure.
    """
    signal = np.asarray(data, dtype=float).ravel() * float(unit_scale)
    signal = signal[np.isfinite(signal)]
    n_per_epoch = int(round(float(sfreq) * float(epoch_length)))
    if n_per_epoch <= 0 or signal.size < n_per_epoch:
        return None

    n_epochs = signal.size // n_per_epoch
    epochs = signal[: n_epochs * n_per_epoch].reshape(n_epochs, n_per_epoch)

    # Robust percentile range rather than raw min-max, immune to spikes.
    p2p = np.percentile(epochs, 99, axis=1) - np.percentile(epochs, 1, axis=1)

    stats: Dict[str, Any] = {
        "n_epoques_analysees": int(n_epochs),
        "amplitude_p2p_uv_mediane": round(float(np.median(p2p)), 1),
        "amplitude_p2p_uv_p95": round(float(np.percentile(p2p, 95)), 1),
    }

    try:
        slow = _bandpass(signal, float(sfreq), 0.5, 2.0)
    except Exception as exc:  # pragma: no cover - depends on sfreq
        logger.debug("Slow-wave filtering failed: %s", exc)
        return stats

    n_per_epoch = int(round(float(sfreq) * float(epoch_length)))
    n_epochs = signal.size // n_per_epoch
    slow_epochs = slow[: n_epochs * n_per_epoch].reshape(n_epochs, n_per_epoch)
    per_second = max(int(round(float(sfreq))), 1)
    n_seconds = n_per_epoch // per_second
    if n_seconds:
        blocks = slow_epochs[:, : n_seconds * per_second].reshape(
            n_epochs, n_seconds, per_second
        )
        block_p2p = blocks.max(axis=2) - blocks.min(axis=2)
        coverage = (block_p2p >= SLOW_WAVE_UV).mean(axis=1)
        stats["onde_lente_p2p_uv_mediane"] = round(
            float(np.median(block_p2p)), 1
        )
        stats["onde_lente_p2p_uv_p95"] = round(
            float(np.percentile(block_p2p, 95)), 1
        )
        stats["pct_epoques_criteres_n3"] = round(
            float((coverage >= 0.20).mean() * 100.0), 1
        )
    return stats


def _fmt_uv(value: float) -> str:
    """Amplitudes span several decades once a scaling error is involved."""
    return f"{value:.3f}" if abs(value) < 1.0 else f"{value:.0f}"


def amplitude_diagnosis(stats: Optional[Dict[str, Any]]) -> Optional[str]:
    """Verdict on the amplitude of the EEG channel, when it looks wrong."""
    if not stats:
        return None
    p2p = stats.get("amplitude_p2p_uv_mediane")
    if p2p is None:
        return None
    low, high = PLAUSIBLE_P2P_UV
    if p2p < low:
        factor = " (facteur 1000 : donnees en volts lues comme des microvolts ?)" \
            if p2p < low / 100 else ""
        return (
            f"L'amplitude crete a crete du canal EEG est de {_fmt_uv(p2p)} uV, sous "
            f"le minimum plausible de {low:.0f} uV pour une derivation de "
            f"sommeil{factor}. Un signal aussi faible ne peut pas franchir le "
            f"seuil de {SLOW_WAVE_UV:.0f} uV qui definit une onde lente : le N3 "
            "est mecaniquement indetectable. Verifiez l'unite du signal et la "
            "calibration avant de mettre en cause le modele."
        )
    if p2p > high:
        return (
            f"L'amplitude crete a crete du canal EEG est de {_fmt_uv(p2p)} uV, au-dela "
            f"du maximum plausible de {high:.0f} uV. Un facteur d'echelle errone "
            "ou une electrode defectueuse fausse toutes les mesures d'amplitude "
            "dont depend la detection du N3."
        )
    return None


def n3_capability_note(
    stats: Optional[Dict[str, Any]],
    predicted_n3_pct: Optional[float] = None,
    manual_n3_pct: Optional[float] = None,
) -> Optional[str]:
    """Whether the signal contains the slow waves the model failed to report.

    Separates a signal that has no slow waves at all from a model that missed
    slow waves which are plainly present.
    """
    if not stats or "pct_epoques_criteres_n3" not in stats:
        return None
    available = float(stats["pct_epoques_criteres_n3"])
    slow_p2p = stats.get("onde_lente_p2p_uv_p95")
    predicted = predicted_n3_pct or 0.0

    if available < 1.0:
        detail = (
            f" L'amplitude des ondes lentes plafonne a {_fmt_uv(slow_p2p)} uV "
            f"(seuil {SLOW_WAVE_UV:.0f} uV)." if slow_p2p is not None else ""
        )
        expected = (
            f" Le scoreur humain a pourtant annote {manual_n3_pct:.1f}% de N3, "
            "ce qui confirme une incoherence entre le signal analyse et la voie "
            "sur laquelle la lecture manuelle a ete faite."
            if manual_n3_pct else ""
        )
        return (
            "Mesure sur le canal EEG utilise : aucune epoque ne remplit le "
            "critere AASM du N3 (ondes lentes > "
            f"{SLOW_WAVE_UV:.0f} uV sur au moins 20% de l'epoque)."
            f"{detail} L'absence de N3 vient donc du signal analyse, pas du "
            f"modele : verifiez l'echelle, la derivation et le filtre passe-haut."
            f"{expected}"
        )
    if available >= 5.0 and predicted < available / 3.0:
        return (
            f"Mesure sur le canal EEG utilise : {available:.1f}% des epoques "
            f"remplissent le critere AASM du N3, alors que le modele n'en a "
            f"predit que {predicted:.1f}%. Les ondes lentes sont bien presentes "
            "dans le signal : la limite vient du modele, et une correction "
            "manuelle de la frontiere N2/N3 est justifiee."
        )
    return None
