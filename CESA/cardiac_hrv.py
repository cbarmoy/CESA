"""
Cardiac HRV utilities for CESA.

Calcule RMSSD et puissances LF/HF sur des époques d'éveil (30 s par défaut),
à partir d'un canal ECG/PPG extrait des fichiers EDF via MNE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging

import numpy as np
from scipy.signal import find_peaks, welch

try:  # pragma: no cover - optionnel pour les tests
    import mne
except Exception:  # pragma: no cover
    mne = None  # type: ignore


LOGGER = logging.getLogger(__name__)

_STAGE_ALIAS = {
    "REM": "R",
    "R": "R",
    "NREM": "NREM",
    "SWS": "N3",
    "N3": "N3",
    "N4": "N3",
    "N2": "N2",
    "N1": "N1",
    "W": "W",
    "WAKE": "W",
    "ÉVEIL": "W",
    "EVEIL": "W",
}


def _normalize_stage_label(label: str) -> str:
    key = str(label or "").strip().upper()
    return _STAGE_ALIAS.get(key, key)


@dataclass
class HRVConfig:
    """Paramètres du calcul HRV."""

    lf_band: Tuple[float, float] = (0.04, 0.15)
    hf_band: Tuple[float, float] = (0.15, 0.4)
    resample_fs: float = 4.0
    filter_band: Tuple[float, float] = (1.0, 40.0)
    peak_detection_method: str = "neurokit2"  # simple | neurokit2
    min_peak_distance_s: float = 0.3
    peak_prominence_std: float = 0.6
    min_rr: int = 4
    spectral_min_rr: int = 60
    spectral_min_duration_s: float = 120.0
    # Segmentation / nettoyage
    stage_filter: Tuple[str, ...] = ("REM", "R")
    min_segment_s: float = 300.0  # 5 minutes par défaut
    allow_short_segments: bool = False
    clean_rr: bool = True
    rr_cleaning_method: str = "kubios"  # simple | neurokit2 | kubios
    rr_min_s: float = 0.3
    rr_max_s: float = 2.5
    rr_successive_thresh: float = 0.3  # rejeter si |diff|/prev > 30 %


class HRVComputationError(RuntimeError):
    """Erreur spécifique au calcul HRV."""


def _pick_ecg_channel(raw, preferred: Optional[Sequence[str]]) -> Optional[str]:
    """Sélectionne un canal ECG/PPG parmi les préférences ou heuristiques."""

    def _is_cardio(name: str) -> bool:
        cardio_tags = (
            "ecg",
            "heart",
            "cardi",
            "pleth",
            "ppg",
            "pulse",
        )
        lname = name.lower()
        return any(tag in lname for tag in cardio_tags)

    # 1) Préférences utilisateur mais uniquement si cela ressemble à de l'ECG/PPG
    if preferred:
        for name in preferred:
            if name in raw.ch_names and _is_cardio(name):
                LOGGER.debug("[HRV] Canal préféré sélectionné: %s", name)
                return name
        # Si aucun préféré ne ressemble à du cardio, on log et on passe aux heuristiques
        LOGGER.debug("[HRV] Aucun canal cardio dans la sélection préférée, on tente la détection auto.")

    # 2) Heuristique automatique
    candidates = (
        "ECG",
        "ECG1",
        "ECG2",
        "HEART RATE",
        "FREQUENCE CARDI",
        "FRÉQUENCE CARDI",
        "PLETH",
        "PPG",
    )
    for key in candidates:
        for ch in raw.ch_names:
            if key.lower() == ch.lower():
                LOGGER.debug("[HRV] Canal détecté (match exact): %s", ch)
                return ch
            if key.lower() in ch.lower():
                LOGGER.debug("[HRV] Canal détecté (match partiel): %s", ch)
                return ch
    return None


def _bandpass(signal: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    low, high = band
    if mne is None:
        raise HRVComputationError("La dépendance mne est requise pour filtrer le canal ECG/PPG.")
    return mne.filter.filter_data(signal, sfreq=sfreq, l_freq=low, h_freq=high, verbose="ERROR")


def _log_peak_rate(peaks: np.ndarray, signal: np.ndarray, sfreq: float, method: str) -> None:
    duration_s = float(len(signal) / sfreq) if sfreq > 0 else 0.0
    if duration_s <= 0:
        return
    bpm = float(len(peaks) * 60.0 / duration_s)
    if bpm < 20.0 or bpm > 200.0:
        LOGGER.warning(
            "[HRV] Détection de pics suspecte (%s): %.1f bpm, %d pics sur %.1fs",
            method,
            bpm,
            len(peaks),
            duration_s,
        )


def _detect_r_peaks_simple(signal: np.ndarray, sfreq: float, cfg: HRVConfig) -> np.ndarray:
    """Détection simple des R-peaks via filtrage puis find_peaks."""
    filtered = _bandpass(signal, sfreq, cfg.filter_band)
    z = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    min_distance = max(1, int(cfg.min_peak_distance_s * sfreq))
    prominence = cfg.peak_prominence_std
    peaks, _ = find_peaks(z, distance=min_distance, prominence=prominence)
    _log_peak_rate(peaks, signal, sfreq, "simple")
    LOGGER.debug("[HRV] R-peaks détectés (simple): %d (fs=%.1f, dist>=%.3fs)", len(peaks), sfreq, cfg.min_peak_distance_s)
    return peaks


def _detect_r_peaks_neurokit(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Détection robuste via NeuroKit2."""
    try:
        import neurokit2 as nk  # type: ignore
    except Exception as exc:  # pragma: no cover - dépendance gérée à l'installation
        raise HRVComputationError("NeuroKit2 est requis pour la détection ECG 'neurokit2'.") from exc

    cleaned = nk.ecg_clean(signal, sampling_rate=sfreq, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sfreq, method="neurokit", correct_artifacts=False)
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int).reshape(-1)
    _log_peak_rate(peaks, signal, sfreq, "neurokit2")
    LOGGER.debug("[HRV] R-peaks détectés (neurokit2): %d", len(peaks))
    return peaks


def _detect_r_peaks(signal: np.ndarray, sfreq: float, cfg: HRVConfig) -> np.ndarray:
    method = str(getattr(cfg, "peak_detection_method", "simple") or "simple").strip().lower()
    if method == "neurokit2":
        try:
            return _detect_r_peaks_neurokit(signal, sfreq)
        except Exception as exc:
            LOGGER.warning("[HRV] Détection neurokit2 indisponible (%s). Repli sur la méthode simple.", exc)
    return _detect_r_peaks_simple(signal, sfreq, cfg)


def _rr_intervals(peaks: np.ndarray, sfreq: float) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([], dtype=float)
    return np.diff(peaks) / float(sfreq)


def _compute_rmssd(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return np.nan
    diff = np.diff(rr)
    return float(np.sqrt(np.mean(np.square(diff))))


def _clean_rr_intervals_simple(
    rr: np.ndarray,
    cfg: HRVConfig,
) -> Tuple[np.ndarray, Optional[float], int]:
    """Nettoyage RR rule-based (seuils + médiane locale).

    Returns:
        (rr_cleaned, reject_pct, correction_events)
    """
    if not cfg.clean_rr:
        return rr, 0.0, 0
    if rr.size == 0:
        return rr, np.nan, 0

    mask = (rr >= cfg.rr_min_s) & (rr <= cfg.rr_max_s)
    cleaned = rr[mask]
    if cleaned.size < 3:
        n_rejected = int(rr.size - cleaned.size)
        reject_pct = float(100.0 * n_rejected / rr.size) if rr.size > 0 else np.nan
        return cleaned, reject_pct, n_rejected

    keep = np.ones(cleaned.size, dtype=bool)
    radius = 2
    for idx, value in enumerate(cleaned):
        start = max(0, idx - radius)
        stop = min(cleaned.size, idx + radius + 1)
        neighborhood = cleaned[start:stop]
        if neighborhood.size <= 1:
            continue
        local = np.delete(neighborhood, min(idx - start, neighborhood.size - 1))
        if local.size == 0:
            continue
        local_median = float(np.median(local))
        if local_median <= 0:
            continue
        keep[idx] = abs(value - local_median) / local_median <= cfg.rr_successive_thresh
    cleaned2 = cleaned[keep]
    n_rejected = int(rr.size - cleaned2.size)
    reject_pct = float(100.0 * n_rejected / rr.size) if rr.size > 0 else np.nan
    return cleaned2, reject_pct, n_rejected


def _extract_peaks_from_neurokit_output(output, original_len: int) -> Optional[np.ndarray]:
    """Best-effort extraction of corrected peaks from NeuroKit2 outputs."""
    if output is None:
        return None

    def _to_peak_array(candidate) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(candidate, dtype=float).reshape(-1)
        except Exception:
            return None
        if arr.size < 2:
            return None
        if np.isnan(arr).any():
            return None
        return arr.astype(int)

    if isinstance(output, dict):
        for key in ("Peaks", "peaks", "ECG_R_Peaks", "R_Peaks"):
            if key in output:
                arr = _to_peak_array(output[key])
                if arr is not None:
                    return arr
        return None

    if isinstance(output, tuple):
        candidates: List[np.ndarray] = []
        for item in output:
            if isinstance(item, dict):
                arr = _extract_peaks_from_neurokit_output(item, original_len)
                if arr is not None:
                    candidates.append(arr)
                continue
            arr = _to_peak_array(item)
            if arr is not None:
                candidates.append(arr)
        if not candidates:
            return None
        # Prefer the candidate closest in length to original peaks.
        candidates.sort(key=lambda a: abs(a.size - original_len))
        return candidates[0]

    return _to_peak_array(output)


def _estimate_reject_and_events_from_nk_info(
    info: Dict[str, Any],
    n_rr_raw: int,
) -> Tuple[Optional[float], int]:
    """Estimate reject % and number of correction events from NeuroKit2 info."""
    if n_rr_raw <= 0:
        return None, 0
    if not isinstance(info, dict):
        return None, 0
    keys = ("ectopic", "missed", "extra", "longshort")
    impacted = 0
    for key in keys:
        value = info.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            impacted += int(len(value))
    if impacted <= 0:
        return 0.0, 0
    reject_pct = float(100.0 * impacted / n_rr_raw)
    return reject_pct, impacted


def _clean_rr_intervals_neurokit(
    peaks: np.ndarray,
    sfreq: float,
    cfg: HRVConfig,
    method: str,
) -> Tuple[Optional[np.ndarray], Optional[float], int]:
    """Clean RR intervals using NeuroKit2 signal_fixpeaks.

    Returns:
        (rr_intervals, reject_pct_estimate, correction_events)
    """
    try:
        import neurokit2 as nk  # type: ignore
    except Exception:
        return None, None, 0

    fixed = None
    method_candidates = []
    if method == "kubios":
        method_candidates = ["Kubios", "kubios"]
    else:
        method_candidates = ["NeuroKit", "neurokit"]

    for method_name in method_candidates:
        try:
            fixed = nk.signal_fixpeaks(peaks, sampling_rate=sfreq, method=method_name)
            break
        except Exception:
            continue

    info: Optional[Dict[str, Any]] = None
    if isinstance(fixed, tuple):
        for item in fixed:
            if isinstance(item, dict):
                info = item
                break
    elif isinstance(fixed, dict):
        info = fixed

    corrected_peaks = _extract_peaks_from_neurokit_output(fixed, original_len=len(peaks))
    if corrected_peaks is None or corrected_peaks.size < 2:
        return None, None, 0
    rr = _rr_intervals(corrected_peaks, sfreq)
    bounds_mask = (rr >= cfg.rr_min_s) & (rr <= cfg.rr_max_s)
    bounds_rejected = int(rr.size - int(np.sum(bounds_mask)))
    rr = rr[bounds_mask]
    reject_pct, impacted = _estimate_reject_and_events_from_nk_info(
        info or {},
        n_rr_raw=max(0, len(peaks) - 1),
    )
    total_events = int(impacted + bounds_rejected)
    n_rr_raw = max(0, len(peaks) - 1)
    total_events = min(total_events, n_rr_raw)
    reject_pct_total = float(100.0 * total_events / n_rr_raw) if n_rr_raw > 0 else np.nan
    return rr, reject_pct_total if reject_pct is not None else np.nan, total_events


def _clean_rr_intervals(
    peaks: np.ndarray,
    sfreq: float,
    cfg: HRVConfig,
) -> Tuple[np.ndarray, Optional[float], int]:
    rr_raw = _rr_intervals(peaks, sfreq)
    if not cfg.clean_rr:
        return rr_raw, 0.0, 0
    if rr_raw.size == 0:
        return rr_raw, np.nan, 0

    method = str(getattr(cfg, "rr_cleaning_method", "simple") or "simple").strip().lower()
    if method == "simple":
        rr_c, reject_pct, events = _clean_rr_intervals_simple(rr_raw, cfg)
        return rr_c, reject_pct, events

    if method in {"neurokit2", "kubios"}:
        rr_nk, reject_pct_nk, events_nk = _clean_rr_intervals_neurokit(peaks, sfreq, cfg, method=method)
        if rr_nk is not None:
            return rr_nk, reject_pct_nk, events_nk
        LOGGER.warning("[HRV] NeuroKit2 cleaning unavailable (%s). Falling back to simple cleaning.", method)
        rr_c, reject_pct, events = _clean_rr_intervals_simple(rr_raw, cfg)
        return rr_c, reject_pct, events

    LOGGER.warning("[HRV] Unknown rr_cleaning_method=%r. Using simple cleaning.", method)
    rr_c, reject_pct, events = _clean_rr_intervals_simple(rr_raw, cfg)
    return rr_c, reject_pct, events


def _frequency_domain_ready(rr: np.ndarray, cfg: HRVConfig) -> Tuple[bool, str]:
    if len(rr) < max(cfg.min_rr, cfg.spectral_min_rr):
        return False, "freq_domain_insufficient_rr"
    rr_duration = float(np.sum(rr))
    if rr_duration < cfg.spectral_min_duration_s:
        return False, "freq_domain_short_duration"
    return True, "ok"


def _compute_frequency_powers(rr: np.ndarray, cfg: HRVConfig) -> Dict[str, float]:
    """Interpolation du tachogramme puis Welch pour LF/HF."""

    freq_ok, _ = _frequency_domain_ready(rr, cfg)
    if not freq_ok:
        return {"lf": np.nan, "hf": np.nan, "lf_hf": np.nan}

    times = np.cumsum(rr)[:-1]  # temps des intervalles (s)
    if times[-1] <= 0:
        return {"lf": np.nan, "hf": np.nan, "lf_hf": np.nan}

    target_fs = float(cfg.resample_fs)
    t_new = np.arange(0, times[-1], 1.0 / target_fs)
    if len(t_new) < 4:
        return {"lf": np.nan, "hf": np.nan, "lf_hf": np.nan}

    rr_interp = np.interp(t_new, times, rr[:-1])
    freqs, pxx = welch(rr_interp, fs=target_fs, nperseg=min(256, len(rr_interp)))

    lf_mask = (freqs >= cfg.lf_band[0]) & (freqs < cfg.lf_band[1])
    hf_mask = (freqs >= cfg.hf_band[0]) & (freqs < cfg.hf_band[1])
    lf_power = float(np.trapz(pxx[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else np.nan
    hf_power = float(np.trapz(pxx[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else np.nan
    lf_hf = float(lf_power / hf_power) if (hf_power and not np.isnan(hf_power) and hf_power > 0) else np.nan
    return {"lf": lf_power, "hf": hf_power, "lf_hf": lf_hf}


def _classify_hrv_quality(rr_reject_pct: float) -> str:
    if np.isnan(rr_reject_pct):
        return "unknown"
    if rr_reject_pct > 50.0:
        return "poor"
    if rr_reject_pct >= 20.0:
        return "acceptable"
    return "good"


def _quality_reason(
    rr_reject_pct: float,
    peak_rate_bpm: float,
    hr_median_bpm: float,
    rmssd: float,
    lf_hf: float,
    freq_qc_reason: str,
    n_rr: int,
    min_rr: int,
) -> str:
    if n_rr < min_rr:
        return "insufficient_rr"
    if np.isnan(rr_reject_pct):
        return "unknown"
    if rr_reject_pct > 50.0:
        return "high_rr_reject_pct"
    if rr_reject_pct >= 20.0:
        return "moderate_rr_reject_pct"
    if not np.isnan(peak_rate_bpm) and (peak_rate_bpm < 20.0 or peak_rate_bpm > 200.0):
        return "peak_rate_out_of_range"
    if not np.isnan(hr_median_bpm) and (hr_median_bpm < 20.0 or hr_median_bpm > 200.0):
        return "hr_median_out_of_range"
    if not np.isnan(peak_rate_bpm) and not np.isnan(hr_median_bpm):
        bpm_gap = abs(peak_rate_bpm - hr_median_bpm)
        bpm_tol = max(8.0, 0.15 * hr_median_bpm)
        if bpm_gap > bpm_tol:
            return "peak_hr_mismatch"
    if freq_qc_reason != "ok":
        return freq_qc_reason
    if not np.isnan(lf_hf) and lf_hf > 10.0:
        return "extreme_lf_hf"
    if not np.isnan(rmssd) and rmssd > 0.25:
        return "high_rmssd_review"
    return "ok"


def compute_epoch_hrv(
    raw,
    scoring_df,
    *,
    epoch_seconds: float = 30.0,
    channel_names: Optional[Sequence[str]] = None,
    config: Optional[HRVConfig] = None,
) -> List[Dict[str, object]]:
    """Calcule HRV sur segments (par défaut REM) en respectant une durée min."""

    cfg = config or HRVConfig()
    stage_set = {_normalize_stage_label(s) for s in cfg.stage_filter}
    # Si l'utilisateur saisit "REM", on ajoute aussi "R" pour compatibilité scoring
    if "REM" in {str(s).strip().upper() for s in cfg.stage_filter}:
        stage_set.add("R")
    if scoring_df is None or scoring_df.empty:
        return []

    sfreq = float(raw.info["sfreq"])
    channel = _pick_ecg_channel(raw, channel_names)
    if channel is None:
        raise HRVComputationError("Aucun canal ECG/PPG trouvé pour le calcul HRV.")

    ch_idx = raw.ch_names.index(channel)
    records: List[Dict[str, object]] = []

    # Regrouper les époques contiguës du/des stades cible(s) en segments
    rows = scoring_df.copy()
    rows = rows.sort_values("time")
    segments: List[Tuple[float, float, str]] = []  # (start, stop, stage)
    current_stage = None
    seg_start = None
    seg_stop = None
    for _, row in rows.iterrows():
        stage = _normalize_stage_label(row["stage"])
        if stage not in stage_set:
            # clôturer segment en cours
            if seg_start is not None and seg_stop is not None and current_stage is not None:
                segments.append((seg_start, seg_stop, current_stage))
            seg_start = seg_stop = current_stage = None
            continue
        t0 = float(row["time"])
        if seg_start is None:
            seg_start = t0
            seg_stop = t0 + float(epoch_seconds)
            current_stage = stage
        else:
            # continuité si chevauchement ou gap <= une demi-epoch
            if t0 <= seg_stop + (epoch_seconds * 0.5):
                seg_stop = max(seg_stop, t0 + float(epoch_seconds))
            else:
                segments.append((seg_start, seg_stop, current_stage))
                seg_start = t0
                seg_stop = t0 + float(epoch_seconds)
                current_stage = stage
    if seg_start is not None and seg_stop is not None and current_stage is not None:
        segments.append((seg_start, seg_stop, current_stage))

    LOGGER.info(
        "[HRV] Démarrage HRV: channel=%s, segments=%d, stages=%s, min_seg=%.1fs",
        channel,
        len(segments),
        ",".join(stage_set),
        cfg.min_segment_s,
    )

    for seg_start, seg_stop, stage in segments:
        duration = seg_stop - seg_start
        if duration < cfg.min_segment_s and not cfg.allow_short_segments:
            continue

        start_samp = int(max(0, min(raw.n_times - 1, np.floor(seg_start * sfreq))))
        stop_samp = int(min(raw.n_times, np.ceil(seg_stop * sfreq)))
        if stop_samp <= start_samp:
            continue

        data = raw.get_data(picks=[ch_idx], start=start_samp, stop=stop_samp)
        if data.size == 0:
            continue
        signal = data[0].astype(np.float64, copy=False)

        peaks = _detect_r_peaks(signal, sfreq, cfg)
        rr_raw = _rr_intervals(peaks, sfreq)
        rr, reject_pct_override, rr_correction_events = _clean_rr_intervals(peaks, sfreq, cfg)
        rmssd = _compute_rmssd(rr)
        freq_domain_ok, freq_qc_reason = _frequency_domain_ready(rr, cfg)
        freq = _compute_frequency_powers(rr, cfg)
        if reject_pct_override is not None:
            rr_reject_pct = float(reject_pct_override)
        else:
            rr_reject_pct = float(100.0 * (len(rr_raw) - len(rr)) / len(rr_raw)) if len(rr_raw) > 0 else np.nan
        rr_median = float(np.median(rr)) if len(rr) else np.nan
        peak_rate_bpm = float(len(peaks) * 60.0 / duration) if duration > 0 else np.nan
        hr_median_bpm = float(60.0 / rr_median) if not np.isnan(rr_median) and rr_median > 0 else np.nan
        hrv_quality = _classify_hrv_quality(rr_reject_pct)
        hrv_quality_reason = _quality_reason(
            rr_reject_pct,
            peak_rate_bpm,
            hr_median_bpm,
            rmssd,
            freq["lf_hf"],
            freq_qc_reason,
            len(rr),
            cfg.min_rr,
        )
        if hrv_quality == "poor":
            LOGGER.info(
                "[HRV] Segment qualité faible: stage=%s %.1f-%.1fs reject=%.1f%% peaks=%d rr=%d/%d method=%s/%s reason=%s",
                stage,
                seg_start,
                seg_stop,
                rr_reject_pct,
                len(peaks),
                len(rr),
                len(rr_raw),
                str(getattr(cfg, "peak_detection_method", "simple") or "simple").lower(),
                str(getattr(cfg, "rr_cleaning_method", "simple") or "simple").lower(),
                hrv_quality_reason,
            )

        records.append(
            {
                "stage": stage,
                "segment_start_s": seg_start,
                "segment_stop_s": seg_stop,
                "duration_s": duration,
                "rmssd": rmssd,
                "lf": freq["lf"],
                "hf": freq["hf"],
                "lf_hf": freq["lf_hf"],
                "n_peaks": int(len(peaks)),
                "n_rr_raw": int(len(rr_raw)),
                "n_rr": int(len(rr)),
                "rr_reject_pct": rr_reject_pct,
                "rr_correction_events": int(rr_correction_events),
                "rr_median": rr_median,
                "peak_rate_bpm": peak_rate_bpm,
                "hr_median_bpm": hr_median_bpm,
                "freq_domain_ok": bool(freq_domain_ok),
                "peak_detection_method": str(getattr(cfg, "peak_detection_method", "simple") or "simple").lower(),
                "rr_cleaning_method": str(getattr(cfg, "rr_cleaning_method", "simple") or "simple").lower(),
                "hrv_quality": hrv_quality,
                "hrv_quality_reason": hrv_quality_reason,
                "channel": channel,
            }
        )

    if not records:
        raise HRVComputationError(
            f"Aucun segment >= {cfg.min_segment_s:.0f}s trouvé pour stades {','.join(stage_set)} "
            f"(segments courts autorisés: {cfg.allow_short_segments}). "
            "Réduisez la durée min ou cochez \"segments courts\"."
        )

    LOGGER.info(
        "[HRV] Segments retenus: %d ; rmssd(med)=%.3f",
        len(records),
        float(np.median([r.get("rmssd", np.nan) for r in records])),
    )

    return records
