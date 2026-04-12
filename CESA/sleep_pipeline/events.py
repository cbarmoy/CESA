"""Respiratory and EEG event detection aligned with AASM criteria.

This module provides *optional* detection of:
- **Arousals** (cortical arousal from EEG alpha/beta bursts during sleep)
- **Apneas** (obstructive / central / mixed) from airflow channels
- **Hypopneas** with configurable criteria (default: >=3% desaturation OR arousal)
- **Oxygen desaturations** from SpO2 channel

All detectors work on the continuous (non-epoched) signal so that event
boundaries are not truncated by epoch limits.

Channels are *optional*: when a channel is not available, the corresponding
detector silently returns an empty list.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .contracts import EventType, ScoredEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: envelope
# ---------------------------------------------------------------------------

def _rms_envelope(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """Running RMS envelope (causal)."""
    sq = signal ** 2
    kernel = np.ones(window_samples) / window_samples
    return np.sqrt(np.convolve(sq, kernel, mode="same"))


def _lowpass(data: np.ndarray, sfreq: float, cutoff: float, order: int = 3) -> np.ndarray:
    nyq = sfreq / 2.0
    if cutoff >= nyq:
        return data
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


# ---------------------------------------------------------------------------
# Arousal detection (EEG)
# ---------------------------------------------------------------------------

def detect_arousals(
    eeg: np.ndarray,
    sfreq: float,
    *,
    min_duration_s: float = 3.0,
    max_duration_s: float = 15.0,
    alpha_beta_band: Tuple[float, float] = (8.0, 30.0),
    threshold_factor: float = 2.0,
) -> List[ScoredEvent]:
    """Detect cortical arousals as transient alpha/beta bursts.

    AASM criterion: abrupt shift in EEG frequency (including alpha, theta,
    or frequencies >16 Hz) lasting >= 3 s, preceded by >= 10 s of stable
    sleep.

    This implementation uses an amplitude-envelope approach on the
    alpha+beta band.
    """
    if eeg is None or len(eeg) == 0:
        return []

    nyq = sfreq / 2.0
    lo = alpha_beta_band[0] / nyq
    hi = min(alpha_beta_band[1] / nyq, 0.9999)
    sos = butter(4, [lo, hi], btype="band", output="sos")
    filtered = sosfiltfilt(sos, eeg)

    win = int(round(1.0 * sfreq))  # 1-s RMS window
    env = _rms_envelope(filtered, max(win, 1))

    baseline = np.median(env)
    if baseline < 1e-6:
        return []

    above = env > (baseline * threshold_factor)
    events: List[ScoredEvent] = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            dur_s = (i - start) / sfreq
            if min_duration_s <= dur_s <= max_duration_s:
                events.append(ScoredEvent(
                    event_type=EventType.AROUSAL,
                    onset_s=start / sfreq,
                    duration_s=dur_s,
                    confidence=0.7,
                ))
        else:
            i += 1

    logger.info("Arousal detection: %d events found", len(events))
    return events


# ---------------------------------------------------------------------------
# SpO2 desaturation detection
# ---------------------------------------------------------------------------

def detect_desaturations(
    spo2: np.ndarray,
    sfreq: float,
    *,
    drop_threshold_pct: float = 3.0,
    min_duration_s: float = 1.0,
    max_duration_s: float = 120.0,
) -> List[ScoredEvent]:
    """Detect oxygen desaturation events (>= *drop_threshold_pct* from baseline).

    AASM default criterion for hypopnea scoring: >= 3% desaturation.
    """
    if spo2 is None or len(spo2) == 0:
        return []

    smoothed = _lowpass(spo2.astype(float), sfreq, cutoff=0.5)
    # Rolling 90th-percentile baseline (2-minute window)
    win = int(round(120.0 * sfreq))
    if win < 1:
        return []

    events: List[ScoredEvent] = []
    stride = max(1, win // 4)
    for pos in range(0, len(smoothed) - win, stride):
        window = smoothed[pos: pos + win]
        baseline = np.percentile(window, 90)
        drop = baseline - np.min(window)
        if drop >= drop_threshold_pct:
            min_idx = pos + int(np.argmin(window))
            onset_s = min_idx / sfreq
            events.append(ScoredEvent(
                event_type=EventType.DESAT,
                onset_s=onset_s,
                duration_s=float(win / sfreq),
                confidence=min(1.0, drop / 10.0),
                metadata={"drop_pct": float(drop), "baseline": float(baseline)},
            ))

    events = _merge_overlapping(events)
    logger.info("SpO2 desaturation detection: %d events found", len(events))
    return events


# ---------------------------------------------------------------------------
# Apnea / hypopnea detection (airflow)
# ---------------------------------------------------------------------------

def detect_apneas_hypopneas(
    airflow: Optional[np.ndarray],
    sfreq: float,
    *,
    effort_signal: Optional[np.ndarray] = None,
    desaturations: Optional[List[ScoredEvent]] = None,
    arousals: Optional[List[ScoredEvent]] = None,
    apnea_amplitude_drop: float = 0.90,
    hypopnea_amplitude_drop: float = 0.30,
    min_duration_s: float = 10.0,
    desat_threshold_pct: float = 3.0,
) -> List[ScoredEvent]:
    """Detect apneas and hypopneas from airflow amplitude reduction.

    AASM criteria:
    - **Apnea**: >= 90% reduction in airflow amplitude for >= 10 s.
    - **Hypopnea**: >= 30% reduction for >= 10 s *with* either a >= 3%
      desaturation or an arousal.

    Central vs obstructive distinction requires an effort channel
    (thoracic or abdominal belt).
    """
    if airflow is None or len(airflow) == 0:
        return []

    env = _rms_envelope(airflow.astype(float), max(int(sfreq), 1))
    # Robust baseline: 95th percentile in 5-minute windows
    baseline_win = int(round(300.0 * sfreq))
    if baseline_win < 1:
        return []

    events: List[ScoredEvent] = []
    for pos in range(0, len(env) - baseline_win, baseline_win // 2):
        window_env = env[pos: pos + baseline_win]
        baseline_amp = np.percentile(window_env, 95)
        if baseline_amp < 1e-6:
            continue

        normed = window_env / baseline_amp
        # Scan for drops
        i = 0
        n = len(normed)
        while i < n:
            if normed[i] < (1.0 - hypopnea_amplitude_drop):
                start_local = i
                while i < n and normed[i] < (1.0 - hypopnea_amplitude_drop):
                    i += 1
                dur_s = (i - start_local) / sfreq
                if dur_s < min_duration_s:
                    continue
                onset_s = (pos + start_local) / sfreq
                min_val = float(np.min(normed[start_local:i]))
                drop_frac = 1.0 - min_val

                if drop_frac >= apnea_amplitude_drop:
                    etype = _classify_apnea_type(
                        effort_signal, sfreq, onset_s, dur_s,
                    )
                    events.append(ScoredEvent(
                        event_type=etype,
                        onset_s=onset_s,
                        duration_s=dur_s,
                        confidence=0.75,
                        metadata={"drop_frac": float(drop_frac)},
                    ))
                else:
                    # Hypopnea: requires associated desaturation or arousal
                    has_desat = _has_associated_event(
                        desaturations or [], onset_s, dur_s, margin_s=30.0,
                    )
                    has_arousal = _has_associated_event(
                        arousals or [], onset_s, dur_s, margin_s=10.0,
                    )
                    if has_desat or has_arousal:
                        events.append(ScoredEvent(
                            event_type=EventType.HYPOPNEA,
                            onset_s=onset_s,
                            duration_s=dur_s,
                            confidence=0.60,
                            metadata={
                                "drop_frac": float(drop_frac),
                                "associated_desat": has_desat,
                                "associated_arousal": has_arousal,
                            },
                        ))
            else:
                i += 1

    events = _merge_overlapping(events)
    logger.info("Apnea/hypopnea detection: %d events found", len(events))
    return events


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_apnea_type(
    effort: Optional[np.ndarray],
    sfreq: float,
    onset_s: float,
    duration_s: float,
) -> EventType:
    """Distinguish obstructive vs central apnea using effort signal."""
    if effort is None or len(effort) == 0:
        return EventType.APNEA_OBSTRUCTIVE  # default when effort unavailable

    start = int(onset_s * sfreq)
    end = int((onset_s + duration_s) * sfreq)
    end = min(end, len(effort))
    if start >= end:
        return EventType.APNEA_OBSTRUCTIVE

    segment = effort[start:end]
    rms_effort = float(np.sqrt(np.mean(segment ** 2)))
    # Very low effort during the event suggests central apnea
    baseline_effort = float(np.sqrt(np.mean(effort ** 2)))
    if baseline_effort < 1e-6:
        return EventType.APNEA_OBSTRUCTIVE

    ratio = rms_effort / baseline_effort
    if ratio < 0.3:
        return EventType.APNEA_CENTRAL
    elif ratio < 0.6:
        return EventType.APNEA_MIXED
    return EventType.APNEA_OBSTRUCTIVE


def _has_associated_event(
    events: List[ScoredEvent],
    onset_s: float,
    duration_s: float,
    margin_s: float,
) -> bool:
    """Check if any event in *events* overlaps with [onset_s, onset_s+duration_s+margin_s]."""
    end = onset_s + duration_s + margin_s
    for ev in events:
        ev_end = ev.onset_s + ev.duration_s
        if ev.onset_s <= end and ev_end >= onset_s:
            return True
    return False


def _merge_overlapping(events: List[ScoredEvent]) -> List[ScoredEvent]:
    """Remove duplicate / overlapping events of the same type."""
    if len(events) <= 1:
        return events
    events = sorted(events, key=lambda e: e.onset_s)
    merged: List[ScoredEvent] = [events[0]]
    for ev in events[1:]:
        last = merged[-1]
        if (
            ev.event_type == last.event_type
            and ev.onset_s <= last.onset_s + last.duration_s
        ):
            new_end = max(last.onset_s + last.duration_s, ev.onset_s + ev.duration_s)
            merged[-1] = ScoredEvent(
                event_type=last.event_type,
                onset_s=last.onset_s,
                duration_s=new_end - last.onset_s,
                confidence=max(last.confidence, ev.confidence),
                metadata={**last.metadata, **ev.metadata},
            )
        else:
            merged.append(ev)
    return merged
