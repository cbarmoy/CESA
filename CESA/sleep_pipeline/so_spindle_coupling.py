"""Classical SO-spindle coupling: filtering, Hilbert phase, preferred phase.

Second detection path next to :mod:`CESA.sleep_pipeline.micro_events` (YASA).
Everything here is explicit and reproducible: zero-phase Butterworth bands,
Hilbert envelope for spindles, Hilbert phase for slow oscillations, then
circular statistics (``pycircstat2``) on the SO phases sampled at the spindle
amplitude peaks.

Phase convention: the analytic phase of the SO-filtered signal is 0 rad at the
positive peak (up-state) and +/-pi at the negative trough (down-state).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from .contracts import EventType, ScoredEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection constants
# ---------------------------------------------------------------------------

SO_BAND_HZ: Tuple[float, float] = (0.5, 1.25)
SO_MIN_DURATION_S = 0.8
SO_MAX_DURATION_S = 2.0
SO_NEG_PEAK_SD = 1.5

SPINDLE_BAND_HZ: Tuple[float, float] = (12.0, 16.0)
SPINDLE_MIN_DURATION_S = 0.5
SPINDLE_MAX_DURATION_S = 2.0
SPINDLE_THRESHOLD_SD = 1.5

FILTER_ORDER = 4

# YASA-style numeric stages: N2 only for spindles, N2+N3 for slow oscillations.
SO_STAGES: Tuple[int, ...] = (2, 3)
SPINDLE_STAGES: Tuple[int, ...] = (2,)

# A 20-min nap yields few events; below this many coupled spindles the
# preferred phase is not reported (counts still are).
MIN_COUPLED_EVENTS = 8

# At least this much usable signal is needed to estimate detection thresholds.
MIN_ANALYSIS_DURATION_S = 30.0

# Surrogate draws used when the permutation z-score of R is requested.
DEFAULT_N_PERM = 500

STATS_COLUMNS: List[str] = [
    "channel", "n_so", "n_spindle", "n_coupled",
    "pct_coupled_spindle", "pct_coupled_so", "coupled_per_h",
    "amp_coupled_uv", "amp_spindle_uv",
    "preferred_phase_rad", "preferred_phase_deg", "resultant_length",
    "rayleigh_z", "rayleigh_p", "v_stat", "v_p", "backend",
]

# Appended only when analyze_channel(..., permute=True).
PERM_STATS_COLUMNS: List[str] = ["r_z", "r_z_p"]


# ---------------------------------------------------------------------------
# Detected structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlowOscillation:
    """One SO cycle, delimited by two descending zero crossings."""

    start_idx: int
    end_idx: int
    trough_idx: int
    trough_uv: float
    ptp_uv: float


@dataclass(frozen=True)
class Spindle:
    """One sigma-band burst, delimited by envelope threshold crossings."""

    start_idx: int
    end_idx: int
    peak_idx: int
    peak_amplitude_uv: float
    frequency_hz: float


@dataclass
class CouplingResult:
    """Events plus per-channel circular statistics for the whole recording."""

    events: List[ScoredEvent] = field(default_factory=list)
    stats: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=STATS_COLUMNS)
    )
    warnings: List[str] = field(default_factory=list)
    backend: str = "none"


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def bandpass(
    signal: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass (``sosfiltfilt``)."""
    nyq = float(sfreq) / 2.0
    lo = max(float(low) / nyq, 1e-5)
    hi = min(float(high) / nyq, 0.9999)
    if lo >= hi:
        return np.asarray(signal, dtype=float)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=float))


def stage_mask(
    hypno: Optional[np.ndarray],
    stages: Sequence[int],
    n_samples: int,
) -> Optional[np.ndarray]:
    """Boolean sample mask for *stages*; ``None`` when no hypnogram is given."""
    if hypno is None:
        return None
    hyp = np.asarray(hypno).ravel()
    if hyp.size == 0:
        return None
    if hyp.size < n_samples:
        hyp = np.pad(hyp, (0, n_samples - hyp.size), constant_values=-1)
    return np.isin(hyp[:n_samples], np.asarray(list(stages), dtype=hyp.dtype))


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to ``[-pi, pi)``."""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def resultant_length(phases: Sequence[float]) -> float:
    """Mean resultant length of *phases* in radians, or NaN if empty."""
    angles = np.asarray(list(phases), dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return float("nan")
    return float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))


def _runs(flags: np.ndarray) -> List[Tuple[int, int]]:
    """Start/stop index pairs of contiguous ``True`` runs in *flags*."""
    if flags.size == 0:
        return []
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def _circular_backend():
    """Return ``(circ_mean_and_r, rayleigh_test)`` or ``None``."""
    try:
        from pycircstat2.descriptive import circ_mean_and_r
        from pycircstat2.hypothesis import rayleigh_test
    except ImportError:
        return None
    return circ_mean_and_r, rayleigh_test


def _v_test_numpy(
    mean_rad: float, r: float, n: int, hypothesized: float = 0.0,
) -> Tuple[float, float]:
    """V-test against *hypothesized* radians (Zar 1999).

    ``V = n R cos(mean - mu0)``; ``u = V * sqrt(2/n)``; ``p = 1 - Phi(u)``.
    """
    if n <= 0 or not np.isfinite(r) or not np.isfinite(mean_rad):
        return float("nan"), float("nan")
    from math import erfc

    v_stat = float(n) * float(r) * float(np.cos(mean_rad - hypothesized))
    u = v_stat * np.sqrt(2.0 / float(n))
    p = 0.5 * float(erfc(u / np.sqrt(2.0)))
    return float(v_stat), float(min(1.0, max(0.0, p)))


def _v_test(angles: np.ndarray, hypothesized: float = 0.0) -> Tuple[float, float]:
    """V-test of clustering around *hypothesized* radians (0 = SO up-state)."""
    n = int(angles.size)
    if n == 0:
        return float("nan"), float("nan")
    try:
        from pycircstat2.hypothesis import V_test

        res = V_test(angle=float(hypothesized), alpha=angles)
        if np.isfinite(res.V) and np.isfinite(res.pval):
            return float(res.V), float(res.pval)
    except Exception:
        pass
    r = resultant_length(angles)
    mean_rad = wrap_to_pi(float(np.arctan2(
        np.mean(np.sin(angles)), np.mean(np.cos(angles)),
    )))
    return _v_test_numpy(mean_rad, r, n, hypothesized)


def permute_resultant_length(
    observed_r: float,
    n: int,
    candidate_phases: Sequence[float],
    *,
    n_perm: int = DEFAULT_N_PERM,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Z-score of *observed_r* against random times inside SO cycles.

    The null draws *n* phases from *candidate_phases* (SO-cycle samples), so
    the surrogate keeps the non-uniform occupancy of the SO and only breaks
    the spindle timing. Returns ``(r_z, r_z_p)``.
    """
    pool = np.asarray(list(candidate_phases), dtype=float)
    pool = pool[np.isfinite(pool)]
    if (
        n <= 0
        or pool.size < 2
        or not np.isfinite(observed_r)
        or int(n_perm) < 2
    ):
        return float("nan"), float("nan")
    generator = rng if rng is not None else np.random.default_rng()
    null = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        draw = pool[generator.integers(0, pool.size, size=int(n))]
        null[i] = resultant_length(draw)
    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1))
    if sd <= 0.0:
        return float("nan"), float("nan")
    z = (float(observed_r) - mu) / sd
    p = float(np.mean(null >= float(observed_r)))
    return float(z), float(min(1.0, max(0.0, p)))


def circular_phase_stats(phases: Sequence[float]) -> Dict[str, Any]:
    """Preferred phase, resultant length, Rayleigh and V-test over *phases*.

    The V-test asks whether the phases cluster around the SO up-state (0 rad),
    which is the direction predicted by the coupling literature. Uses
    ``pycircstat2`` when installed and falls back to the equivalent closed-form
    estimators otherwise; the backend used is reported so a result is never
    silently computed by a different implementation than expected.
    """
    angles = np.asarray(list(phases), dtype=float)
    angles = angles[np.isfinite(angles)]
    n = int(angles.size)
    out: Dict[str, Any] = {
        "n": n,
        "preferred_phase_rad": float("nan"),
        "preferred_phase_deg": float("nan"),
        "resultant_length": float("nan"),
        "rayleigh_z": float("nan"),
        "rayleigh_p": float("nan"),
        "v_stat": float("nan"),
        "v_p": float("nan"),
        "backend": "none",
    }
    if n == 0:
        return out

    backend = _circular_backend()
    if backend is not None:
        circ_mean_and_r, rayleigh_test = backend
        try:
            mean_angle, r = circ_mean_and_r(angles)
            res = rayleigh_test(r=float(r), n=n)
            v_stat, v_p = _v_test(angles, hypothesized=0.0)
            out.update(
                preferred_phase_rad=wrap_to_pi(mean_angle),
                resultant_length=float(r),
                rayleigh_z=float(res.z),
                rayleigh_p=float(res.pval),
                v_stat=v_stat,
                v_p=v_p,
                backend="pycircstat2",
            )
            out["preferred_phase_deg"] = float(
                np.degrees(out["preferred_phase_rad"])
            )
            return out
        except Exception as exc:
            logger.warning("pycircstat2 failed, falling back to numpy: %s", exc)

    cos_m = float(np.mean(np.cos(angles)))
    sin_m = float(np.mean(np.sin(angles)))
    r = float(np.hypot(cos_m, sin_m))
    resultant = n * r
    z = resultant * resultant / n
    # Zar (1999), eq. 27.4: small-sample approximation of the Rayleigh p-value.
    p = float(np.exp(
        np.sqrt(1.0 + 4.0 * n + 4.0 * (n * n - resultant * resultant))
        - (1.0 + 2.0 * n)
    ))
    mean_rad = wrap_to_pi(np.arctan2(sin_m, cos_m))
    v_stat, v_p = _v_test_numpy(mean_rad, r, n, hypothesized=0.0)
    out.update(
        preferred_phase_rad=mean_rad,
        resultant_length=r,
        rayleigh_z=float(z),
        rayleigh_p=min(1.0, p),
        v_stat=v_stat,
        v_p=v_p,
        backend="numpy",
    )
    out["preferred_phase_deg"] = float(np.degrees(out["preferred_phase_rad"]))
    return out


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def detect_slow_oscillations(
    signal: np.ndarray,
    sfreq: float,
    *,
    mask: Optional[np.ndarray] = None,
    band: Tuple[float, float] = SO_BAND_HZ,
    min_duration_s: float = SO_MIN_DURATION_S,
    max_duration_s: float = SO_MAX_DURATION_S,
    neg_peak_sd: float = SO_NEG_PEAK_SD,
    filtered: Optional[np.ndarray] = None,
) -> List[SlowOscillation]:
    """Detect SO cycles on a single channel (uV) by zero-crossing segmentation.

    A cycle runs from one descending zero crossing to the next; it is kept when
    its duration fits *min_duration_s*..*max_duration_s* and its negative peak
    falls below ``mean - neg_peak_sd * SD`` of the filtered signal.
    """
    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size < 2 or sfreq <= 0:
        return []
    so = filtered if filtered is not None else bandpass(sig, sfreq, *band)

    usable = so if mask is None else so[mask]
    if usable.size < 2:
        return []
    threshold = float(np.mean(usable) - neg_peak_sd * np.std(usable))

    descending = np.flatnonzero((so[:-1] >= 0.0) & (so[1:] < 0.0)) + 1
    if descending.size < 2:
        return []

    min_samples = int(round(min_duration_s * sfreq))
    max_samples = int(round(max_duration_s * sfreq))

    events: List[SlowOscillation] = []
    for start, end in zip(descending[:-1].tolist(), descending[1:].tolist()):
        length = end - start
        if not (min_samples <= length <= max_samples):
            continue
        if mask is not None and not mask[start:end].all():
            continue
        cycle = so[start:end]
        trough_offset = int(np.argmin(cycle))
        trough = float(cycle[trough_offset])
        if trough > threshold:
            continue
        events.append(SlowOscillation(
            start_idx=start,
            end_idx=end,
            trough_idx=start + trough_offset,
            trough_uv=trough,
            ptp_uv=float(np.max(cycle) - trough),
        ))
    return events


def detect_spindles_hilbert(
    signal: np.ndarray,
    sfreq: float,
    *,
    mask: Optional[np.ndarray] = None,
    band: Tuple[float, float] = SPINDLE_BAND_HZ,
    min_duration_s: float = SPINDLE_MIN_DURATION_S,
    max_duration_s: float = SPINDLE_MAX_DURATION_S,
    threshold_sd: float = SPINDLE_THRESHOLD_SD,
) -> List[Spindle]:
    """Detect sigma bursts on a single channel (uV) via Hilbert envelope.

    A spindle is a run where the envelope exceeds ``mean + threshold_sd * SD``
    for *min_duration_s*..*max_duration_s*; its peak is the envelope maximum.
    """
    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size < 2 or sfreq <= 0:
        return []
    sigma = bandpass(sig, sfreq, *band)
    envelope = np.abs(hilbert(sigma))

    usable = envelope if mask is None else envelope[mask]
    if usable.size < 2:
        return []
    threshold = float(np.mean(usable) + threshold_sd * np.std(usable))

    above = envelope > threshold
    if mask is not None:
        above &= mask

    min_samples = int(round(min_duration_s * sfreq))
    max_samples = int(round(max_duration_s * sfreq))

    events: List[Spindle] = []
    for start, end in _runs(above):
        length = end - start
        if not (min_samples <= length <= max_samples):
            continue
        segment = envelope[start:end]
        peak_offset = int(np.argmax(segment))
        events.append(Spindle(
            start_idx=start,
            end_idx=end,
            peak_idx=start + peak_offset,
            peak_amplitude_uv=float(segment[peak_offset]),
            frequency_hz=_segment_frequency(sigma[start:end], sfreq),
        ))
    return events


def _segment_frequency(segment: np.ndarray, sfreq: float) -> float:
    """Mean oscillation frequency of *segment* from its zero crossings."""
    if segment.size < 2 or sfreq <= 0:
        return float("nan")
    crossings = int(np.count_nonzero(np.diff(np.sign(segment)) != 0))
    duration = segment.size / float(sfreq)
    if duration <= 0 or crossings == 0:
        return float("nan")
    return float(crossings / (2.0 * duration))


def so_phase(
    signal: np.ndarray,
    sfreq: float,
    *,
    band: Tuple[float, float] = SO_BAND_HZ,
    filtered: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Instantaneous SO phase in radians (0 = positive peak, +/-pi = trough)."""
    sig = np.asarray(signal, dtype=float).ravel()
    so = filtered if filtered is not None else bandpass(sig, sfreq, *band)
    return np.angle(hilbert(so))


def couple_spindles_to_so(
    slow_oscillations: Sequence[SlowOscillation],
    spindles: Sequence[Spindle],
    phase: np.ndarray,
) -> Dict[int, Tuple[int, float]]:
    """Map spindle index -> (SO index, SO phase at the spindle peak).

    A spindle is coupled when its envelope peak falls inside an SO cycle.
    """
    if not slow_oscillations or not spindles:
        return {}
    starts = np.array([so.start_idx for so in slow_oscillations])
    ends = np.array([so.end_idx for so in slow_oscillations])

    coupled: Dict[int, Tuple[int, float]] = {}
    for i, spindle in enumerate(spindles):
        peak = spindle.peak_idx
        hits = np.flatnonzero((starts <= peak) & (ends > peak))
        if hits.size == 0:
            continue
        so_index = int(hits[0])
        coupled[i] = (so_index, wrap_to_pi(phase[peak]))
    return coupled


# ---------------------------------------------------------------------------
# Per-channel and whole-recording analysis
# ---------------------------------------------------------------------------

def analyze_channel(
    signal: np.ndarray,
    sfreq: float,
    channel: str = "",
    *,
    so_mask: Optional[np.ndarray] = None,
    spindle_mask: Optional[np.ndarray] = None,
    so_band: Tuple[float, float] = SO_BAND_HZ,
    spindle_band: Tuple[float, float] = SPINDLE_BAND_HZ,
    min_coupled: int = MIN_COUPLED_EVENTS,
    permute: bool = False,
    n_perm: int = DEFAULT_N_PERM,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[ScoredEvent], Dict[str, Any]]:
    """Detect SO + spindles on one channel and compute their coupling stats.

    *permute* adds ``r_z`` / ``r_z_p``, a z-score of R against random times
    inside the detected SO cycles. Skip it when speed matters: the rest of
    the table does not need a second pass over the signal.
    """
    sig = np.array(signal, dtype=float, copy=True).ravel()
    so_filtered = bandpass(sig, sfreq, *so_band)
    phase = so_phase(sig, sfreq, filtered=so_filtered)

    slow_oscillations = detect_slow_oscillations(
        sig, sfreq, mask=so_mask, band=so_band, filtered=so_filtered,
    )
    spindles = detect_spindles_hilbert(
        sig, sfreq, mask=spindle_mask, band=spindle_band,
    )
    coupled = couple_spindles_to_so(slow_oscillations, spindles, phase)

    stats = circular_phase_stats([ph for _, ph in coupled.values()])
    n_coupled = int(stats.pop("n"))
    enough = n_coupled >= int(min_coupled)
    if not enough:
        stats.update(
            preferred_phase_rad=float("nan"),
            preferred_phase_deg=float("nan"),
            resultant_length=float("nan"),
            rayleigh_z=float("nan"),
            rayleigh_p=float("nan"),
            v_stat=float("nan"),
            v_p=float("nan"),
        )

    coupled_so = {so_idx for so_idx, _ in coupled.values()}
    events: List[ScoredEvent] = []
    for i, so in enumerate(slow_oscillations):
        events.append(ScoredEvent(
            event_type=EventType.SLOW_WAVE,
            onset_s=so.start_idx / sfreq,
            duration_s=(so.end_idx - so.start_idx) / sfreq,
            metadata={
                "channel": channel,
                "detector": "cesa_classic",
                "PTP": round(so.ptp_uv, 3),
                "trough_uv": round(so.trough_uv, 3),
                "trough_s": round(so.trough_idx / sfreq, 3),
                "CooccurringSpindle": i in coupled_so,
            },
        ))
    for i, spindle in enumerate(spindles):
        meta: Dict[str, Any] = {
            "channel": channel,
            "detector": "cesa_classic",
            "Frequency": (
                round(spindle.frequency_hz, 3)
                if np.isfinite(spindle.frequency_hz) else None
            ),
            "peak_amplitude_uv": round(spindle.peak_amplitude_uv, 3),
            "peak_s": round(spindle.peak_idx / sfreq, 3),
        }
        if i in coupled:
            so_index, ph = coupled[i]
            meta["so_phase_rad"] = round(ph, 4)
            meta["so_phase_deg"] = round(float(np.degrees(ph)), 2)
            meta["coupled_so_onset_s"] = round(
                slow_oscillations[so_index].start_idx / sfreq, 3
            )
        events.append(ScoredEvent(
            event_type=EventType.SPINDLE,
            onset_s=spindle.start_idx / sfreq,
            duration_s=(spindle.end_idx - spindle.start_idx) / sfreq,
            metadata=meta,
        ))

    n_so = len(slow_oscillations)
    n_spindle = len(spindles)
    hours = (float(sig.size) / float(sfreq) / 3600.0) if sfreq > 0 else 0.0
    amps = [sp.peak_amplitude_uv for sp in spindles]
    amp_coupled = [
        spindles[i].peak_amplitude_uv for i in coupled
    ]
    row: Dict[str, Any] = {
        "channel": channel,
        "n_so": n_so,
        "n_spindle": n_spindle,
        "n_coupled": n_coupled,
        "pct_coupled_spindle": (
            round(100.0 * n_coupled / n_spindle, 2) if n_spindle else 0.0
        ),
        "pct_coupled_so": (
            round(100.0 * len(coupled_so) / n_so, 2) if n_so else 0.0
        ),
        "coupled_per_h": round(n_coupled / hours, 2) if hours > 0 else 0.0,
        "amp_coupled_uv": (
            round(float(np.mean(amp_coupled)), 3) if amp_coupled else float("nan")
        ),
        "amp_spindle_uv": (
            round(float(np.mean(amps)), 3) if amps else float("nan")
        ),
        **stats,
    }
    if permute:
        if enough:
            pool: List[float] = []
            for so in slow_oscillations:
                pool.extend(phase[so.start_idx:so.end_idx].tolist())
            r_z, r_z_p = permute_resultant_length(
                float(stats.get("resultant_length", float("nan"))),
                n_coupled, pool, n_perm=n_perm, rng=rng,
            )
        else:
            r_z = r_z_p = float("nan")
        row["r_z"] = r_z
        row["r_z_p"] = r_z_p
    return events, row


def detect_so_spindle_coupling(
    data: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    hypno: Optional[np.ndarray] = None,
    *,
    so_band: Tuple[float, float] = SO_BAND_HZ,
    spindle_band: Tuple[float, float] = SPINDLE_BAND_HZ,
    so_stages: Sequence[int] = SO_STAGES,
    spindle_stages: Sequence[int] = SPINDLE_STAGES,
    min_coupled: int = MIN_COUPLED_EVENTS,
    permute: bool = False,
    n_perm: int = DEFAULT_N_PERM,
    progress_callback: Optional[Any] = None,
) -> CouplingResult:
    """Run the classical SO-spindle coupling pipeline over all channels.

    *data* is ``(n_channels, n_samples)`` in uV; *hypno* is the sample-level
    YASA-style hypnogram built by
    :func:`CESA.sleep_pipeline.micro_events.scoring_df_to_hypno`.
    *permute* adds the surrogate z-score of R (see :func:`analyze_channel`).
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
        ch_names = list(ch_names) or ["EEG"]
    names = list(ch_names)
    if arr.shape[0] != len(names):
        raise ValueError(
            f"data channels ({arr.shape[0]}) != ch_names ({len(names)})"
        )
    if sfreq <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")

    result = CouplingResult()
    n_samples = int(arr.shape[1])
    if n_samples < int(MIN_ANALYSIS_DURATION_S * sfreq):
        result.warnings.append(
            "Signal trop court pour l'analyse "
            f"({n_samples / sfreq:.1f} s < {MIN_ANALYSIS_DURATION_S:.0f} s)."
        )
        return result

    so_mask = stage_mask(hypno, so_stages, n_samples)
    spindle_mask = stage_mask(hypno, spindle_stages, n_samples)
    min_samples = int(MIN_ANALYSIS_DURATION_S * sfreq)
    if so_mask is not None and int(so_mask.sum()) < min_samples:
        result.warnings.append(
            "Trop peu de N2/N3 pour detecter les ondes lentes "
            f"({so_mask.sum() / sfreq:.0f} s)."
        )
    if spindle_mask is not None and int(spindle_mask.sum()) < min_samples:
        result.warnings.append(
            "Trop peu de N2 pour detecter les fuseaux "
            f"({spindle_mask.sum() / sfreq:.0f} s)."
        )

    rows: List[Dict[str, Any]] = []
    backends = set()
    for i, name in enumerate(names):
        if progress_callback:
            progress_callback(f"Couplage SO-fuseau: {name}...")
        try:
            events, row = analyze_channel(
                np.array(arr[i], dtype=float, copy=True), sfreq, name,
                so_mask=so_mask,
                spindle_mask=spindle_mask,
                so_band=so_band,
                spindle_band=spindle_band,
                min_coupled=min_coupled,
                permute=permute,
                n_perm=n_perm,
            )
        except Exception as exc:
            logger.error("Coupling failed on %s: %s", name, exc, exc_info=True)
            result.warnings.append(f"{name}: {exc}")
            continue
        result.events.extend(events)
        rows.append(row)
        backends.add(str(row.get("backend", "none")))
        if row["n_coupled"] < int(min_coupled):
            result.warnings.append(
                f"{name}: seulement {row['n_coupled']} fuseau(x) couple(s) "
                f"(minimum {int(min_coupled)}); phase preferee non publiee."
            )

    columns = STATS_COLUMNS + (PERM_STATS_COLUMNS if permute else [])
    result.stats = (
        pd.DataFrame(rows)[columns] if rows
        else pd.DataFrame(columns=columns)
    )
    if "pycircstat2" in backends:
        result.backend = "pycircstat2"
    elif "numpy" in backends:
        result.backend = "numpy"
    if result.backend == "numpy":
        result.warnings.append(
            "pycircstat2 absent: statistiques circulaires calculees avec numpy."
        )
    logger.info(
        "[COUPLING] channels=%d events=%d backend=%s",
        len(rows), len(result.events), result.backend,
    )
    return result
