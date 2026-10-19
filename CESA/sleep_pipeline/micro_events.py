"""YASA-backed micro-architecture event detection (spindles, slow waves, REM).

Wraps ``yasa.spindles_detect``, ``yasa.sw_detect`` (optional SO–sigma PAC) and
``yasa.rem_detect`` into :class:`~CESA.sleep_pipeline.contracts.ScoredEvent`
lists suitable for the Qt viewer overlays and CSV export.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import EventType, ScoredEvent, StageLabel

logger = logging.getLogger(__name__)

# YASA numeric stage codes
_YASA_STAGE_INT = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
    "U": -1,
}

_EVENT_ANN_TYPE = {
    EventType.SPINDLE: "spindle",
    EventType.SLOW_WAVE: "slow_wave",
    EventType.REM_BURST: "rem_burst",
    EventType.K_COMPLEX: "k_complex",
}


class YasaUnavailableError(RuntimeError):
    """Raised when the ``yasa`` package cannot be imported."""


def _require_yasa():
    try:
        import yasa  # noqa: F401
        return yasa
    except ImportError as exc:
        raise YasaUnavailableError(
            "Le package YASA est requis pour la detection micro-architecture. "
            "Installez-le avec: pip install yasa>=0.6.0"
        ) from exc


def scoring_df_to_hypno(
    scoring_df: Optional[pd.DataFrame],
    n_samples: int,
    sfreq: float,
    epoch_seconds: float = 30.0,
) -> Optional[np.ndarray]:
    """Build a sample-level YASA hypnogram (0=W … 4=REM, -1=U).

    Returns ``None`` when *scoring_df* is missing or empty.
    """
    if scoring_df is None or scoring_df.empty or n_samples <= 0 or sfreq <= 0:
        return None
    hypno = np.full(int(n_samples), -1, dtype=np.int16)
    epoch_s = float(epoch_seconds) if epoch_seconds and epoch_seconds > 0 else 30.0
    for _, row in scoring_df.iterrows():
        try:
            t0 = float(row["time"])
        except (KeyError, TypeError, ValueError):
            continue
        stage = StageLabel.from_string(str(row.get("stage", "U")))
        code = _YASA_STAGE_INT.get(stage.value, -1)
        i0 = max(0, int(round(t0 * sfreq)))
        i1 = min(n_samples, int(round((t0 + epoch_s) * sfreq)))
        if i1 > i0:
            hypno[i0:i1] = code
    return hypno


def _safe_float(val: Any, default: float = float("nan")) -> float:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        if pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _row_to_scored_event(
    row: pd.Series,
    event_type: EventType,
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> ScoredEvent:
    """Convert one YASA summary row into a :class:`ScoredEvent`."""
    onset = _safe_float(row.get("Start", row.get("start", 0.0)), 0.0)
    duration = _safe_float(row.get("Duration", row.get("duration", 0.0)), 0.0)
    channel = str(row.get("Channel", row.get("channel", "")) or "")
    conf = _safe_float(row.get("Confidence", row.get("confidence", 1.0)), 1.0)
    if not np.isfinite(conf):
        conf = 1.0

    meta: Dict[str, Any] = {"channel": channel}
    for key in (
        "Frequency", "AbsPower", "RelPower", "RMS", "Oscillations",
        "Symmetry", "Peak", "ValNegPeak", "ValPosPeak", "PTP",
        "Slope", "Frequency", "Stage",
        "ndPAC", "PhaseAtSigmaPeak", "SigmaPeak",
        "CooccurringSpindle", "CooccurringSpindlePeak", "DistanceSpindleToSW",
        "Idx", "IdxChannel",
    ):
        if key in row.index:
            val = row[key]
            if pd.isna(val):
                continue
            if isinstance(val, (np.floating, float)):
                meta[key] = float(val)
            elif isinstance(val, (np.integer, int)):
                meta[key] = int(val)
            elif isinstance(val, (bool, np.bool_)):
                meta[key] = bool(val)
            else:
                meta[key] = val
    if extra_meta:
        meta.update(extra_meta)
    return ScoredEvent(
        event_type=event_type,
        onset_s=float(onset),
        duration_s=max(0.0, float(duration)),
        confidence=float(conf),
        metadata=meta,
    )


def _summary_to_events(
    summary: Optional[pd.DataFrame],
    event_type: EventType,
) -> List[ScoredEvent]:
    if summary is None or summary.empty:
        return []
    return [_row_to_scored_event(row, event_type) for _, row in summary.iterrows()]


def detect_spindles(
    data: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    hypno: Optional[np.ndarray] = None,
    include: Sequence[int] = (2,),
    **kwargs: Any,
) -> List[ScoredEvent]:
    """Detect sleep spindles via YASA (data in µV, shape ``(n_ch, n_times)``)."""
    yasa = _require_yasa()
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
        ch_names = list(ch_names) or ["EEG"]
    if data.shape[0] != len(ch_names):
        raise ValueError(
            f"data channels ({data.shape[0]}) != ch_names ({len(ch_names)})"
        )
    try:
        res = yasa.spindles_detect(
            data,
            sf=float(sfreq),
            ch_names=list(ch_names),
            hypno=hypno,
            include=tuple(include),
            **kwargs,
        )
    except Exception as exc:
        logger.error("spindles_detect failed: %s", exc, exc_info=True)
        return []
    if res is None:
        return []
    return _summary_to_events(res.summary(), EventType.SPINDLE)


def detect_slow_waves(
    data: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    hypno: Optional[np.ndarray] = None,
    include: Sequence[int] = (2, 3),
    *,
    coupling: bool = True,
    spindle_summary: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> List[ScoredEvent]:
    """Detect slow waves; optionally compute SO–sigma PAC and co-occurrence."""
    yasa = _require_yasa()
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
        ch_names = list(ch_names) or ["EEG"]
    if data.shape[0] != len(ch_names):
        raise ValueError(
            f"data channels ({data.shape[0]}) != ch_names ({len(ch_names)})"
        )
    try:
        res = yasa.sw_detect(
            data,
            sf=float(sfreq),
            ch_names=list(ch_names),
            hypno=hypno,
            include=tuple(include),
            coupling=bool(coupling),
            **kwargs,
        )
    except Exception as exc:
        logger.error("sw_detect failed: %s", exc, exc_info=True)
        return []
    if res is None:
        return []
    if spindle_summary is not None and not spindle_summary.empty:
        try:
            res.find_cooccurring_spindles(spindle_summary)
        except Exception as exc:
            logger.warning("find_cooccurring_spindles failed: %s", exc)
    return _summary_to_events(res.summary(), EventType.SLOW_WAVE)


def detect_rem_bursts(
    loc: np.ndarray,
    roc: np.ndarray,
    sfreq: float,
    *,
    hypno: Optional[np.ndarray] = None,
    include: int = 4,
    loc_name: str = "LOC",
    roc_name: str = "ROC",
    **kwargs: Any,
) -> List[ScoredEvent]:
    """Detect rapid eye movements from left/right EOG (µV)."""
    yasa = _require_yasa()
    loc = np.asarray(loc, dtype=np.float64).ravel()
    roc = np.asarray(roc, dtype=np.float64).ravel()
    if loc.size == 0 or roc.size == 0:
        return []
    n = min(loc.size, roc.size)
    loc, roc = loc[:n], roc[:n]
    try:
        res = yasa.rem_detect(
            loc,
            roc,
            sf=float(sfreq),
            hypno=hypno[:n] if hypno is not None else None,
            include=include,
            **kwargs,
        )
    except Exception as exc:
        logger.error("rem_detect failed: %s", exc, exc_info=True)
        return []
    if res is None:
        return []
    events = _summary_to_events(res.summary(), EventType.REM_BURST)
    # Tag channel pair in metadata
    tagged: List[ScoredEvent] = []
    for ev in events:
        meta = dict(ev.metadata)
        meta["channel"] = meta.get("channel") or f"{loc_name}/{roc_name}"
        meta["loc_channel"] = loc_name
        meta["roc_channel"] = roc_name
        tagged.append(ScoredEvent(
            event_type=ev.event_type,
            onset_s=ev.onset_s,
            duration_s=ev.duration_s,
            confidence=ev.confidence,
            metadata=meta,
        ))
    return tagged


def events_to_yasa_summary(events: Sequence[ScoredEvent]) -> pd.DataFrame:
    """Rebuild a minimal YASA-like summary DataFrame from ScoredEvents."""
    rows = []
    for ev in events:
        meta = ev.metadata or {}
        row = {
            "Start": ev.onset_s,
            "Duration": ev.duration_s,
            "Channel": meta.get("channel", ""),
        }
        for k, v in meta.items():
            if k == "channel":
                continue
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Start", "Duration", "Channel"]
    )


def summarize_densities(
    events: Sequence[ScoredEvent],
    scoring_df: Optional[pd.DataFrame] = None,
    epoch_seconds: float = 30.0,
) -> pd.DataFrame:
    """Per-channel / per-stage density summary (count, /min, mean duration, freq)."""
    if not events:
        return pd.DataFrame(columns=[
            "type", "channel", "stage", "count", "duration_min",
            "density_per_min", "mean_duration_s", "mean_frequency_hz",
        ])

    def _stage_at(t: float) -> str:
        if scoring_df is None or scoring_df.empty:
            return "all"
        ep = float(epoch_seconds) if epoch_seconds > 0 else 30.0
        times = pd.to_numeric(scoring_df["time"], errors="coerce")
        mask = (times <= t) & (times + ep > t)
        hits = scoring_df.loc[mask]
        if hits.empty:
            return "U"
        return StageLabel.from_string(str(hits.iloc[0]["stage"])).value

    # Duration available per stage (minutes) for density denominator
    stage_minutes: Dict[str, float] = {"all": 0.0}
    if scoring_df is not None and not scoring_df.empty:
        ep_min = float(epoch_seconds) / 60.0
        for _, row in scoring_df.iterrows():
            st = StageLabel.from_string(str(row.get("stage", "U"))).value
            stage_minutes[st] = stage_minutes.get(st, 0.0) + ep_min
            stage_minutes["all"] = stage_minutes.get("all", 0.0) + ep_min
    else:
        # Fallback: span of events
        onsets = [e.onset_s for e in events]
        ends = [e.onset_s + e.duration_s for e in events]
        span_min = max(1e-6, (max(ends) - min(onsets)) / 60.0)
        stage_minutes["all"] = span_min

    buckets: Dict[Tuple[str, str, str], List[ScoredEvent]] = {}
    for ev in events:
        ch = str((ev.metadata or {}).get("channel", "") or "unknown")
        st = _stage_at(ev.onset_s)
        key = (ev.event_type.value, ch, st)
        buckets.setdefault(key, []).append(ev)

    rows = []
    for (etype, ch, st), group in sorted(buckets.items()):
        durs = [e.duration_s for e in group]
        freqs = [
            _safe_float((e.metadata or {}).get("Frequency"), float("nan"))
            for e in group
        ]
        freqs_ok = [f for f in freqs if np.isfinite(f)]
        denom = stage_minutes.get(st, 0.0) or stage_minutes.get("all", 1e-6)
        rows.append({
            "type": etype,
            "channel": ch,
            "stage": st,
            "count": len(group),
            "duration_min": round(denom, 3),
            "density_per_min": round(len(group) / max(denom, 1e-6), 4),
            "mean_duration_s": round(float(np.mean(durs)), 4) if durs else 0.0,
            "mean_frequency_hz": (
                round(float(np.mean(freqs_ok)), 4) if freqs_ok else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def cooccurrence_summary(sw_events: Sequence[ScoredEvent]) -> pd.DataFrame:
    """Aggregate SO–spindle co-occurrence / PAC stats from slow-wave events."""
    if not sw_events:
        return pd.DataFrame(columns=[
            "channel", "n_sw", "n_cooccurring", "pct_cooccurring",
            "mean_ndPAC", "mean_phase_at_sigma_peak",
        ])
    by_ch: Dict[str, List[ScoredEvent]] = {}
    for ev in sw_events:
        if ev.event_type != EventType.SLOW_WAVE:
            continue
        ch = str((ev.metadata or {}).get("channel", "") or "unknown")
        by_ch.setdefault(ch, []).append(ev)

    rows = []
    for ch, group in sorted(by_ch.items()):
        n = len(group)
        co_flags = []
        ndpacs = []
        phases = []
        for ev in group:
            meta = ev.metadata or {}
            flag = meta.get("CooccurringSpindle")
            if flag is not None:
                co_flags.append(bool(flag))
            nd = _safe_float(meta.get("ndPAC"), float("nan"))
            if np.isfinite(nd) and nd > 0:
                ndpacs.append(nd)
            ph = _safe_float(meta.get("PhaseAtSigmaPeak"), float("nan"))
            if np.isfinite(ph):
                phases.append(ph)
        n_co = int(sum(co_flags)) if co_flags else 0
        rows.append({
            "channel": ch,
            "n_sw": n,
            "n_cooccurring": n_co,
            "pct_cooccurring": round(100.0 * n_co / n, 2) if n and co_flags else float("nan"),
            "mean_ndPAC": round(float(np.mean(ndpacs)), 4) if ndpacs else float("nan"),
            "mean_phase_at_sigma_peak": (
                round(float(np.mean(phases)), 4) if phases else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def event_type_to_annotation(event_type: EventType) -> str:
    """Map pipeline EventType to AnnotationStore ``ann_type``."""
    return _EVENT_ANN_TYPE.get(event_type, event_type.value)


def scored_events_to_dicts(events: Iterable[ScoredEvent]) -> List[Dict[str, Any]]:
    return [e.to_dict() for e in events]
