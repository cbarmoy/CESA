# -*- coding: utf-8 -*-
"""CESA - Shared batch analysis engine (FFT/PSD and PSG metrics).

This module decouples the batch-processing logic that historically lived inside
the Tk studio (``EEGAnalysisStudio`` in :mod:`CESA.eeg_studio_fixed`) so it can
be driven from the Qt viewer (or any other UI / script) without a GUI toolkit.

It reuses the numerical helpers from :mod:`CESA.spectral_analysis` and
:mod:`CESA.sleep_metrics`; no spectral maths are duplicated here.

Public API:
    - :class:`BatchFFTConfig` / :func:`run_batch_fft`
    - :class:`BatchPSGConfig` / :func:`run_batch_psg`
    - :class:`BatchCouplingConfig` / :func:`run_batch_so_spindle_coupling`
    - :func:`resolve_scoring_for_file`
    - :func:`compute_channel_psd_by_stage` / :func:`export_psd_results_to_csv`

Auteur: Come Barmoy (Unite Neuropsychologie du Stress - IRBA)
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.raw_loader import open_raw_file
from CESA.spectral_analysis import (
    compute_stage_psd_fft_for_array,
    compute_stage_psd_welch_for_array,
)

logger = logging.getLogger(__name__)

# Flushed on every line so a hard crash still leaves the last checkpoint.
_BATCH_LOG = Path("logs") / "batch_analysis.log"


def _checkpoint(
    progress_cb: Optional[ProgressCallback],
    percent: float,
    message: str,
) -> None:
    """Surface one batch step in the UI, the logger and ``logs/batch_analysis.log``."""
    logger.info("[BATCH] %s", message)
    try:
        _BATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _BATCH_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")
            handle.flush()
    except Exception:
        logger.debug("Could not write %s", _BATCH_LOG, exc_info=True)
    if progress_cb is not None:
        progress_cb(min(100.0, max(0.0, percent)), message)

# Progress callback: (percent in [0, 100], human-readable message) -> None
ProgressCallback = Callable[[float, str], None]
# Cooperative cancellation: returns True when the caller requests a stop.
StopCallback = Callable[[], bool]

DEFAULT_STAGES: Tuple[str, ...] = ("W", "N1", "N2", "N3", "R")
PSDResults = Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BatchFFTConfig:
    """Parameters for the FFT/PSD-by-stage batch pipeline."""

    output_dir: str
    method: str = "fft"  # "fft" (magnitude) or "welch" (uV^2/Hz)
    fmin: float = 0.5
    fmax: float = 45.0
    epoch_len: float = 30.0
    nperseg_sec: float = 4.0
    equalize_epochs: bool = True
    robust_stats: bool = True
    include_scoring: bool = True
    band_filter: Optional[Tuple[float, float]] = (0.3, 40.0)
    notch_hz: Optional[float] = 50.0
    stages: List[str] = field(default_factory=lambda: list(DEFAULT_STAGES))


@dataclass
class BatchPSGConfig:
    """Parameters for the multi-subject PSG metrics batch pipeline."""

    output_dir: str
    epoch_seconds: float = 30.0
    output_filename: str = "psg_summary_batch.csv"


@dataclass
class BatchCouplingConfig:
    """Parameters for the SO-spindle coupling batch pipeline.

    An empty *channels* list means "every channel detected as EEG", or every
    frontal derivation when *headband* is set.
    """

    output_dir: str
    channels: List[str] = field(default_factory=list)
    epoch_seconds: float = 30.0
    include_scoring: bool = True
    min_coupled: Optional[int] = None
    export_events: bool = False
    export_figures: bool = False
    headband: bool = False
    output_filename: str = "so_spindle_coupling_batch.csv"


@dataclass
class BatchResult:
    """Summary returned by the batch runners."""

    processed: int = 0
    failed: int = 0
    outputs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stopped: bool = False


# ---------------------------------------------------------------------------
# Scoring resolution
# ---------------------------------------------------------------------------

_SCORING_SIDE_EXTENSIONS = (".csv", ".xlsx", ".xls", ".json")


def _scoring_stem(path: str) -> str:
    """Lower-case recording stem used to match ``S030_AP_sleep scoring.xls``."""
    return os.path.splitext(os.path.basename(path))[0].lower().replace(" ", "")


def find_scoring_sidecars(recording_path: str) -> List[str]:
    """Scoring files next to *recording_path*, best match first.

    Accepts the exact stem (``S030_Ap.xls``) and the clinical export pattern
    ``S030_AP_sleep scoring.xls``: same stem, any case, then a suffix.
    """
    directory = os.path.dirname(os.path.abspath(recording_path))
    stem = _scoring_stem(recording_path)
    if not stem or not os.path.isdir(directory):
        return []
    ranked: List[Tuple[int, str]] = []
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    for name in names:
        ext = os.path.splitext(name)[1].lower()
        if ext not in _SCORING_SIDE_EXTENSIONS:
            continue
        if name.lower().endswith("_coupling_events.csv"):
            continue
        body = _scoring_stem(name)
        if body == stem:
            rank = 0
        elif body.startswith(stem + "_") or body.startswith(stem + "-"):
            rank = 1
        else:
            continue
        if any(token in body for token in ("sleep", "scoring", "score", "hypno", "sommeil")):
            rank -= 1
        ranked.append((rank, os.path.join(directory, name)))
    ranked.sort(key=lambda item: (item[0], item[1].lower()))
    return [path for _rank, path in ranked]


def _load_scoring_table(
    candidate: str,
    *,
    epoch_seconds: float,
    recording_duration_s: float,
    absolute_start_datetime: Optional[object] = None,
) -> pd.DataFrame:
    from CESA import scoring_io

    ext = os.path.splitext(candidate)[1].lower()
    if ext == ".json":
        return scoring_io.import_scoring_json(
            candidate,
            epoch_seconds=epoch_seconds,
            recording_duration_s=recording_duration_s,
        )
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(candidate)
    else:
        df = pd.read_csv(candidate)
    return scoring_io.import_scoring_table(
        df,
        absolute_start_datetime=absolute_start_datetime,
        epoch_seconds=epoch_seconds,
        recording_duration_s=recording_duration_s,
    )


def resolve_scoring_for_file(
    recording_path: str,
    *,
    epoch_seconds: float = 30.0,
    recording_duration_s: float = 0.0,
    absolute_start_datetime: Optional[object] = None,
) -> Optional[pd.DataFrame]:
    """Locate and load a sidecar scoring file for *recording_path*.

    Searches the recording directory for a table whose name starts with the
    EDF stem (``S030_Ap.edf`` matches ``S030_AP_sleep scoring.xls``), then for
    an EDF+ hypnogram (``*Hypnogram*.edf``), then for ``Sleep stage``
    annotations inside the recording itself.

    Returns a normalized ``['time', 'stage']`` DataFrame (seconds), or ``None``
    when no scoring could be resolved.
    """
    from CESA import scoring_io

    for candidate in find_scoring_sidecars(recording_path):
        try:
            table = _load_scoring_table(
                candidate,
                epoch_seconds=epoch_seconds,
                recording_duration_s=recording_duration_s,
                absolute_start_datetime=absolute_start_datetime,
            )
        except Exception as exc:  # pragma: no cover - defensive I/O
            logger.warning("Scoring sidecar %s unreadable: %s", candidate, exc)
            continue
        if table is not None and not table.empty:
            logger.info("[SCORING] %s -> %s", os.path.basename(recording_path), candidate)
            return table

    # EDF+ hypnogram in the same folder
    try:
        import glob

        directory = os.path.dirname(recording_path)
        for hyp in glob.glob(os.path.join(directory, "*Hypnogram*.edf")):
            try:
                return scoring_io.import_edf_hypnogram(
                    hyp,
                    recording_duration_s=recording_duration_s,
                    epoch_seconds=epoch_seconds,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("EDF hypnogram %s unreadable: %s", hyp, exc)
    except Exception:  # pragma: no cover
        pass

    # Stages are often stored in the recording itself (EDF+ "Sleep stage" annotations).
    try:
        embedded = scoring_io.import_edf_hypnogram(
            recording_path,
            recording_duration_s=recording_duration_s,
            epoch_seconds=epoch_seconds,
        )
    except Exception as exc:  # pragma: no cover - not every EDF carries annotations
        logger.info("Pas d'hypnogramme dans %s: %s", os.path.basename(recording_path), exc)
        return None
    if embedded is None or embedded.empty:
        return None
    if not embedded["stage"].astype(str).isin(["W", "N1", "N2", "N3", "R"]).any():
        return None
    return embedded


# ---------------------------------------------------------------------------
# Single-channel PSD-by-stage (core numeric, no GUI)
# ---------------------------------------------------------------------------

def compute_channel_psd_by_stage(
    signal: np.ndarray,
    fs: float,
    scoring_df: Optional[pd.DataFrame],
    config: BatchFFTConfig,
) -> PSDResults:
    """Compute PSD-by-stage for a single channel using the configured method."""
    stages = list(config.stages)
    if config.method == "welch":
        return compute_stage_psd_welch_for_array(
            signal=signal,
            fs=fs,
            scoring_df=scoring_df,
            epoch_len=config.epoch_len,
            stages=stages,
            fmin=config.fmin,
            fmax=config.fmax,
            nperseg_sec=config.nperseg_sec,
            band_filter=config.band_filter,
            notch_hz=config.notch_hz,
            equalize_epochs=config.equalize_epochs,
            robust_stats=config.robust_stats,
        )
    return compute_stage_psd_fft_for_array(
        signal=signal,
        fs=fs,
        scoring_df=scoring_df,
        epoch_len=config.epoch_len,
        stages=stages,
        fmin=config.fmin,
        fmax=config.fmax,
        equalize_epochs=config.equalize_epochs,
        robust_stats=config.robust_stats,
    )


def export_psd_results_to_csv(results: PSDResults, output_path: str) -> int:
    """Write PSD-by-stage results to CSV.

    Columns: ``stage, n_epochs, freq_hz, magnitude, sem``.
    Returns the number of data rows written.
    """
    total_rows = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "n_epochs", "freq_hz", "magnitude", "sem"])
        for stage, payload in results.items():
            if not payload or len(payload) != 4:
                continue
            freqs, mean_vals, sem_vals, n_epochs = payload
            if len(freqs) != len(mean_vals) or len(freqs) != len(sem_vals):
                logger.warning("Inconsistent PSD lengths for stage %s, skipped", stage)
                continue
            for i in range(len(freqs)):
                writer.writerow([
                    stage,
                    int(n_epochs),
                    float(freqs[i]),
                    float(mean_vals[i]),
                    float(sem_vals[i]),
                ])
                total_rows += 1
    return total_rows


def _channel_signal(raw, channel_name: str) -> Optional[Tuple[np.ndarray, float]]:
    """Extract a 1D signal (in uV) and sampling frequency for *channel_name*."""
    if channel_name not in raw.ch_names:
        return None
    idx = raw.ch_names.index(channel_name)
    data = raw.get_data(picks=[idx])[0]
    # MNE stores EEG in volts; convert to microvolts for consistency with the
    # spectral helpers which expect uV.
    signal = np.asarray(data, dtype=float) * 1e6
    fs = float(raw.info["sfreq"])
    return signal, fs


# ---------------------------------------------------------------------------
# Batch runners
# ---------------------------------------------------------------------------

def run_batch_fft(
    files: Sequence[str],
    channels: Sequence[str],
    config: BatchFFTConfig,
    *,
    progress_cb: Optional[ProgressCallback] = None,
    should_stop: Optional[StopCallback] = None,
) -> BatchResult:
    """Run FFT/PSD-by-stage over ``files`` x ``channels`` and export one CSV each.

    Output naming: ``{recording_basename}_{channel}_FFT.csv`` in
    ``config.output_dir``.
    """
    result = BatchResult()
    os.makedirs(config.output_dir, exist_ok=True)

    files = list(files)
    channels = list(channels)
    total_ops = max(1, len(files) * len(channels))
    op = 0

    def _emit(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(min(100.0, op / total_ops * 100.0), msg)

    for file_path in files:
        if should_stop is not None and should_stop():
            result.stopped = True
            break
        file_name = os.path.basename(file_path)
        try:
            _emit(f"Chargement: {file_name}")
            raw = open_raw_file(file_path, preload=True, verbose="ERROR")
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"{file_name}: chargement echoue ({exc})")
            op += len(channels)
            continue

        duration_s = float(raw.n_times) / float(raw.info["sfreq"])
        scoring_df = None
        if config.include_scoring:
            scoring_df = resolve_scoring_for_file(
                file_path,
                epoch_seconds=config.epoch_len,
                recording_duration_s=duration_s,
            )

        base_name = os.path.splitext(file_name)[0]
        for channel in channels:
            if should_stop is not None and should_stop():
                result.stopped = True
                break
            op += 1
            _emit(f"FFT: {file_name} - {channel}")
            extracted = _channel_signal(raw, channel)
            if extracted is None:
                result.failed += 1
                result.errors.append(f"{file_name}: canal '{channel}' introuvable")
                continue
            signal, fs = extracted
            try:
                psd = compute_channel_psd_by_stage(signal, fs, scoring_df, config)
                if not psd:
                    result.errors.append(
                        f"{file_name}/{channel}: aucun resultat (scoring/segments insuffisants)"
                    )
                    result.failed += 1
                    continue
                safe_channel = channel.replace("-", "_").replace("/", "_")
                out_path = os.path.join(
                    config.output_dir, f"{base_name}_{safe_channel}_FFT.csv"
                )
                export_psd_results_to_csv(psd, out_path)
                result.outputs.append(out_path)
                result.processed += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{file_name}/{channel}: {exc}")
        if result.stopped:
            break

    if progress_cb is not None:
        progress_cb(100.0, "Termine" if not result.stopped else "Arrete")
    return result


def _psg_row_for_file(
    recording_path: str,
    config: BatchPSGConfig,
) -> Dict[str, object]:
    """Compute one PSG metrics row for a single recording."""
    from CESA.sleep_metrics import compute_psg_summary

    file_name = os.path.basename(recording_path)
    raw = open_raw_file(recording_path, preload=False, verbose="ERROR")
    duration_s = float(raw.n_times) / float(raw.info["sfreq"])
    scoring_df = resolve_scoring_for_file(
        recording_path,
        epoch_seconds=config.epoch_seconds,
        recording_duration_s=duration_s,
    )
    if scoring_df is None or len(scoring_df) == 0:
        raise ValueError("aucun scoring resolu")

    summary = compute_psg_summary(scoring_df, epoch_seconds=config.epoch_seconds)
    row: Dict[str, object] = {
        "subject": os.path.splitext(file_name)[0],
        "file": file_name,
    }
    row.update(summary.clinical.to_dict())
    row["awakenings"] = summary.awakenings
    for s in DEFAULT_STAGES:
        row[f"{s}_duration_min"] = round(summary.stage_durations_min.get(s, 0.0), 2)
    return row


def run_batch_psg(
    files: Sequence[str],
    config: BatchPSGConfig,
    *,
    progress_cb: Optional[ProgressCallback] = None,
    should_stop: Optional[StopCallback] = None,
) -> BatchResult:
    """Compute PSG metrics for each recording and aggregate into one CSV."""
    result = BatchResult()
    os.makedirs(config.output_dir, exist_ok=True)

    files = list(files)
    total = max(1, len(files))
    rows: List[Dict[str, object]] = []

    for idx, file_path in enumerate(files):
        if should_stop is not None and should_stop():
            result.stopped = True
            break
        file_name = os.path.basename(file_path)
        if progress_cb is not None:
            progress_cb(idx / total * 100.0, f"PSG: {file_name}")
        try:
            rows.append(_psg_row_for_file(file_path, config))
            result.processed += 1
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"{file_name}: {exc}")

    if rows:
        out_path = os.path.join(config.output_dir, config.output_filename)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
        result.outputs.append(out_path)

    if progress_cb is not None:
        progress_cb(100.0, "Termine" if not result.stopped else "Arrete")
    return result


# ---------------------------------------------------------------------------
# SO-spindle coupling (classical detection + circular statistics)
# ---------------------------------------------------------------------------

COUPLING_EVENT_COLUMNS: Tuple[str, ...] = (
    "file", "channel", "onset_s", "duration_s", "frequency_hz",
    "peak_s", "coupled", "so_phase_rad", "so_phase_deg", "coupled_so_onset_s",
)


def export_coupling_events_to_csv(
    events: Sequence[object],
    output_path: str,
    file_name: str = "",
) -> int:
    """Write the detected spindles and their SO phase to CSV.

    Returns the number of data rows written.
    """
    from CESA.sleep_pipeline.contracts import EventType

    rows = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(COUPLING_EVENT_COLUMNS))
        for event in events:
            if getattr(event, "event_type", None) != EventType.SPINDLE:
                continue
            meta = getattr(event, "metadata", None) or {}
            phase = meta.get("so_phase_rad")
            writer.writerow([
                file_name,
                meta.get("channel", ""),
                round(float(event.onset_s), 3),
                round(float(event.duration_s), 3),
                meta.get("Frequency", ""),
                meta.get("peak_s", ""),
                phase is not None,
                "" if phase is None else phase,
                "" if phase is None else meta.get("so_phase_deg", ""),
                meta.get("coupled_so_onset_s", ""),
            ])
            rows += 1
    return rows


def _coupling_channels(
    raw, requested: Sequence[str], *, headband: bool = False,
) -> Tuple[List[str], List[str]]:
    """Resolve EEG channels and the non-EEG names that were left out.

    EOG, ECG and heart-rate traces are never analysed, even when they appear
    in *requested*: a slow-oscillation detector on those signals is not a
    cortical coupling. With *headband*, the montage is further reduced to its
    frontal derivations.
    """
    from CESA.filters import detect_signal_type

    available = list(raw.ch_names)
    if headband:
        from CESA.sleep_pipeline.headband_coupling import select_headband_channels

        return select_headband_channels(available, requested)

    names = [c for c in requested if c in available] if requested else available
    if requested and not names:
        raise ValueError(
            "aucun canal demande n'est present (" + ", ".join(requested) + ")"
        )
    picked = [c for c in names if detect_signal_type(c) in ("eeg", "sas_eeg")]
    skipped = [c for c in names if c not in picked]
    if not picked:
        raise ValueError(
            "aucun canal EEG parmi " + ", ".join(names or requested)
        )
    return picked, skipped


def _summarize_names(names: Sequence[str], limit: int = 8) -> str:
    """Comma-separated *names*, truncated: a PSG montage has 70+ channels."""
    listed = list(names)
    if len(listed) <= limit:
        return ", ".join(listed)
    return ", ".join(listed[:limit]) + f", ... (+{len(listed) - limit})"


def _coupling_rows_for_file(
    recording_path: str,
    config: BatchCouplingConfig,
    *,
    group: str = "",
    on_step: Optional[Callable[[float, str], None]] = None,
    should_stop: Optional[StopCallback] = None,
) -> Tuple[List[Dict[str, object]], List[str], bool]:
    """Compute per-channel coupling rows for one recording.

    Returns ``(rows, warnings, stopped)``. Channels are filtered one at a time
    so a long recording does not keep every filtered copy in memory, and
    *on_step* is called before each slow step (``fraction`` in ``[0, 1]``
    within this file).
    """
    from CESA.sleep_pipeline.contracts import EventType
    from CESA.sleep_pipeline.micro_events import scoring_df_to_hypno
    from CESA.sleep_pipeline.so_spindle_coupling import (
        HEADBAND_STATS_COLUMNS,
        MIN_COUPLED_EVENTS,
        STATS_COLUMNS,
        analyze_channel,
    )

    def step(fraction: float, message: str) -> None:
        if on_step is not None:
            on_step(fraction, message)

    file_name = os.path.basename(recording_path)
    base_name = os.path.splitext(file_name)[0]
    warnings: List[str] = []
    step(0.02, f"{file_name} : ouverture de l'EDF (le chargement peut prendre une minute)")
    raw = open_raw_file(recording_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n_samples = int(raw.n_times)
    duration_s = n_samples / sfreq
    step(
        0.08,
        f"{file_name} : charge, {duration_s / 60.0:.1f} min a {sfreq:.0f} Hz "
        f"({n_samples} echantillons)",
    )

    channels, skipped = _coupling_channels(
        raw, config.channels, headband=config.headband,
    )
    kind = "frontal(aux)" if config.headband else "EEG"
    step(0.10, f"{file_name} : {len(channels)} canal(aux) {kind} — {', '.join(channels)}")
    if skipped:
        reason = (
            "pas frontaux (montage bandeau)" if config.headband
            else "pas de l'EEG cortical"
        )
        skipped_msg = (
            f"{file_name}: {len(skipped)} canaux ignores ({reason}) — "
            f"{_summarize_names(skipped)}"
        )
        warnings.append(skipped_msg)
        step(0.11, skipped_msg)

    scoring_df = None
    if config.include_scoring:
        step(0.12, f"{file_name} : recherche du scoring")
        info = getattr(raw, "info", {}) or {}
        scoring_df = resolve_scoring_for_file(
            recording_path,
            epoch_seconds=config.epoch_seconds,
            recording_duration_s=duration_s,
            absolute_start_datetime=info.get("meas_date") if hasattr(info, "get") else None,
        )

    hypno = None
    if scoring_df is not None and not scoring_df.empty:
        hypno = scoring_df_to_hypno(
            scoring_df, n_samples, sfreq, config.epoch_seconds,
        )
        step(0.16, f"{file_name} : scoring utilise ({len(scoring_df)} epoques)")
    elif config.include_scoring:
        step(0.16, f"{file_name} : pas de scoring, analyse sans restriction de stade")

    min_coupled = (
        MIN_COUPLED_EVENTS if config.min_coupled is None else int(config.min_coupled)
    )
    so_mask = None
    spindle_mask = None
    if hypno is not None:
        from CESA.sleep_pipeline.so_spindle_coupling import SO_STAGES, SPINDLE_STAGES, stage_mask
        so_mask = stage_mask(hypno, SO_STAGES, n_samples)
        spindle_mask = stage_mask(hypno, SPINDLE_STAGES, n_samples)

    rows: List[Dict[str, object]] = []
    events: List[object] = []
    n_channels = max(1, len(channels))
    stopped = False
    for index, channel in enumerate(channels):
        if should_stop is not None and should_stop():
            stopped = True
            break
        fraction = 0.18 + 0.75 * index / n_channels
        step(
            fraction,
            f"{file_name} : canal {index + 1}/{len(channels)} {channel} "
            f"— filtre SO, fuseaux, phase",
        )
        extracted = _channel_signal(raw, channel)
        if extracted is None:
            raise ValueError(f"canal '{channel}' illisible")
        signal, channel_sf = extracted
        try:
            channel_events, record = analyze_channel(
                signal, channel_sf, channel,
                so_mask=so_mask,
                spindle_mask=spindle_mask,
                min_coupled=min_coupled,
                reject_blinks=config.headband,
            )
        finally:
            del signal
        events.extend(channel_events)
        if record["n_coupled"] < int(min_coupled):
            warnings.append(
                f"{file_name}: {channel}: seulement {record['n_coupled']} "
                f"fuseau(x) couple(s) (minimum {int(min_coupled)}); "
                f"phase preferee non publiee."
            )
        row: Dict[str, object] = {
            "subject": base_name,
            "file": file_name,
            "group": group,
            "scoring_used": hypno is not None,
            "duration_min": round(duration_s / 60.0, 2),
        }
        row.update(record)
        rows.append(row)
        blink_note = (
            f", {record.get('n_blinks', 0)} clignements masques "
            f"({record.get('blink_rejected_s', 0.0)} s)"
            if config.headband else ""
        )
        step(
            0.18 + 0.75 * (index + 1) / n_channels,
            f"{file_name} : {channel} termine — "
            f"{record['n_so']} SO, {record['n_spindle']} fuseaux, "
            f"{record['n_coupled']} couples{blink_note}",
        )

    if scoring_df is None and config.include_scoring:
        warnings.append(
            f"{file_name}: aucun scoring resolu, analyse sans restriction de stade"
        )

    if config.export_events and events and not stopped:
        events_path = os.path.join(
            config.output_dir, f"{base_name}_coupling_events.csv"
        )
        spindles = [e for e in events if getattr(e, "event_type", None) == EventType.SPINDLE]
        step(0.96, f"{file_name} : ecriture de {len(spindles)} fuseaux")
        export_coupling_events_to_csv(events, events_path, file_name)

    # Keep the public column order even when a channel produced no stats keys.
    columns = HEADBAND_STATS_COLUMNS if config.headband else STATS_COLUMNS
    for row in rows:
        for column in columns:
            row.setdefault(column, float("nan"))
    return rows, warnings, stopped


def run_batch_so_spindle_coupling(
    files: Sequence[str],
    config: BatchCouplingConfig,
    *,
    groups: Optional[Dict[str, str]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    should_stop: Optional[StopCallback] = None,
) -> BatchResult:
    """Compute the SO-spindle preferred phase for each recording x channel.

    Aggregates every channel of every file into one CSV, so the cohort can be
    compared afterwards.  *groups* optionally maps a file path to a design
    label (e.g. ``AVANT`` / ``APRES``) written to the ``group`` column.
    """
    result = BatchResult()
    os.makedirs(config.output_dir, exist_ok=True)

    files = list(files)
    total = max(1, len(files))
    rows: List[Dict[str, object]] = []
    label = "Couplage SO-fuseau bandeau" if config.headband else "Couplage SO-fuseau"
    _checkpoint(progress_cb, 0.0, f"{label} : {len(files)} fichier(s)")

    for idx, file_path in enumerate(files):
        if should_stop is not None and should_stop():
            result.stopped = True
            break
        file_name = os.path.basename(file_path)

        def _on_step(fraction: float, message: str, _idx: int = idx) -> None:
            _checkpoint(progress_cb, (_idx + fraction) / total * 100.0, message)

        try:
            file_rows, warnings, stopped = _coupling_rows_for_file(
                file_path,
                config,
                group=(groups or {}).get(file_path, ""),
                on_step=_on_step,
                should_stop=should_stop,
            )
        except MemoryError:
            result.failed += 1
            message = (
                f"{file_name}: memoire insuffisante. Relancez avec moins de canaux."
            )
            result.errors.append(message)
            _checkpoint(progress_cb, (idx + 1) / total * 100.0, message)
            continue
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"{file_name}: {exc}")
            _checkpoint(
                progress_cb, (idx + 1) / total * 100.0, f"{file_name} : echec ({exc})",
            )
            logger.error("Coupling batch failed on %s", file_name, exc_info=True)
            continue
        rows.extend(file_rows)
        result.warnings.extend(warnings)
        if stopped:
            result.stopped = True
            break
        result.processed += 1
        _checkpoint(
            progress_cb,
            (idx + 1) / total * 100.0,
            f"{file_name} : fichier termine ({len(file_rows)} canal/canaux)",
        )

    if rows:
        filename = config.output_filename
        if config.headband and filename == BatchCouplingConfig.output_filename:
            # Never overwrite the PSG table with a frontal, blink-masked one.
            filename = "so_spindle_coupling_headband.csv"
        out_path = os.path.join(config.output_dir, filename)
        table = pd.DataFrame(rows)
        table.to_csv(out_path, index=False, encoding="utf-8")
        result.outputs.append(out_path)
        _checkpoint(progress_cb, 100.0, f"CSV ecrit : {out_path}")

        if config.export_figures:
            _checkpoint(progress_cb, 100.0, "Figures de phase preferee")
            prefix = "coupling_headband" if config.headband else "coupling"
            try:
                from CESA.coupling_plots import save_coupling_figures

                for figure in save_coupling_figures(
                    table, config.output_dir, prefix=prefix,
                ):
                    result.outputs.append(figure)
                    _checkpoint(progress_cb, 100.0, f"Figure ecrite : {figure}")
            except Exception as exc:
                message = f"figures non generees: {exc}"
                result.warnings.append(message)
                _checkpoint(progress_cb, 100.0, message)
                logger.error("Coupling figures failed", exc_info=True)

    _checkpoint(progress_cb, 100.0, "Termine" if not result.stopped else "Arrete")
    return result
