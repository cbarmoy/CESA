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

    An empty *channels* list means "every channel detected as EEG".
    """

    output_dir: str
    channels: List[str] = field(default_factory=list)
    epoch_seconds: float = 30.0
    include_scoring: bool = True
    min_coupled: Optional[int] = None
    export_events: bool = False
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


def resolve_scoring_for_file(
    recording_path: str,
    *,
    epoch_seconds: float = 30.0,
    recording_duration_s: float = 0.0,
) -> Optional[pd.DataFrame]:
    """Locate and load a sidecar scoring file for *recording_path*.

    Searches, in order, for a file sharing the recording base name with a
    ``.csv``/``.xlsx``/``.xls``/``.json`` extension, then for an EDF+
    hypnogram (``*Hypnogram*.edf``) in the same directory.

    Returns a normalized ``['time', 'stage']`` DataFrame (seconds), or ``None``
    when no scoring could be resolved.
    """
    from CESA import scoring_io

    base = os.path.splitext(recording_path)[0]

    for ext in _SCORING_SIDE_EXTENSIONS:
        candidate = base + ext
        if not os.path.exists(candidate):
            continue
        try:
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
                epoch_seconds=epoch_seconds,
                recording_duration_s=recording_duration_s,
            )
        except Exception as exc:  # pragma: no cover - defensive I/O
            logger.warning("Scoring sidecar %s unreadable: %s", candidate, exc)

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

    return None


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


def _coupling_channels(raw, requested: Sequence[str]) -> List[str]:
    """Resolve the EEG channels to analyse for one recording."""
    from CESA.filters import detect_signal_type

    available = list(raw.ch_names)
    if requested:
        picked = [c for c in requested if c in available]
        if not picked:
            raise ValueError(
                "aucun canal demande n'est present (" + ", ".join(requested) + ")"
            )
        return picked
    picked = [c for c in available if detect_signal_type(c) == "eeg"]
    if not picked:
        raise ValueError("aucun canal EEG detecte")
    return picked


def _coupling_rows_for_file(
    recording_path: str,
    config: BatchCouplingConfig,
    *,
    group: str = "",
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Compute per-channel coupling rows for one recording.

    Returns ``(rows, warnings)``; the events CSV is written as a side effect
    when :pyattr:`BatchCouplingConfig.export_events` is set.
    """
    from CESA.sleep_pipeline.micro_events import scoring_df_to_hypno
    from CESA.sleep_pipeline.so_spindle_coupling import (
        MIN_COUPLED_EVENTS,
        detect_so_spindle_coupling,
    )

    file_name = os.path.basename(recording_path)
    base_name = os.path.splitext(file_name)[0]
    raw = open_raw_file(recording_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    duration_s = float(raw.n_times) / sfreq

    channels = _coupling_channels(raw, config.channels)

    scoring_df = None
    if config.include_scoring:
        scoring_df = resolve_scoring_for_file(
            recording_path,
            epoch_seconds=config.epoch_seconds,
            recording_duration_s=duration_s,
        )

    signals = []
    for channel in channels:
        extracted = _channel_signal(raw, channel)
        if extracted is None:
            raise ValueError(f"canal '{channel}' illisible")
        signals.append(extracted[0])
    data = np.vstack(signals)

    hypno = None
    if scoring_df is not None and not scoring_df.empty:
        hypno = scoring_df_to_hypno(
            scoring_df, data.shape[1], sfreq, config.epoch_seconds,
        )

    min_coupled = (
        MIN_COUPLED_EVENTS if config.min_coupled is None else int(config.min_coupled)
    )
    coupling = detect_so_spindle_coupling(
        data, sfreq, channels, hypno=hypno, min_coupled=min_coupled,
    )

    if config.export_events and coupling.events:
        events_path = os.path.join(
            config.output_dir, f"{base_name}_coupling_events.csv"
        )
        export_coupling_events_to_csv(coupling.events, events_path, file_name)

    rows: List[Dict[str, object]] = []
    for record in coupling.stats.to_dict(orient="records"):
        row: Dict[str, object] = {
            "subject": base_name,
            "file": file_name,
            "group": group,
            "scoring_used": hypno is not None,
            "duration_min": round(duration_s / 60.0, 2),
        }
        row.update(record)
        rows.append(row)

    warnings = [f"{file_name}: {w}" for w in coupling.warnings]
    if scoring_df is None and config.include_scoring:
        warnings.append(
            f"{file_name}: aucun scoring resolu, analyse sans restriction de stade"
        )
    return rows, warnings


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

    for idx, file_path in enumerate(files):
        if should_stop is not None and should_stop():
            result.stopped = True
            break
        file_name = os.path.basename(file_path)
        if progress_cb is not None:
            progress_cb(idx / total * 100.0, f"Couplage SO-fuseau: {file_name}")
        try:
            file_rows, warnings = _coupling_rows_for_file(
                file_path, config, group=(groups or {}).get(file_path, ""),
            )
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"{file_name}: {exc}")
            continue
        rows.extend(file_rows)
        result.warnings.extend(warnings)
        result.processed += 1

    if rows:
        out_path = os.path.join(config.output_dir, config.output_filename)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
        result.outputs.append(out_path)

    if progress_cb is not None:
        progress_cb(100.0, "Termine" if not result.stopped else "Arrete")
    return result
