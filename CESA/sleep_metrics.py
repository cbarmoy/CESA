"""Clinical PSG summary metrics shared between Qt and Tk UIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from CESA.sleep_pipeline.contracts import Epoch, ScoringResult, StageLabel
from CESA.sleep_pipeline.evaluation import ClinicalMetrics, compute_clinical_metrics


@dataclass
class StageEpisode:
  start_min: float
  end_min: float
  stage: str
  duration_min: float


@dataclass
class PsgSummary:
  clinical: ClinicalMetrics
  stage_durations_min: Dict[str, float] = field(default_factory=dict)
  stage_pct_of_tst: Dict[str, float] = field(default_factory=dict)
  episode_counts: Dict[str, int] = field(default_factory=dict)
  episode_mean_duration_min: Dict[str, float] = field(default_factory=dict)
  episodes: List[StageEpisode] = field(default_factory=list)
  night_start_s: Optional[float] = None
  night_end_s: Optional[float] = None
  awakenings: int = 0
  annotations: List[Dict[str, Any]] = field(default_factory=list)


def _df_to_scoring_result(df: pd.DataFrame, epoch_seconds: float) -> ScoringResult:
  stages = df["stage"].astype(str).str.upper().tolist()
  epochs = [
    Epoch(index=i, start_s=float(t), stage=StageLabel.from_string(s), confidence=1.0)
    for i, (t, s) in enumerate(zip(df["time"].values, stages))
  ]
  return ScoringResult(epochs=epochs, epoch_duration_s=float(epoch_seconds))


def _filter_window(
  df: pd.DataFrame,
  epoch_seconds: float,
  night_start_s: Optional[float],
  night_end_s: Optional[float],
) -> pd.DataFrame:
  if night_start_s is None or night_end_s is None:
    return df.copy()
  times = pd.to_numeric(df["time"], errors="coerce")
  ep = float(epoch_seconds)
  mask = (times >= float(night_start_s) - 1e-6) & (times + ep <= float(night_end_s) + 1e-6)
  return df.loc[mask].copy().reset_index(drop=True)


def _build_episodes(
  stages: np.ndarray,
  times: np.ndarray,
  epoch_len: float,
) -> List[StageEpisode]:
  episodes: List[StageEpisode] = []
  if len(stages) == 0:
    return episodes
  start_idx = 0
  for i in range(1, len(stages)):
    if stages[i] != stages[i - 1]:
      a = float(times[start_idx]) / 60.0
      b = float(times[i]) / 60.0
      dur = b - a
      episodes.append(StageEpisode(start_min=a, end_min=b, stage=str(stages[start_idx]), duration_min=dur))
      start_idx = i
  a = float(times[start_idx]) / 60.0
  b = float((times[-1] + epoch_len)) / 60.0
  dur = b - a
  episodes.append(StageEpisode(start_min=a, end_min=b, stage=str(stages[start_idx]), duration_min=dur))
  return episodes


def _apply_night_window_tib(
  clinical: ClinicalMetrics,
  *,
  stages: np.ndarray,
  times: np.ndarray,
  epoch_len: float,
  night_start_s: float,
  night_end_s: float,
) -> ClinicalMetrics:
  """Override TIB/SE/SOL using the wall-clock analysis window (lights off → on)."""
  tib_min = max((float(night_end_s) - float(night_start_s)) / 60.0, 0.0)
  clinical.time_in_bed_min = tib_min
  clinical.sleep_efficiency_pct = (
    clinical.total_sleep_time_min / tib_min * 100.0 if tib_min > 0 else 0.0
  )

  sleep_mask = np.isin(stages, ["N1", "N2", "N3", "R"])
  if sleep_mask.any():
    first_sleep_t = float(times[np.where(sleep_mask)[0][0]])
    clinical.sleep_onset_latency_min = max(
      0.0, (first_sleep_t - float(night_start_s)) / 60.0,
    )
    sleep_onset_idx = int(np.where(sleep_mask)[0][0])
    rem_idxs = np.where(stages == "R")[0]
    rem_after = rem_idxs[rem_idxs >= sleep_onset_idx]
    if len(rem_after):
      clinical.rem_latency_min = (
        float(times[rem_after[0]]) - first_sleep_t
      ) / 60.0
    else:
      clinical.rem_latency_min = tib_min
  else:
    clinical.sleep_onset_latency_min = tib_min
    clinical.rem_latency_min = tib_min

  clinical.wake_pct = (
    float(np.sum(stages == "W")) * epoch_len / 60.0 / tib_min * 100.0
    if tib_min > 0 else 0.0
  )
  return clinical


def compute_psg_summary(
  df: pd.DataFrame,
  epoch_seconds: float,
  night_start_s: Optional[float] = None,
  night_end_s: Optional[float] = None,
) -> PsgSummary:
  """Compute PSG metrics and contiguous stage episodes for a scoring DataFrame."""
  work = _filter_window(df, epoch_seconds, night_start_s, night_end_s)
  if work is None or len(work) == 0:
    return PsgSummary(
      clinical=ClinicalMetrics(),
      night_start_s=night_start_s,
      night_end_s=night_end_s,
    )

  epoch_len = float(epoch_seconds)
  stages = work["stage"].astype(str).str.upper().values
  times = pd.to_numeric(work["time"], errors="coerce").values

  clinical = compute_clinical_metrics(_df_to_scoring_result(work, epoch_len))

  if night_start_s is not None and night_end_s is not None:
    clinical = _apply_night_window_tib(
      clinical,
      stages=stages,
      times=times,
      epoch_len=epoch_len,
      night_start_s=float(night_start_s),
      night_end_s=float(night_end_s),
    )
  else:
    # Without an explicit window, TIB spans first to last epoch in the scoring.
    if len(times):
      span_min = (float(times[-1]) + epoch_len - float(times[0])) / 60.0
      if span_min > clinical.total_sleep_time_min + 1e-6:
        clinical.time_in_bed_min = span_min
        clinical.sleep_efficiency_pct = (
          clinical.total_sleep_time_min / span_min * 100.0 if span_min > 0 else 0.0
        )

  stage_durations_min: Dict[str, float] = {}
  for s in ("W", "N1", "N2", "N3", "R", "U"):
    stage_durations_min[s] = float(np.sum(stages == s) * epoch_len / 60.0)

  tst_epochs = max(int(np.sum(np.isin(stages, ["N1", "N2", "N3", "R"]))), 1)
  stage_pct_of_tst: Dict[str, float] = {}
  for s in ("N1", "N2", "N3", "R"):
    stage_pct_of_tst[s] = float(np.sum(stages == s) / tst_epochs * 100.0)
  tib_min = max(clinical.time_in_bed_min, 1e-6)
  stage_pct_of_tst["W"] = float(stage_durations_min.get("W", 0.0) / tib_min * 100.0)

  episodes = _build_episodes(stages, times, epoch_len)
  episode_counts: Dict[str, int] = {}
  episode_durations: Dict[str, List[float]] = {}
  for ep in episodes:
    episode_counts[ep.stage] = episode_counts.get(ep.stage, 0) + 1
    episode_durations.setdefault(ep.stage, []).append(ep.duration_min)

  episode_mean_duration_min = {
    s: float(np.mean(durs)) if durs else 0.0
    for s, durs in episode_durations.items()
  }

  awakenings = 0
  for i in range(1, len(stages)):
    if stages[i] == "W" and stages[i - 1] != "W":
      awakenings += 1

  return PsgSummary(
    clinical=clinical,
    stage_durations_min=stage_durations_min,
    stage_pct_of_tst=stage_pct_of_tst,
    episode_counts=episode_counts,
    episode_mean_duration_min=episode_mean_duration_min,
    episodes=episodes,
    night_start_s=night_start_s,
    night_end_s=night_end_s,
    awakenings=awakenings,
  )


def export_psg_summary_csv(
  summary: PsgSummary,
  path: str,
  annotations: Optional[List[Dict[str, Any]]] = None,
) -> None:
  """Write PSG summary to a structured CSV file."""
  import csv

  cm = summary.clinical
  ann_list = annotations if annotations is not None else summary.annotations
  with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    if summary.night_start_s is not None and summary.night_end_s is not None:
      writer.writerow(["night_start_min", f"{summary.night_start_s / 60.0:.2f}"])
      writer.writerow(["night_end_min", f"{summary.night_end_s / 60.0:.2f}"])
    writer.writerow(["TIB_min", f"{cm.time_in_bed_min:.2f}"])
    writer.writerow(["TST_min", f"{cm.total_sleep_time_min:.2f}"])
    writer.writerow(["SOL_min", f"{cm.sleep_onset_latency_min:.2f}"])
    writer.writerow(["WASO_min", f"{cm.waso_min:.2f}"])
    writer.writerow(["SE_percent", f"{cm.sleep_efficiency_pct:.2f}"])
    writer.writerow(["REM_latency_min", f"{cm.rem_latency_min:.2f}"])
    writer.writerow(["N1_pct_TST", f"{cm.n1_pct:.2f}"])
    writer.writerow(["N2_pct_TST", f"{cm.n2_pct:.2f}"])
    writer.writerow(["N3_pct_TST", f"{cm.n3_pct:.2f}"])
    writer.writerow(["REM_pct_TST", f"{cm.rem_pct:.2f}"])
    writer.writerow(["Wake_pct", f"{cm.wake_pct:.2f}"])
    writer.writerow(["Awakenings", summary.awakenings])
    writer.writerow([])
    writer.writerow(["stage", "duration_min", "pct_TST", "episode_count", "mean_episode_min"])
    for s in ("W", "N1", "N2", "N3", "R"):
      writer.writerow([
        s,
        f"{summary.stage_durations_min.get(s, 0.0):.2f}",
        f"{summary.stage_pct_of_tst.get(s, 0.0):.2f}",
        summary.episode_counts.get(s, 0),
        f"{summary.episode_mean_duration_min.get(s, 0.0):.2f}",
      ])
    writer.writerow([])
    writer.writerow(["start_min", "end_min", "stage", "duration_min"])
    for ep in summary.episodes:
      writer.writerow([
        f"{ep.start_min:.2f}",
        f"{ep.end_min:.2f}",
        ep.stage,
        f"{ep.duration_min:.2f}",
      ])
    if ann_list:
      writer.writerow([])
      writer.writerow([
        "id", "onset_min", "duration_min", "end_min", "type", "label", "channel", "source",
      ])
      for ann in ann_list:
        onset_s = float(ann.get("onset", ann.get("onset_s", 0.0)))
        dur_s = float(ann.get("duration", ann.get("duration_s", 0.0)))
        writer.writerow([
          str(ann.get("id", "")),
          f"{onset_s / 60.0:.2f}",
          f"{dur_s / 60.0:.2f}",
          f"{(onset_s + dur_s) / 60.0:.2f}",
          str(ann.get("type", ann.get("ann_type", ""))),
          str(ann.get("label", ann.get("text", ""))),
          str(ann.get("channel", "")),
          str(ann.get("source", "manual")),
        ])
