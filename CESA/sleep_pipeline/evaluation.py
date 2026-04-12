"""Evaluation metrics for sleep scoring.

Provides:
* ``compare``        -- epoch-by-epoch accuracy, kappa, F1, confusion matrix.
* ``compute_clinical_metrics`` -- TST, SE, SOL, WASO, REM latency.
* ``error_analysis`` -- focused confusion pairs, transition accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import CLINICAL_STAGE_STRINGS, ScoringResult, StageLabel


# ---------------------------------------------------------------------------
# Report data-class
# ---------------------------------------------------------------------------

@dataclass
class StageMetrics:
    """Per-stage precision / recall / F1 / specificity."""

    stage: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    support: int = 0  # true count


@dataclass
class ComparisonReport:
    """Full comparison between two scoring runs."""

    n_epochs: int = 0
    accuracy: float = 0.0
    cohen_kappa: float = 0.0
    macro_f1: float = 0.0
    confusion_matrix: Optional[pd.DataFrame] = None
    per_stage: Dict[str, StageMetrics] = field(default_factory=dict)
    labels: List[str] = field(default_factory=lambda: list(CLINICAL_STAGE_STRINGS))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_text(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Epochs compared : {self.n_epochs}",
            f"Accuracy        : {self.accuracy:.1%}",
            f"Cohen's kappa   : {self.cohen_kappa:.3f}",
            f"Macro-F1        : {self.macro_f1:.3f}",
            "",
            "Per-stage:",
            f"  {'Stage':<6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Spec':>6} {'TP':>5} {'FP':>5} {'FN':>5}",
        ]
        for s in self.labels:
            m = self.per_stage.get(s)
            if m is None:
                continue
            lines.append(
                f"  {s:<6} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} "
                f"{m.specificity:>6.3f} {m.tp:>5} {m.fp:>5} {m.fn:>5}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON export."""
        return {
            "n_epochs": self.n_epochs,
            "accuracy": self.accuracy,
            "cohen_kappa": self.cohen_kappa,
            "macro_f1": self.macro_f1,
            "per_stage": {
                s: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "specificity": m.specificity,
                    "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
                    "support": m.support,
                }
                for s, m in self.per_stage.items()
            },
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Cohen's kappa (standalone -- no sklearn dependency required)
# ---------------------------------------------------------------------------

def _cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    """Cohen's kappa from integer-encoded labels."""
    n = len(y_true)
    if n == 0:
        return 0.0
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    label_idx = {lab: i for i, lab in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        ti = label_idx.get(int(t))
        pi = label_idx.get(int(p))
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    total = cm.sum()
    if total == 0:
        return 0.0
    po = np.trace(cm) / total
    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (total ** 2)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    return float((po - pe) / (1.0 - pe))


# ---------------------------------------------------------------------------
# Public comparison API
# ---------------------------------------------------------------------------

def compare(
    reference: ScoringResult,
    predicted: ScoringResult,
    *,
    exclude_u: bool = True,
    epoch_tolerance_s: float = 5.0,
) -> ComparisonReport:
    """Compare two scoring results epoch-by-epoch.

    Parameters
    ----------
    reference : ScoringResult
        Ground-truth (typically manual scoring).
    predicted : ScoringResult
        System output (auto-scoring).
    exclude_u : bool
        If *True*, epochs labelled U in the reference are excluded.
    epoch_tolerance_s : float
        Maximum time-offset (in seconds) to consider two epochs as matching.

    Returns
    -------
    ComparisonReport
    """
    labels = CLINICAL_STAGE_STRINGS  # ["W","N1","N2","N3","R"]
    label_to_int = {s: i for i, s in enumerate(labels)}

    # Build lookup: reference time -> stage
    ref_lookup: Dict[float, StageLabel] = {}
    for ep in reference.epochs:
        if exclude_u and ep.stage == StageLabel.U:
            continue
        key = round(ep.start_s, 1)
        ref_lookup[key] = ep.stage

    y_true_int: List[int] = []
    y_pred_int: List[int] = []
    y_true_str: List[str] = []
    y_pred_str: List[str] = []

    for ep in predicted.epochs:
        # Find matching reference epoch
        key = round(ep.start_s, 1)
        ref_stage = ref_lookup.pop(key, None)
        if ref_stage is None:
            # Try with tolerance
            for k in list(ref_lookup.keys()):
                if abs(k - key) <= epoch_tolerance_s:
                    ref_stage = ref_lookup.pop(k)
                    break
        if ref_stage is None:
            continue
        if ref_stage.value not in label_to_int or ep.stage.value not in label_to_int:
            continue
        y_true_str.append(ref_stage.value)
        y_pred_str.append(ep.stage.value)
        y_true_int.append(label_to_int[ref_stage.value])
        y_pred_int.append(label_to_int[ep.stage.value])

    n = len(y_true_int)
    if n == 0:
        return ComparisonReport(metadata={"warning": "no_matching_epochs"})

    y_t = np.array(y_true_int)
    y_p = np.array(y_pred_int)

    accuracy = float(np.mean(y_t == y_p))
    kappa = _cohen_kappa(y_t, y_p, list(range(len(labels))))

    # Confusion matrix
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_t, y_p):
        cm[t, p] += 1
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Per-stage metrics
    per_stage: Dict[str, StageMetrics] = {}
    f1_list: List[float] = []
    for i, stage in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(n - tp - fp - fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        per_stage[stage] = StageMetrics(
            stage=stage, precision=prec, recall=rec, f1=f1,
            specificity=spec, tp=tp, fp=fp, fn=fn, tn=tn,
            support=int(cm[i, :].sum()),
        )

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    report = ComparisonReport(
        n_epochs=n,
        accuracy=accuracy,
        cohen_kappa=kappa,
        macro_f1=macro_f1,
        confusion_matrix=cm_df,
        per_stage=per_stage,
        labels=list(labels),
    )

    # Attach clinical metrics and error analysis if enough data
    if n > 0:
        report.metadata["clinical_ref"] = compute_clinical_metrics(reference).to_dict()
        report.metadata["clinical_pred"] = compute_clinical_metrics(predicted).to_dict()
        ea = error_analysis(
            [StageLabel.from_string(s) for s in y_true_str],
            [StageLabel.from_string(s) for s in y_pred_str],
        )
        report.metadata["error_analysis"] = ea.to_dict()

    return report


# ---------------------------------------------------------------------------
# Clinical sleep metrics
# ---------------------------------------------------------------------------

@dataclass
class ClinicalMetrics:
    """Standard clinical sleep metrics (AASM-derived)."""

    total_sleep_time_min: float = 0.0
    time_in_bed_min: float = 0.0
    sleep_efficiency_pct: float = 0.0
    sleep_onset_latency_min: float = 0.0
    rem_latency_min: float = 0.0
    waso_min: float = 0.0
    n1_pct: float = 0.0
    n2_pct: float = 0.0
    n3_pct: float = 0.0
    rem_pct: float = 0.0
    wake_pct: float = 0.0
    n_epochs: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_sleep_time_min": self.total_sleep_time_min,
            "time_in_bed_min": self.time_in_bed_min,
            "sleep_efficiency_pct": self.sleep_efficiency_pct,
            "sleep_onset_latency_min": self.sleep_onset_latency_min,
            "rem_latency_min": self.rem_latency_min,
            "waso_min": self.waso_min,
            "n1_pct": self.n1_pct,
            "n2_pct": self.n2_pct,
            "n3_pct": self.n3_pct,
            "rem_pct": self.rem_pct,
            "wake_pct": self.wake_pct,
        }


def compute_clinical_metrics(result: ScoringResult) -> ClinicalMetrics:
    """Derive standard clinical sleep metrics from a ScoringResult.

    Parameters
    ----------
    result : ScoringResult

    Returns
    -------
    ClinicalMetrics
    """
    if not result.epochs:
        return ClinicalMetrics()

    epoch_s = result.epoch_duration_s
    stages = [ep.stage for ep in result.epochs]
    n = len(stages)

    tib_min = n * epoch_s / 60.0

    # Sleep = N1 + N2 + N3 + R
    sleep_mask = [s.is_sleep for s in stages]
    tst_min = sum(sleep_mask) * epoch_s / 60.0

    se_pct = (tst_min / tib_min * 100.0) if tib_min > 0 else 0.0

    # Sleep onset latency: first epoch to first sleep epoch
    sol_min = 0.0
    for i, s in enumerate(stages):
        if s.is_sleep:
            sol_min = i * epoch_s / 60.0
            break
    else:
        sol_min = tib_min  # never fell asleep

    # REM latency: from sleep onset to first REM
    sleep_onset_idx = next((i for i, s in enumerate(stages) if s.is_sleep), 0)
    rem_lat_min = 0.0
    for i in range(sleep_onset_idx, n):
        if stages[i] == StageLabel.R:
            rem_lat_min = (i - sleep_onset_idx) * epoch_s / 60.0
            break
    else:
        rem_lat_min = tib_min

    # WASO: Wake epochs after sleep onset and before final sleep
    last_sleep_idx = 0
    for i in range(n - 1, -1, -1):
        if stages[i].is_sleep:
            last_sleep_idx = i
            break
    waso_epochs = sum(
        1 for i in range(sleep_onset_idx, last_sleep_idx + 1)
        if stages[i] == StageLabel.W
    )
    waso_min = waso_epochs * epoch_s / 60.0

    # Stage percentages (of TST)
    tst_epochs = max(sum(sleep_mask), 1)
    n1_pct = sum(1 for s in stages if s == StageLabel.N1) / tst_epochs * 100
    n2_pct = sum(1 for s in stages if s == StageLabel.N2) / tst_epochs * 100
    n3_pct = sum(1 for s in stages if s == StageLabel.N3) / tst_epochs * 100
    rem_pct = sum(1 for s in stages if s == StageLabel.R) / tst_epochs * 100
    wake_pct = sum(1 for s in stages if s == StageLabel.W) / n * 100

    return ClinicalMetrics(
        total_sleep_time_min=tst_min,
        time_in_bed_min=tib_min,
        sleep_efficiency_pct=se_pct,
        sleep_onset_latency_min=sol_min,
        rem_latency_min=rem_lat_min,
        waso_min=waso_min,
        n1_pct=n1_pct,
        n2_pct=n2_pct,
        n3_pct=n3_pct,
        rem_pct=rem_pct,
        wake_pct=wake_pct,
        n_epochs=n,
    )


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

@dataclass
class ConfusionPair:
    """Focused analysis of a specific confusion pair."""
    true_stage: str
    pred_stage: str
    count: int = 0
    rate: float = 0.0  # fraction of true_stage epochs confused as pred_stage


@dataclass
class ErrorAnalysis:
    """Detailed error analysis for publication-level reporting."""

    confusion_pairs: List[ConfusionPair] = field(default_factory=list)
    transition_accuracy: float = 0.0
    boundary_accuracy: float = 0.0
    n_transitions_ref: int = 0
    n_transitions_pred: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confusion_pairs": [
                {"true": cp.true_stage, "pred": cp.pred_stage,
                 "count": cp.count, "rate": round(cp.rate, 4)}
                for cp in self.confusion_pairs
            ],
            "transition_accuracy": round(self.transition_accuracy, 4),
            "boundary_accuracy": round(self.boundary_accuracy, 4),
            "n_transitions_ref": self.n_transitions_ref,
            "n_transitions_pred": self.n_transitions_pred,
        }


def error_analysis(
    y_true: List[StageLabel],
    y_pred: List[StageLabel],
) -> ErrorAnalysis:
    """Perform detailed error analysis on aligned stage sequences.

    Analyses:
    - Key confusion pairs (N1/W, REM/N1, N2/N3).
    - Transition accuracy: how often a stage *change* in the reference
      is also detected as a change in the prediction.
    - Boundary accuracy: agreement in the +/-1 epoch around transitions.
    """
    n = len(y_true)
    if n == 0:
        return ErrorAnalysis()

    # Focused confusion pairs
    _PAIRS = [("N1", "W"), ("W", "N1"), ("R", "N1"), ("N1", "R"), ("N2", "N3"), ("N3", "N2")]
    pair_results = []
    for true_s, pred_s in _PAIRS:
        true_sl = StageLabel.from_string(true_s)
        pred_sl = StageLabel.from_string(pred_s)
        true_count = sum(1 for s in y_true if s == true_sl)
        confused = sum(
            1 for t, p in zip(y_true, y_pred)
            if t == true_sl and p == pred_sl
        )
        rate = confused / max(true_count, 1)
        pair_results.append(ConfusionPair(true_s, pred_s, confused, rate))

    # Transition accuracy
    ref_transitions = []
    pred_transitions = []
    for i in range(1, n):
        if y_true[i] != y_true[i - 1]:
            ref_transitions.append(i)
        if y_pred[i] != y_pred[i - 1]:
            pred_transitions.append(i)

    # A reference transition at epoch i is "detected" if the prediction
    # also has a transition within +/-1 epoch
    detected = 0
    for rt in ref_transitions:
        for pt in pred_transitions:
            if abs(rt - pt) <= 1:
                detected += 1
                break
    trans_acc = detected / max(len(ref_transitions), 1)

    # Boundary accuracy: agreement at epochs within +/-1 of reference transitions
    boundary_indices = set()
    for rt in ref_transitions:
        for offset in (-1, 0, 1):
            idx = rt + offset
            if 0 <= idx < n:
                boundary_indices.add(idx)
    if boundary_indices:
        boundary_correct = sum(1 for i in boundary_indices if y_true[i] == y_pred[i])
        boundary_acc = boundary_correct / len(boundary_indices)
    else:
        boundary_acc = 1.0

    return ErrorAnalysis(
        confusion_pairs=pair_results,
        transition_accuracy=trans_acc,
        boundary_accuracy=boundary_acc,
        n_transitions_ref=len(ref_transitions),
        n_transitions_pred=len(pred_transitions),
    )
