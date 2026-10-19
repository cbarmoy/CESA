"""Agreement metrics between two hypnograms.

Toolkit-independent counterpart of ``AppController.compare_scoring``: it turns
the concordance DataFrame (columns ``auto``, ``manual``) into the figures shown
by the comparison dialog. Kept free of Qt so the math can be tested and reused
without a GUI.

Convention: the manual scoring is the reference. Sensitivity (recall) answers
"how many manual epochs of this stage did the model find", precision "when the
model announces this stage, how often is it right" -- same convention as
``sleep_pipeline.evaluation.compare``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

#: Display order for stage rows and confusion-matrix axes.
STAGE_ORDER = ("W", "N1", "N2", "N3", "R", "U")


def cohens_kappa(auto: Sequence[str], manual: Sequence[str]) -> float:
    """Cohen's kappa between two categorical sequences (best effort)."""
    try:
        from sklearn.metrics import cohen_kappa_score

        return float(cohen_kappa_score(list(auto), list(manual)))
    except Exception:
        pass
    auto = list(auto)
    manual = list(manual)
    n = len(auto)
    if n == 0:
        return float("nan")
    labels = sorted(set(auto) | set(manual))
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=float)
    for a, m in zip(auto, manual):
        cm[idx[a], idx[m]] += 1.0
    po = float(np.trace(cm)) / n
    row = cm.sum(axis=1) / n
    col = cm.sum(axis=0) / n
    pe = float(np.sum(row * col))
    if abs(1.0 - pe) < 1e-12:
        return 1.0
    return (po - pe) / (1.0 - pe)


def kappa_label(kappa: float) -> str:
    """Qualitative reading of a kappa value (Landis & Koch style)."""
    if kappa != kappa:  # NaN
        return "indetermine"
    if kappa < 0.0:
        return "desaccord"
    if kappa < 0.20:
        return "tres faible"
    if kappa < 0.40:
        return "faible"
    if kappa < 0.60:
        return "modere"
    if kappa < 0.80:
        return "bon"
    return "excellent"


def stage_metrics(
    merged: pd.DataFrame,
    stages: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Per-stage confusion counts and metrics, manual scoring as reference."""
    auto = merged["auto"].to_numpy()
    manual = merged["manual"].to_numpy()
    total = len(merged)
    out: Dict[str, Dict[str, float]] = {}
    for stage in stages:
        auto_is = auto == stage
        man_is = manual == stage
        tp = int(np.sum(auto_is & man_is))
        fp = int(np.sum(auto_is & ~man_is))
        fn = int(np.sum(~auto_is & man_is))
        tn = total - tp - fp - fn
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        if recall == recall and precision == precision and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if (tp + fn + fp) else float("nan")
        out[stage] = {
            "support": tp + fn,
            "n_auto": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "f1": f1,
        }
    return out


def macro_f1(metrics: Dict[str, Dict[str, float]]) -> float:
    """Unweighted mean F1 over the stages that are actually present."""
    values = [m["f1"] for m in metrics.values() if m["f1"] == m["f1"]]
    return float(np.mean(values)) if values else float("nan")


def present_stages(merged: pd.DataFrame) -> List[str]:
    """Stages appearing on either side, in canonical display order."""
    seen = set(merged["auto"]) | set(merged["manual"])
    return [s for s in STAGE_ORDER if s in seen]


def confusion_matrix(merged: pd.DataFrame, stages: Sequence[str]) -> np.ndarray:
    """Counts with automatic stages as rows and manual stages as columns."""
    idx = {s: i for i, s in enumerate(stages)}
    cm = np.zeros((len(stages), len(stages)), dtype=int)
    for a, m in zip(merged["auto"], merged["manual"]):
        if a in idx and m in idx:
            cm[idx[a], idx[m]] += 1
    return cm
