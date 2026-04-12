"""Inter-scorer agreement analysis.

Compares N scorers (CESA backends + human references) pairwise and
globally, producing Cohen's kappa matrices, Fleiss' kappa, disagreement
timelines, and focused critical-confusion breakdowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

STAGES = ["W", "N1", "N2", "N3", "R"]

# Clinically critical confusion pairs
CRITICAL_PAIRS = [
    ("N1", "W"),
    ("W", "N1"),
    ("R", "N1"),
    ("N1", "R"),
    ("N2", "N3"),
    ("N3", "N2"),
]


@dataclass
class DisagreementEvent:
    """One epoch where two scorers disagree."""

    epoch_index: int
    time_s: float
    scorer_a_stage: str
    scorer_b_stage: str


@dataclass
class CriticalConfusion:
    """Focused analysis of a specific confusion pair."""

    true_stage: str
    pred_stage: str
    count: int
    rate: float
    total_true: int


@dataclass
class InterScorerReport:
    """Full inter-scorer agreement report."""

    scorer_names: List[str]
    pairwise_kappa: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pairwise_accuracy: Dict[Tuple[str, str], float] = field(default_factory=dict)
    fleiss_kappa: Optional[float] = None
    n_epochs: int = 0
    disagreement_rate: float = 0.0
    critical_confusions: List[CriticalConfusion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorer_names": self.scorer_names,
            "pairwise_kappa": {
                f"{a}_vs_{b}": v
                for (a, b), v in self.pairwise_kappa.items()
            },
            "pairwise_accuracy": {
                f"{a}_vs_{b}": v
                for (a, b), v in self.pairwise_accuracy.items()
            },
            "fleiss_kappa": self.fleiss_kappa,
            "n_epochs": self.n_epochs,
            "disagreement_rate": self.disagreement_rate,
            "critical_confusions": [
                {"true": c.true_stage, "pred": c.pred_stage,
                 "count": c.count, "rate": c.rate, "total": c.total_true}
                for c in self.critical_confusions
            ],
        }

    def kappa_matrix_as_array(self) -> Tuple[List[str], np.ndarray]:
        """Return (names, matrix) for heatmap plotting."""
        names = self.scorer_names
        n = len(names)
        mat = np.ones((n, n))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i != j:
                    k = self.pairwise_kappa.get((a, b), self.pairwise_kappa.get((b, a), 0.0))
                    mat[i, j] = k
        return names, mat


# ---------------------------------------------------------------------
# Cohen's kappa (standalone, no sklearn dependency)
# ---------------------------------------------------------------------

def _cohens_kappa(y1: List[str], y2: List[str], labels: List[str]) -> float:
    """Cohen's kappa between two scorer label lists."""
    n = len(y1)
    if n == 0:
        return 0.0

    label_to_idx = {s: i for i, s in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(y1, y2):
        ia = label_to_idx.get(a)
        ib = label_to_idx.get(b)
        if ia is not None and ib is not None:
            cm[ia, ib] += 1

    total = cm.sum()
    if total == 0:
        return 0.0

    po = np.trace(cm) / total
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (total * total)

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _accuracy(y1: List[str], y2: List[str]) -> float:
    if not y1:
        return 0.0
    agree = sum(1 for a, b in zip(y1, y2) if a == b)
    return agree / len(y1)


# ---------------------------------------------------------------------
# Fleiss' kappa
# ---------------------------------------------------------------------

def _fleiss_kappa(scorers: Dict[str, List[str]], labels: List[str]) -> float:
    """Fleiss' kappa for 3+ raters."""
    names = list(scorers.keys())
    n_raters = len(names)
    if n_raters < 2:
        return 0.0

    n_epochs = len(next(iter(scorers.values())))
    n_categories = len(labels)
    label_to_idx = {s: i for i, s in enumerate(labels)}

    # Build rating matrix: n_epochs x n_categories
    # counts[i][j] = number of raters who assigned epoch i to category j
    counts = np.zeros((n_epochs, n_categories), dtype=np.int64)
    for name in names:
        for i, stage in enumerate(scorers[name]):
            j = label_to_idx.get(stage)
            if j is not None and i < n_epochs:
                counts[i, j] += 1

    N = n_epochs
    n = n_raters

    # Proportion per category
    p_j = counts.sum(axis=0) / (N * n)

    # Per-epoch agreement
    P_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1)) if n > 1 else np.zeros(N)

    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)

    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def compute_pairwise_kappa(
    scorers: Dict[str, List[str]],
    labels: Optional[List[str]] = None,
) -> Dict[Tuple[str, str], float]:
    """Compute Cohen's kappa for all pairs of scorers."""
    if labels is None:
        labels = STAGES

    result: Dict[Tuple[str, str], float] = {}
    names = list(scorers.keys())
    for a, b in combinations(names, 2):
        k = _cohens_kappa(scorers[a], scorers[b], labels)
        result[(a, b)] = k
    return result


def compute_pairwise_accuracy(
    scorers: Dict[str, List[str]],
) -> Dict[Tuple[str, str], float]:
    result: Dict[Tuple[str, str], float] = {}
    names = list(scorers.keys())
    for a, b in combinations(names, 2):
        result[(a, b)] = _accuracy(scorers[a], scorers[b])
    return result


def disagreement_timeline(
    scorer_a: List[str],
    scorer_b: List[str],
    epoch_duration_s: float = 30.0,
) -> List[DisagreementEvent]:
    """Return epochs where two scorers disagree."""
    events: List[DisagreementEvent] = []
    for i, (a, b) in enumerate(zip(scorer_a, scorer_b)):
        if a != b:
            events.append(DisagreementEvent(
                epoch_index=i,
                time_s=i * epoch_duration_s,
                scorer_a_stage=a,
                scorer_b_stage=b,
            ))
    return events


def critical_confusion_analysis(
    reference: List[str],
    predicted: List[str],
) -> List[CriticalConfusion]:
    """Focused analysis on clinically critical confusion pairs."""
    results: List[CriticalConfusion] = []

    for true_stage, pred_stage in CRITICAL_PAIRS:
        total_true = sum(1 for s in reference if s == true_stage)
        if total_true == 0:
            results.append(CriticalConfusion(
                true_stage=true_stage, pred_stage=pred_stage,
                count=0, rate=0.0, total_true=0,
            ))
            continue

        confused = sum(
            1 for r, p in zip(reference, predicted)
            if r == true_stage and p == pred_stage
        )
        results.append(CriticalConfusion(
            true_stage=true_stage, pred_stage=pred_stage,
            count=confused, rate=confused / total_true,
            total_true=total_true,
        ))

    return results


def compute_inter_scorer_report(
    scorers: Dict[str, List[str]],
    *,
    epoch_duration_s: float = 30.0,
    labels: Optional[List[str]] = None,
) -> InterScorerReport:
    """Compute full inter-scorer agreement report."""
    if labels is None:
        labels = STAGES

    names = list(scorers.keys())
    n_epochs = len(next(iter(scorers.values()))) if scorers else 0

    report = InterScorerReport(
        scorer_names=names,
        n_epochs=n_epochs,
    )

    report.pairwise_kappa = compute_pairwise_kappa(scorers, labels)
    report.pairwise_accuracy = compute_pairwise_accuracy(scorers)

    if len(names) >= 3:
        report.fleiss_kappa = _fleiss_kappa(scorers, labels)

    # Disagreement rate: fraction of epochs where not all scorers agree
    if n_epochs > 0 and len(names) >= 2:
        n_disagree = 0
        for i in range(n_epochs):
            stages = set()
            for name in names:
                if i < len(scorers[name]):
                    stages.add(scorers[name][i])
            if len(stages) > 1:
                n_disagree += 1
        report.disagreement_rate = n_disagree / n_epochs

    # Critical confusions (first scorer as reference, each other as predicted)
    if len(names) >= 2:
        ref_name = names[0]
        for pred_name in names[1:]:
            confusions = critical_confusion_analysis(
                scorers[ref_name], scorers[pred_name],
            )
            report.critical_confusions.extend(confusions)

    return report
