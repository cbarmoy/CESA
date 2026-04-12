"""Structured paper report generator.

Produces a Markdown document following the typical structure of an
IEEE TBME / clinical sleep science paper, with sections auto-populated
from :class:`ExperimentResult`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .experiment import ExperimentResult

logger = logging.getLogger(__name__)


def generate_paper_report(result: "ExperimentResult", output_dir: Path) -> Path:
    """Write ``paper_draft.md`` into *output_dir* and return its path."""
    sections = [
        _title_section(result),
        _abstract_section(result),
        _introduction_section(result),
        _methods_section(result),
        _results_section(result),
        _discussion_section(result),
        _conclusion_section(result),
        _references_section(),
    ]

    text = "\n\n---\n\n".join(s for s in sections if s)

    p = output_dir / "paper_draft.md"
    p.write_text(text, encoding="utf-8")
    logger.info("Paper draft written to %s", p)
    return p


# -----------------------------------------------------------------
# Section generators
# -----------------------------------------------------------------

def _title_section(r: "ExperimentResult") -> str:
    ds = r.config.dataset.replace("_", "-").upper()
    return (
        f"# CESA: A Hybrid Rule-Based and Machine-Learning Sleep Staging System\n"
        f"# Validated on {ds}\n\n"
        f"*Auto-generated draft — edit before submission*"
    )


def _abstract_section(r: "ExperimentResult") -> str:
    lines = ["## Abstract\n"]

    # Best backend
    best_backend = ""
    best_kappa = -1.0
    for bk, agg in r.per_backend.items():
        if agg.kappa_mean > best_kappa:
            best_kappa = agg.kappa_mean
            best_backend = bk

    n_sub = r.n_subjects
    n_folds = r.config.n_folds
    lines.append(
        f"**Background:** Automatic sleep staging is essential for scalable "
        f"sleep medicine. We present CESA, a hybrid system combining "
        f"AASM-compliant rules, gradient-boosted machine learning, and "
        f"Hidden Markov Model temporal smoothing.\n"
    )
    lines.append(
        f"**Methods:** We evaluated {len(r.config.backends)} backends "
        f"({', '.join(r.config.backends)}) on the {r.config.dataset.replace('_', '-')} "
        f"dataset ({n_sub} subjects) using {n_folds}-fold subject-level "
        f"cross-validation.\n"
    )

    if best_backend:
        agg = r.per_backend[best_backend]
        lines.append(
            f"**Results:** The best-performing backend ({best_backend}) achieved "
            f"a Cohen's kappa of {agg.kappa_mean:.3f} ± {agg.kappa_std:.3f}, "
            f"accuracy of {agg.accuracy_mean:.1%} ± {agg.accuracy_std:.1%}, "
            f"and macro-F1 of {agg.macro_f1_mean:.3f} ± {agg.macro_f1_std:.3f}.\n"
        )

    lines.append(
        "**Conclusion:** CESA provides an interpretable, modular, and "
        "competitive alternative to black-box deep-learning approaches, "
        "with full traceability of scoring decisions."
    )
    return "\n".join(lines)


def _introduction_section(r: "ExperimentResult") -> str:
    return (
        "## 1. Introduction\n\n"
        "Sleep staging is the classification of polysomnographic (PSG) "
        "recordings into discrete sleep stages (Wake, N1, N2, N3, REM) "
        "according to the American Academy of Sleep Medicine (AASM) manual. "
        "Manual scoring by trained technologists is the gold standard but is "
        "time-consuming, subjective (inter-scorer kappa ≈ 0.76–0.82), and "
        "limits large-scale studies.\n\n"
        "Recent deep-learning approaches (U-Sleep, SleepTransformer) achieve "
        "high accuracy but lack interpretability — a critical requirement in "
        "clinical diagnostics. Rule-based systems are interpretable but rigid. "
        "Machine-learning systems with handcrafted features offer a middle "
        "ground but typically ignore temporal context.\n\n"
        "CESA bridges these approaches by combining:\n"
        "1. **Explicit AASM rules** for transparent baseline scoring\n"
        "2. **Gradient-boosted ML** with 30+ physiological features "
        "(spindles, K-complexes, spectral entropy, EMG/EOG indicators)\n"
        "3. **HMM temporal modelling** with biologically plausible "
        "transition constraints\n\n"
        "This paper evaluates CESA's performance on public benchmarks using "
        "a rigorous subject-level cross-validation protocol."
    )


def _methods_section(r: "ExperimentResult") -> str:
    cfg = r.config
    lines = [
        "## 2. Methods\n",
        "### 2.1 Dataset\n",
        f"We used the **{cfg.dataset.replace('_', '-')}** dataset "
        f"({r.n_subjects} subjects, {r.n_records} recordings). "
        f"Signals were resampled to {cfg.target_sfreq} Hz and segmented "
        f"into {cfg.epoch_duration_s}-second epochs.\n",
        "### 2.2 Preprocessing\n",
        "Standard preprocessing was applied: "
        "bandpass filtering (EEG: 0.3–35 Hz, EOG: 0.3–10 Hz, EMG: 10–100 Hz), "
        "artefact rejection, and z-score normalisation per channel per recording.\n",
        "### 2.3 Feature Extraction\n",
        "We extracted 30+ features per epoch including: "
        "absolute and relative band powers (delta, theta, alpha, sigma, beta), "
        "spectral entropy, zero-crossing rate, spindle count and density, "
        "K-complex detection, slow and rapid eye movement indices, "
        "tonic EMG level, and temporal context features (neighbouring epochs, "
        "cumulative stage duration, night fraction).\n",
        "### 2.4 Scoring Backends\n",
    ]

    for bk in cfg.backends:
        if bk == "aasm_rules":
            lines.append(
                "- **AASM Rules**: Deterministic rule-based scoring implementing "
                "AASM v2.6 criteria with explicit thresholds for each stage.\n"
            )
        elif bk == "ml":
            lines.append(
                f"- **ML ({cfg.ml_model_type.upper()})**: "
                f"{'HistGradientBoosting' if cfg.ml_model_type == 'hgb' else 'Random Forest'} "
                "classifier trained on extracted features with class-weight "
                "balancing and temporal context (±2 epochs).\n"
            )
        elif bk == "ml_hmm":
            lines.append(
                "- **ML + HMM**: ML predictions used as emission probabilities "
                "for a 5-state Hidden Markov Model with biologically constrained "
                "transition matrix. Viterbi decoding ensures physiologically "
                "plausible stage sequences.\n"
            )

    lines.extend([
        "### 2.5 Validation Protocol\n",
        f"We employed **{cfg.n_folds}-fold subject-level cross-validation** "
        f"(GroupKFold) ensuring no data leakage between train and test sets. "
        f"All recordings from the same subject remain in the same fold.\n",
        "### 2.6 Evaluation Metrics\n",
        "Primary: **Cohen's kappa** (agreement beyond chance). "
        "Secondary: overall accuracy, macro-F1 score. "
        "Clinical metrics: Total Sleep Time (TST), Sleep Efficiency (SE), "
        "Sleep Onset Latency (SOL), WASO, REM latency. "
        f"Bootstrap confidence intervals ({cfg.bootstrap_ci:.0%}) were computed "
        f"over {cfg.bootstrap_n} resamples.\n",
    ])

    return "\n".join(lines)


def _results_section(r: "ExperimentResult") -> str:
    lines = ["## 3. Results\n", "### 3.1 Overall Performance\n"]

    # Table header
    lines.append("| Backend | Accuracy | Cohen's κ | Macro-F1 |")
    lines.append("|---------|----------|-----------|----------|")

    for bk, agg in r.per_backend.items():
        lines.append(
            f"| {bk} | {agg.accuracy_mean:.3f} ± {agg.accuracy_std:.3f} "
            f"| {agg.kappa_mean:.3f} ± {agg.kappa_std:.3f} "
            f"| {agg.macro_f1_mean:.3f} ± {agg.macro_f1_std:.3f} |"
        )

    lines.append("\n*Table 1: Overall performance across k-fold CV.*\n")

    # Per-stage performance
    lines.append("### 3.2 Per-Stage Performance\n")
    for bk, agg in r.per_backend.items():
        if not agg.per_stage_mean:
            continue
        lines.append(f"\n**{bk}**\n")
        lines.append("| Stage | Precision | Recall | F1 |")
        lines.append("|-------|-----------|--------|-----|")
        for stage, metrics in sorted(agg.per_stage_mean.items()):
            lines.append(
                f"| {stage} "
                f"| {metrics.get('precision', 0):.3f} "
                f"| {metrics.get('recall', 0):.3f} "
                f"| {metrics.get('f1', 0):.3f} |"
            )

    # Clinical metrics
    lines.append("\n### 3.3 Clinical Metrics\n")
    lines.append("| Metric | " + " | ".join(r.per_backend.keys()) + " |")
    lines.append("|--------| " + " | ".join("---" for _ in r.per_backend) + " |")

    metric_names = [
        ("total_sleep_time_min", "TST (min)"),
        ("sleep_efficiency_pct", "SE (%)"),
        ("sleep_onset_latency_min", "SOL (min)"),
        ("waso_min", "WASO (min)"),
        ("rem_latency_min", "REM Lat (min)"),
    ]
    for key, name in metric_names:
        row = f"| {name}"
        for agg in r.per_backend.values():
            m = agg.clinical_metrics_mean.get(key, 0)
            s = agg.clinical_metrics_std.get(key, 0)
            row += f" | {m:.1f} ± {s:.1f}"
        row += " |"
        lines.append(row)

    # Confidence intervals
    lines.append("\n### 3.4 Confidence Intervals\n")
    for bk, agg in r.per_backend.items():
        lo, hi = agg.kappa_ci
        lines.append(f"- **{bk}**: κ = {agg.kappa_mean:.3f} "
                      f"[{lo:.3f}, {hi:.3f}]")

    # Figure references
    lines.append("\n### 3.5 Figures\n")
    lines.append("- Figure 1: Cross-validation boxplot (see `figures/cv_boxplot`)")
    lines.append("- Figure 2: Confusion matrices (see `figures/confusion_*`)")
    lines.append("- Figure 3: Feature importance (see `figures/feature_importance`)")
    lines.append("- Figure 4: Inter-scorer kappa heatmap (see `figures/kappa_heatmap`)")
    lines.append("- Figure 5: Clinical metrics table (see `figures/clinical_metrics`)")

    return "\n".join(lines)


def _discussion_section(r: "ExperimentResult") -> str:
    lines = ["## 4. Discussion\n"]

    # Identify best/worst
    best_bk, worst_bk = "", ""
    best_k, worst_k = -1.0, 2.0
    for bk, agg in r.per_backend.items():
        if agg.kappa_mean > best_k:
            best_k = agg.kappa_mean
            best_bk = bk
        if agg.kappa_mean < worst_k:
            worst_k = agg.kappa_mean
            worst_bk = bk

    if best_bk:
        lines.append(
            f"The {best_bk} backend achieved the highest agreement with "
            f"reference scoring (κ = {best_k:.3f}), "
        )
        if "hmm" in best_bk:
            lines.append(
                "suggesting that temporal modelling via HMM effectively "
                "constrains biologically implausible transitions."
            )
        else:
            lines.append(
                "indicating that the feature set captures relevant "
                "physiological information."
            )

    # Per-stage analysis
    lines.append("\n### Stage-Specific Analysis\n")
    for bk, agg in r.per_backend.items():
        for stage, m in agg.per_stage_mean.items():
            f1 = m.get("f1", 0)
            if f1 < 0.5:
                lines.append(
                    f"- **{bk}**: Low F1 for {stage} ({f1:.3f}) — "
                    "this is consistent with known inter-scorer variability "
                    "for this stage."
                )

    # Comparison to SOTA
    lines.append("\n### Comparison to State of the Art\n")
    lines.append(
        "Human inter-scorer agreement is typically κ ≈ 0.76–0.82 "
        "(Danker-Hopfe et al., 2009). "
        "Recent DL approaches report κ ≈ 0.80–0.85 on Sleep-EDF "
        "(Perslev et al., 2021; Eldele et al., 2021). "
    )
    if best_k >= 0.75:
        lines.append(
            f"CESA's best result (κ = {best_k:.3f}) is within the range "
            "of human agreement, demonstrating clinical relevance."
        )
    elif best_k >= 0.60:
        lines.append(
            f"CESA's performance (κ = {best_k:.3f}) is moderate and "
            "could benefit from additional feature engineering or "
            "integration of deep-learning components."
        )

    lines.append("\n### Limitations\n")
    lines.append(
        "- Single-dataset evaluation (generalisation to other datasets pending)\n"
        "- No comparison to multi-channel deep learning architectures\n"
        "- Feature extraction assumes standard 10-20 montage\n"
        "- HMM transition matrix is currently fixed, not learned per-subject"
    )

    return "\n".join(lines)


def _conclusion_section(r: "ExperimentResult") -> str:
    return (
        "## 5. Conclusion\n\n"
        "CESA provides a transparent, modular, and competitive sleep "
        "staging system that combines the interpretability of AASM rules "
        "with the performance of machine learning and the temporal "
        "coherence of Hidden Markov Models. "
        "The system is fully open-source, reproducible, and designed "
        "for both clinical and research applications."
    )


def _references_section() -> str:
    return (
        "## References\n\n"
        "1. Berry, R.B. et al. (2017). *The AASM Manual for the Scoring of "
        "Sleep and Associated Events*, Version 2.4.\n"
        "2. Danker-Hopfe, H. et al. (2009). Interrater reliability for a new "
        "sleep scoring standard. *J. Sleep Res.*, 18(1), 74–84.\n"
        "3. Perslev, M. et al. (2021). U-Sleep: resilient high-frequency "
        "sleep staging. *npj Digital Medicine*, 4(1), 72.\n"
        "4. Eldele, E. et al. (2021). An attention-based deep learning "
        "approach for sleep stage classification. *IEEE TBME*, 68(5).\n"
        "5. Kemp, B. et al. (2000). Analysis of a sleep-dependent neuronal "
        "feedback loop: the sleep-EDF database. *IEEE Trans. Biomed. Eng.*, "
        "47(9), 1185–1194.\n"
        "6. Supratak, A. et al. (2017). DeepSleepNet: a model for automatic "
        "sleep stage scoring. *IEEE TNSRE*, 25(11).\n"
        "7. Phan, H. et al. (2022). SleepTransformer: automatic sleep staging "
        "with interpretability. *Biomed. Signal Process. Control*, 79.\n"
    )
