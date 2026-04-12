"""Publication-ready figure generation for sleep scoring benchmarks.

All functions save figures as PDF (vector) and PNG (300 DPI) and return
the matplotlib ``Figure`` object for further customisation if needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

STAGE_ORDER = ["W", "N1", "N2", "N3", "R"]
STAGE_COLORS = {
    "W": "#EBA0AC", "N1": "#89B4FA", "N2": "#74C7EC",
    "N3": "#89DCEB", "R": "#CBA6F7", "U": "#6C7086",
}

# IEEE TBME compatible defaults
_RC_PARAMS = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _save(fig, path: Path, fmt: str = "pdf", dpi: int = 300) -> None:
    """Save figure in the requested format + PNG."""
    fig.savefig(str(path.with_suffix(f".{fmt}")), format=fmt, dpi=dpi)
    if fmt != "png":
        fig.savefig(str(path.with_suffix(".png")), format="png", dpi=dpi)


def _apply_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_RC_PARAMS)


# -----------------------------------------------------------------
# 1. Hypnogram comparison
# -----------------------------------------------------------------

def plot_hypnogram_comparison(
    reference: List[str],
    predictions: Dict[str, List[str]],
    epoch_duration_s: float = 30.0,
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Aligned hypnograms: reference on top, each backend below.

    Disagreement epochs are marked with red ticks.
    """
    _apply_style()
    import matplotlib.pyplot as plt

    n_panels = 1 + len(predictions)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 1.5 * n_panels),
                             sharex=True, gridspec_kw={"hspace": 0.35})
    if n_panels == 1:
        axes = [axes]

    stage_y = {"W": 4, "N1": 3, "N2": 2, "N3": 1, "R": 0, "U": -1}
    times_h = np.arange(len(reference)) * epoch_duration_s / 3600.0

    def _draw(ax, labels, title):
        ys = [stage_y.get(s, -1) for s in labels]
        colors = [STAGE_COLORS.get(s, "#6C7086") for s in labels]
        for i in range(len(labels)):
            ax.barh(ys[i], epoch_duration_s / 3600.0, left=times_h[i],
                    height=0.8, color=colors[i], edgecolor="none")
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["REM", "N3", "N2", "N1", "W"])
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlim(0, times_h[-1] + epoch_duration_s / 3600.0 if len(times_h) else 1)

    _draw(axes[0], reference, "Reference (human)")

    for idx, (name, pred) in enumerate(predictions.items(), 1):
        _draw(axes[idx], pred[:len(reference)], name)
        # Mark disagreements
        for i in range(min(len(reference), len(pred))):
            if reference[i] != pred[i]:
                axes[idx].axvline(times_h[i], color="#F38BA8", alpha=0.4, linewidth=0.3)

    axes[-1].set_xlabel("Time (hours)")
    fig.suptitle("Hypnogram Comparison", fontsize=12, y=1.01)

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 2. Confusion matrix heatmap
# -----------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    *,
    normalize: bool = True,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Normalised confusion matrix heatmap with count annotations."""
    _apply_style()
    import matplotlib.pyplot as plt

    if labels is None:
        labels = STAGE_ORDER[:cm.shape[0]]

    cm_display = cm.copy().astype(float)
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_display / row_sums
    else:
        cm_norm = cm_display

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = cm_norm[i, j]
            text_color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0%})", ha="center", va="center",
                    fontsize=7, color=text_color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 3. Stage distribution comparison
# -----------------------------------------------------------------

def plot_stage_distribution(
    reference: List[str],
    predictions: Dict[str, List[str]],
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Grouped bar chart comparing stage distributions."""
    _apply_style()
    import matplotlib.pyplot as plt

    all_scorers = {"Reference": reference, **predictions}
    n_scorers = len(all_scorers)
    n_stages = len(STAGE_ORDER)

    bar_width = 0.8 / n_scorers
    x = np.arange(n_stages)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    for idx, (name, labels) in enumerate(all_scorers.items()):
        total = len(labels) or 1
        pcts = [sum(1 for s in labels if s == st) / total * 100 for st in STAGE_ORDER]
        offset = (idx - n_scorers / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, pcts, bar_width, label=name, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Sleep Stage Distribution")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 4. ML probability curves
# -----------------------------------------------------------------

def plot_ml_probabilities(
    probabilities: np.ndarray,
    reference: List[str],
    epoch_duration_s: float = 30.0,
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Per-epoch class probabilities overlaid on hypnogram.

    *probabilities* shape: ``(n_epochs, 5)`` for W/N1/N2/N3/R.
    """
    _apply_style()
    import matplotlib.pyplot as plt

    n_epochs = probabilities.shape[0]
    times_h = np.arange(n_epochs) * epoch_duration_s / 3600.0

    fig, (ax_hyp, ax_prob) = plt.subplots(2, 1, figsize=(10, 4),
                                           sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    # Hypnogram
    stage_y = {"W": 4, "N1": 3, "N2": 2, "N3": 1, "R": 0}
    ys = [stage_y.get(s, -1) for s in reference[:n_epochs]]
    ax_hyp.step(times_h, ys, where="post", color="#CDD6F4", linewidth=1)
    ax_hyp.set_yticks([0, 1, 2, 3, 4])
    ax_hyp.set_yticklabels(["REM", "N3", "N2", "N1", "W"])
    ax_hyp.set_title("Reference Hypnogram + ML Probabilities", fontsize=10)

    # Probabilities as stacked area
    ax_prob.stackplot(
        times_h,
        *[probabilities[:, i] for i in range(5)],
        labels=STAGE_ORDER,
        colors=[STAGE_COLORS[s] for s in STAGE_ORDER],
        alpha=0.7,
    )
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xlabel("Time (hours)")
    ax_prob.legend(loc="upper right", fontsize=7, ncol=5)
    ax_prob.set_ylim(0, 1)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 5. EEG excerpt panels
# -----------------------------------------------------------------

def plot_eeg_excerpts(
    signal: np.ndarray,
    sfreq: float,
    epochs_info: List[Dict[str, Any]],
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Annotated 30s EEG epochs showing classification examples.

    *epochs_info*: list of dicts with keys ``epoch_index``, ``true_stage``,
    ``pred_stage``, ``label`` (e.g. "Correct N2", "Misclassified REM->N1").
    """
    _apply_style()
    import matplotlib.pyplot as plt

    n = len(epochs_info)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, info in zip(axes, epochs_info):
        idx = info["epoch_index"]
        epoch_dur = info.get("epoch_duration_s", 30.0)
        i0 = int(idx * epoch_dur * sfreq)
        i1 = int((idx + 1) * epoch_dur * sfreq)
        i1 = min(i1, len(signal))
        segment = signal[i0:i1]
        t = np.linspace(0, epoch_dur, len(segment))

        correct = info.get("true_stage") == info.get("pred_stage")
        color = "#A6E3A1" if correct else "#F38BA8"
        ax.plot(t, segment, color=color, linewidth=0.4)
        ax.set_title(info.get("label", ""), fontsize=8, color=color)
        ax.set_ylabel("uV")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("EEG Epoch Examples", fontsize=11)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 6. PSD by stage
# -----------------------------------------------------------------

def plot_psd_by_stage(
    stage_psds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Welch PSD per stage (mean + CI shading).

    *stage_psds*: ``{stage: (freqs, psd_matrix)}`` where ``psd_matrix``
    has shape ``(n_epochs, n_freqs)``.
    """
    _apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    for stage in STAGE_ORDER:
        if stage not in stage_psds:
            continue
        freqs, psd_mat = stage_psds[stage]
        mean = np.mean(psd_mat, axis=0)
        std = np.std(psd_mat, axis=0)
        color = STAGE_COLORS[stage]
        ax.semilogy(freqs, mean, label=stage, color=color, linewidth=1.2)
        ax.fill_between(freqs, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (uV^2/Hz)")
    ax.set_title("Power Spectral Density by Sleep Stage")
    ax.set_xlim(0, 35)
    ax.legend(ncol=5, fontsize=7)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 7. Error timeline
# -----------------------------------------------------------------

def plot_error_timeline(
    reference: List[str],
    predicted: List[str],
    epoch_duration_s: float = 30.0,
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Coloured markers along the night showing disagreement locations."""
    _apply_style()
    import matplotlib.pyplot as plt

    n = min(len(reference), len(predicted))
    times_h = np.arange(n) * epoch_duration_s / 3600.0

    fig, ax = plt.subplots(figsize=(10, 1.5))

    for i in range(n):
        if reference[i] != predicted[i]:
            ax.axvline(times_h[i], color="#F38BA8", alpha=0.6, linewidth=0.5)

    # Shade by stage
    stage_y = {"W": 4, "N1": 3, "N2": 2, "N3": 1, "R": 0}
    ys = [stage_y.get(s, -1) for s in reference[:n]]
    ax.step(times_h, ys, where="post", color="#CDD6F4", linewidth=0.8)

    ax.set_xlabel("Time (hours)")
    ax.set_title("Error Timeline (red = disagreement)")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["REM", "N3", "N2", "N1", "W"])
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 8. Cross-validation boxplot
# -----------------------------------------------------------------

def plot_cv_boxplot(
    results: Dict[str, Dict[str, List[float]]],
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Boxplot of kappa/accuracy/F1 across folds per backend.

    *results*: ``{backend_name: {"accuracy": [...], "kappa": [...], "f1": [...]}}``
    """
    _apply_style()
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "kappa", "f1"]
    n_metrics = len(metrics)
    n_backends = len(results)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 3.5))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#89B4FA", "#A6E3A1", "#CBA6F7", "#F9E2AF", "#FAB387"]

    for ax, metric in zip(axes, metrics):
        data = []
        labels = []
        for idx, (backend, vals) in enumerate(results.items()):
            if metric in vals:
                data.append(vals[metric])
                labels.append(backend)

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Cross-Validation Performance", fontsize=11)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 9. Inter-scorer kappa heatmap
# -----------------------------------------------------------------

def plot_kappa_heatmap(
    names: List[str],
    kappa_matrix: np.ndarray,
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Pairwise kappa heatmap."""
    _apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(kappa_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")

    n = len(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{kappa_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if kappa_matrix[i, j] > 0.5 else "white")

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    ax.set_title("Inter-Scorer Agreement (Cohen's kappa)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 10. Feature importance
# -----------------------------------------------------------------

def plot_feature_importance(
    importance: Dict[str, float],
    top_n: int = 20,
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Horizontal bar chart of top-N feature importances."""
    _apply_style()
    import matplotlib.pyplot as plt

    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_items][::-1]
    values = [x[1] for x in sorted_items][::-1]

    fig, ax = plt.subplots(figsize=(5, max(3, 0.3 * len(names))))
    bars = ax.barh(range(len(names)), values, color="#89B4FA", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_n} Feature Importance")
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig


# -----------------------------------------------------------------
# 11. Clinical metrics table
# -----------------------------------------------------------------

def plot_clinical_metrics_table(
    metrics: Dict[str, Dict[str, float]],
    *,
    output_path: Optional[Path] = None,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Table figure comparing clinical metrics across backends.

    *metrics*: ``{backend_or_ref: {metric_name: value}}``
    """
    _apply_style()
    import matplotlib.pyplot as plt

    metric_keys = [
        "total_sleep_time_min", "sleep_efficiency_pct",
        "sleep_onset_latency_min", "waso_min", "rem_latency_min",
        "n1_pct", "n2_pct", "n3_pct", "rem_pct",
    ]
    display_names = [
        "TST (min)", "SE (%)", "SOL (min)", "WASO (min)",
        "REM Lat (min)", "N1%", "N2%", "N3%", "REM%",
    ]

    backends = list(metrics.keys())
    cell_text = []
    for key, dname in zip(metric_keys, display_names):
        row = [dname]
        for b in backends:
            val = metrics[b].get(key, 0.0)
            row.append(f"{val:.1f}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(backends)), 4))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric"] + backends,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title("Clinical Sleep Metrics Comparison", fontsize=11, pad=20)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path, fmt, dpi)
    return fig
