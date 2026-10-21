"""Figures for the SO-spindle coupling: circular statistics only.

A preferred phase is an angle, so it is never drawn on a linear axis and never
averaged arithmetically. Every figure here is polar: the distribution of the
coupled spindle phases inside one recording, the per-subject preferred phases
of a cohort, and the paired AVANT -> APRES angular shift.

Phase convention, as everywhere in :mod:`CESA.sleep_pipeline.so_spindle_coupling`:
0 rad is the SO positive peak (up-state), +/-pi the negative trough.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from CESA.sleep_pipeline.so_spindle_coupling import (
    MIN_COUPLED_EVENTS,
    circular_phase_stats,
    wrap_to_pi,
)

logger = logging.getLogger(__name__)

#: Below this many subjects, a group mean vector is not drawn: a resultant
#: length computed on a handful of angles is not interpretable.
MIN_GROUP_SUBJECTS = 5

DEFAULT_BINS = 18
BEFORE_LABEL = "AVANT"
AFTER_LABEL = "APRES"

#: Group tokens stripped from a file stem to pair the two nights of a subject.
_GROUP_TOKENS = {
    "AV", "AVANT", "BEFORE", "PRE", "PRETEST", "PRENAP", "BASELINE", "BASE",
    "AP", "APRES", "APRÈS", "AFTER", "POST", "POSTTEST", "POSTNAP",
    "RECOVERY", "RECUP", "NAP1", "NAP2", "J1", "J2", "T0", "T1",
}

_LIGHT = {
    "figure": "#FFFFFF",
    "axes": "#FAFAFA",
    "text": "#1A1A1A",
    "grid": "#C8C8C8",
    "bar": "#4C78A8",
    "mean": "#111111",
    "before": "#4C78A8",
    "after": "#E45756",
    "warn": "#B45309",
}
_DARK = {
    "figure": "#1E1E2E",
    "axes": "#1E1E2E",
    "text": "#CDD6F4",
    "grid": "#45475A",
    "bar": "#89B4FA",
    "mean": "#F9E2AF",
    "before": "#89B4FA",
    "after": "#F38BA8",
    "warn": "#FAB387",
}

StatsInput = Union[pd.DataFrame, str, "os.PathLike[str]"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _palette(dark: bool) -> Dict[str, str]:
    return dict(_DARK if dark else _LIGHT)


def _new_figure(dark: bool, figsize: Tuple[float, float]):
    from matplotlib.figure import Figure

    colors = _palette(dark)
    fig = Figure(figsize=figsize, dpi=100)
    fig.set_facecolor(colors["figure"])
    return fig


#: Reminder of the phase convention, written once per figure.
PHASE_LEGEND = (
    "0° = pic positif de l'onde lente (up-state) · ±180° = creux (down-state)"
)


def _style_polar(ax, colors: Dict[str, str], *, integer_radius: bool = False) -> None:
    """Common polar styling: up-state on the right, trough on the left."""
    from matplotlib.ticker import MaxNLocator

    ax.set_facecolor(colors["axes"])
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"])
    ax.set_rlabel_position(112.5)
    if integer_radius:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(colors=colors["text"], labelsize=8)
    ax.grid(True, color=colors["grid"], alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(colors["grid"])


def _mean_arrow(ax, angle: float, length: float, colors: Dict[str, str], **kwargs) -> None:
    """Draw the mean resultant vector from the centre of a polar axes."""
    ax.annotate(
        "",
        xy=(float(angle), float(length)),
        xytext=(0.0, 0.0),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=kwargs.get("linewidth", 2.2),
            color=kwargs.get("color", colors["mean"]),
            shrinkA=0.0,
            shrinkB=0.0,
        ),
    )


def _clean_angles(values: Sequence[float]) -> np.ndarray:
    """Finite angles wrapped to ``[-pi, pi)``."""
    arr = np.asarray(list(values), dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return np.array([wrap_to_pi(a) for a in arr], dtype=float)


def _as_frame(stats: StatsInput) -> pd.DataFrame:
    if isinstance(stats, pd.DataFrame):
        return stats.copy()
    return pd.read_csv(stats)


def participant_key(subject: str) -> str:
    """Subject identity shared by the two nights (``S030_Ap`` -> ``S030``).

    The batch writes one row per recording, so ``subject`` carries the
    condition. Pairing AVANT with APRES needs that token removed.
    """
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", str(subject)) if p]
    kept = [p for p in parts if p.upper() not in _GROUP_TOKENS]
    return "_".join(kept) if kept else str(subject)


def _channel_groups(stats: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """``(channel, rows)`` pairs, tolerating a table without a channel column."""
    if "channel" not in stats.columns:
        return [("", stats)]
    return [(str(name), rows) for name, rows in stats.groupby("channel", sort=True)]


def _panel_caption(ax, text: str, colors: Dict[str, str], color: str) -> None:
    """Caption under a panel, offset in points so grids of any size match."""
    ax.annotate(
        text, xy=(0.5, 0.0), xycoords="axes fraction",
        xytext=(0, -26), textcoords="offset points",
        ha="center", va="top", color=color, fontsize=8,
    )


def _grid_margins(fig, n_rows: int, *, caption_in: float) -> float:
    """Margins expressed in inches, so a 1-row and a 3-row grid both breathe.

    Returns the figure fraction where the bottom legend should sit. Without
    this the captions of one row land on the titles of the next one.
    """
    height = float(fig.get_figheight())
    top = 1.0 - 1.25 / height
    bottom = (caption_in + 0.45) / height
    hspace = 0.25
    if n_rows > 1:
        gap = caption_in + 0.40
        axes_height = ((top - bottom) * height - gap * (n_rows - 1)) / n_rows
        hspace = gap / max(axes_height, 0.5)
    fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace)
    return 0.22 / height


# ---------------------------------------------------------------------------
# One recording: distribution of the coupled spindle phases
# ---------------------------------------------------------------------------

def plot_phase_rose(
    phases: Sequence[float],
    *,
    channel: str = "",
    title: Optional[str] = None,
    bins: int = DEFAULT_BINS,
    stats: Optional[Dict[str, Any]] = None,
    min_coupled: int = MIN_COUPLED_EVENTS,
    dark: bool = False,
    fig=None,
    ax=None,
):
    """Circular histogram of the SO phases sampled at the spindle peaks.

    *phases* are the ``so_phase_rad`` of the coupled spindles of one channel.
    The mean resultant vector is drawn only above *min_coupled* events, the
    same guard as the tables.
    """
    angles = _clean_angles(phases)
    if angles.size == 0:
        raise ValueError("aucune phase a tracer")

    colors = _palette(dark)
    if ax is None:
        fig = fig if fig is not None else _new_figure(dark, (6.0, 6.0))
        ax = fig.add_subplot(111, projection="polar")
    elif fig is None:
        fig = ax.get_figure()

    counts, edges = np.histogram(angles, bins=int(bins), range=(-np.pi, np.pi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.bar(
        centers, counts,
        width=(2.0 * np.pi / float(bins)),
        bottom=0.0,
        color=colors["bar"],
        edgecolor=colors["figure"],
        linewidth=0.6,
        alpha=0.85,
        align="center",
    )
    _style_polar(ax, colors, integer_radius=True)

    row = dict(stats) if stats else circular_phase_stats(angles)
    n = int(row.get("n", angles.size) or angles.size)
    top = float(counts.max()) if counts.size else 1.0
    r = float(row.get("resultant_length", float("nan")))
    mean_rad = float(row.get("preferred_phase_rad", float("nan")))

    if n >= int(min_coupled) and np.isfinite(r) and np.isfinite(mean_rad):
        _mean_arrow(ax, mean_rad, r * top, colors)
        caption = (
            f"n = {n}   phase préférée = {np.degrees(mean_rad):.1f}°   "
            f"R = {r:.3f}   Rayleigh p = {row.get('rayleigh_p', float('nan')):.3g}"
        )
        caption_color = colors["text"]
    else:
        caption = (
            f"n = {n} < {int(min_coupled)} fuseaux couplés : "
            f"phase préférée non publiée"
        )
        caption_color = colors["warn"]

    heading = title or (f"Couplage SO-fuseau — {channel}" if channel else "Couplage SO-fuseau")
    ax.set_title(heading, color=colors["text"], fontsize=11, pad=18)
    ax.text(
        0.5, -0.10, caption, transform=ax.transAxes, ha="center", va="top",
        color=caption_color, fontsize=9,
    )
    ax.text(
        0.5, -0.16, PHASE_LEGEND, transform=ax.transAxes, ha="center", va="top",
        color=colors["text"], fontsize=8,
    )
    fig.subplots_adjust(bottom=0.16)
    return fig


# ---------------------------------------------------------------------------
# Cohort: one preferred phase per subject
# ---------------------------------------------------------------------------

def plot_group_phases(
    stats: StatsInput,
    *,
    group_column: str = "group",
    before: str = BEFORE_LABEL,
    after: str = AFTER_LABEL,
    min_subjects: int = MIN_GROUP_SUBJECTS,
    dark: bool = False,
):
    """Per-subject preferred phases of a cohort, one polar axes per channel.

    Each point is one recording: angle = preferred phase, radius = resultant
    length. The arrow is the second-order mean, i.e. the circular mean of the
    subject phases, which is the only valid way to summarise them.
    """
    table = _as_frame(stats)
    if "preferred_phase_rad" not in table.columns:
        raise ValueError("colonne 'preferred_phase_rad' absente")

    channels = _channel_groups(table)
    colors = _palette(dark)
    n_cols = min(3, len(channels))
    n_rows = int(np.ceil(len(channels) / n_cols))
    fig = _new_figure(dark, (5.0 * n_cols, 5.4 * n_rows))
    seen_labels: Dict[str, str] = {}

    for index, (channel, rows) in enumerate(channels, start=1):
        ax = fig.add_subplot(n_rows, n_cols, index, projection="polar")
        _style_polar(ax, colors)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])

        usable = rows[np.isfinite(rows["preferred_phase_rad"].astype(float))]
        hidden = len(rows) - len(usable)
        groups = (
            [(str(g), sub) for g, sub in usable.groupby(group_column, sort=True)]
            if group_column in usable.columns and usable[group_column].notna().any()
            else [("", usable)]
        )

        lines: List[str] = []
        for label, sub in groups:
            key = label.strip().upper()
            color = (
                colors["before"] if key == before.upper()
                else colors["after"] if key == after.upper()
                else colors["bar"]
            )
            angles = _clean_angles(sub["preferred_phase_rad"])
            radii = (
                np.clip(sub["resultant_length"].astype(float).to_numpy(), 0.0, 1.0)
                if "resultant_length" in sub.columns
                else np.full(angles.size, 0.8)
            )
            ax.scatter(
                angles, radii[: angles.size],
                s=46, color=color, edgecolor=colors["figure"],
                linewidth=0.8, zorder=3,
            )
            if label:
                seen_labels[label] = color
            summary = circular_phase_stats(angles)
            if angles.size >= int(min_subjects):
                _mean_arrow(
                    ax, summary["preferred_phase_rad"], summary["resultant_length"],
                    colors, color=color, linewidth=2.6,
                )
                lines.append(
                    f"{label or 'tous'} : n = {angles.size}, "
                    f"{summary['preferred_phase_deg']:.1f}°, "
                    f"R = {summary['resultant_length']:.3f}, "
                    f"p = {summary['rayleigh_p']:.3g}"
                )
            else:
                lines.append(
                    f"{label or 'tous'} : n = {angles.size} < {int(min_subjects)}, "
                    "moyenne de groupe non tracée"
                )
        if hidden:
            lines.append(f"{hidden} enregistrement(s) sans phase publiée")

        ax.set_title(channel or "Tous canaux", color=colors["text"], fontsize=11, pad=14)
        _panel_caption(ax, "\n".join(lines), colors, colors["text"])

    legend_y = _grid_margins(fig, n_rows, caption_in=0.75)
    _figure_legend(fig, seen_labels, colors)
    fig.suptitle(
        "Phase préférée du couplage SO-fuseau par sujet\n"
        "rayon = R (force du couplage)",
        color=colors["text"], fontsize=12,
    )
    fig.text(0.5, legend_y, PHASE_LEGEND, ha="center", color=colors["text"], fontsize=8)
    return fig


def _figure_legend(fig, labels: Dict[str, str], colors: Dict[str, str]) -> None:
    """One legend for the whole figure, so panels stay uncluttered."""
    if len(labels) < 2:
        return
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markersize=7,
            markerfacecolor=color, markeredgecolor=colors["figure"], label=label,
        )
        for label, color in labels.items()
    ]
    legend = fig.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
        fontsize=9, frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(colors["text"])


def plot_phase_shift(
    stats: StatsInput,
    *,
    group_column: str = "group",
    subject_column: str = "subject",
    before: str = BEFORE_LABEL,
    after: str = AFTER_LABEL,
    bins: int = 12,
    min_subjects: int = MIN_GROUP_SUBJECTS,
    dark: bool = False,
):
    """Paired angular shift APRES - AVANT, one polar axes per channel.

    The test of "the coupling moved" is a Rayleigh test on the per-subject
    angular differences: a shift concentrated away from 0° means the preferred
    phase changed in a consistent direction. A Wilcoxon on the angles
    themselves would be meaningless.
    """
    table = _as_frame(stats)
    for column in ("preferred_phase_rad", group_column, subject_column):
        if column not in table.columns:
            raise ValueError(f"colonne '{column}' absente")

    colors = _palette(dark)
    channels = _channel_groups(table)
    n_cols = min(3, len(channels))
    n_rows = int(np.ceil(len(channels) / n_cols))
    fig = _new_figure(dark, (5.0 * n_cols, 5.4 * n_rows))

    for index, (channel, rows) in enumerate(channels, start=1):
        ax = fig.add_subplot(n_rows, n_cols, index, projection="polar")
        _style_polar(ax, colors, integer_radius=True)

        deltas = paired_phase_shifts(
            rows, before=before, after=after,
            group_column=group_column, subject_column=subject_column,
        )
        if deltas.size:
            counts, edges = np.histogram(deltas, bins=int(bins), range=(-np.pi, np.pi))
            centers = (edges[:-1] + edges[1:]) / 2.0
            ax.bar(
                centers, counts,
                width=(2.0 * np.pi / float(bins)), bottom=0.0,
                color=colors["after"], edgecolor=colors["figure"],
                linewidth=0.6, alpha=0.85, align="center",
            )
            summary = circular_phase_stats(deltas)
            if deltas.size >= int(min_subjects):
                _mean_arrow(
                    ax, summary["preferred_phase_rad"],
                    summary["resultant_length"] * float(counts.max()), colors,
                )
                caption = (
                    f"{deltas.size} paires   décalage moyen = "
                    f"{summary['preferred_phase_deg']:.1f}°   "
                    f"R = {summary['resultant_length']:.3f}   "
                    f"Rayleigh p = {summary['rayleigh_p']:.3g}"
                )
                caption_color = colors["text"]
            else:
                caption = (
                    f"{deltas.size} paire(s) < {int(min_subjects)} : "
                    "décalage moyen non tracé"
                )
                caption_color = colors["warn"]
        else:
            caption = "aucune paire AVANT/APRES complète"
            caption_color = colors["warn"]

        ax.set_title(channel or "Tous canaux", color=colors["text"], fontsize=11, pad=14)
        _panel_caption(ax, caption, colors, caption_color)

    legend_y = _grid_margins(fig, n_rows, caption_in=0.45)
    fig.suptitle(
        f"Décalage de phase {after} − {before} par sujet\n"
        "test de Rayleigh sur les différences angulaires",
        color=colors["text"], fontsize=12,
    )
    fig.text(
        0.5, legend_y,
        "0° = phase inchangée · rayon = nombre de sujets",
        ha="center", color=colors["text"], fontsize=8,
    )
    return fig


def paired_phase_shifts(
    stats: StatsInput,
    *,
    before: str = BEFORE_LABEL,
    after: str = AFTER_LABEL,
    group_column: str = "group",
    subject_column: str = "subject",
) -> np.ndarray:
    """Per-subject angular differences ``after - before``, wrapped to [-pi, pi).

    The batch names a row after its recording (``S030_Ap``), so the two
    conditions of a subject are paired through :func:`participant_key`.
    Subjects missing one condition, or whose preferred phase was withheld, are
    dropped.
    """
    table = _as_frame(stats)
    if subject_column not in table.columns or group_column not in table.columns:
        return np.array([], dtype=float)

    phases = table["preferred_phase_rad"].astype(float)
    usable = table[np.isfinite(phases)].copy()
    if usable.empty:
        return np.array([], dtype=float)
    usable["_participant"] = usable[subject_column].map(participant_key)
    usable["_condition"] = usable[group_column].astype(str).str.strip().str.upper()

    first = usable.groupby(["_participant", "_condition"])["preferred_phase_rad"].first()
    shifts: List[float] = []
    for participant in first.index.get_level_values(0).unique():
        row = first.loc[participant]
        if before.upper() in row.index and after.upper() in row.index:
            shifts.append(
                wrap_to_pi(float(row[after.upper()]) - float(row[before.upper()]))
            )
    return np.asarray(shifts, dtype=float)


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def save_coupling_figures(
    stats: StatsInput,
    output_dir: Union[str, "os.PathLike[str]"],
    *,
    prefix: str = "coupling",
    before: str = BEFORE_LABEL,
    after: str = AFTER_LABEL,
    dpi: int = 200,
    dark: bool = False,
) -> List[str]:
    """Write the cohort figures next to the batch CSV; return the paths.

    A figure that cannot be built (no usable phase, no complete pair) is
    skipped instead of raising: a batch must not fail on its illustrations.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    table = _as_frame(stats)
    os.makedirs(str(output_dir), exist_ok=True)
    written: List[str] = []

    jobs = [
        ("phases", lambda: plot_group_phases(
            table, before=before, after=after, dark=dark,
        )),
    ]
    if _has_both_conditions(table, before, after):
        jobs.append(("decalage", lambda: plot_phase_shift(
            table, before=before, after=after, dark=dark,
        )))

    for name, build in jobs:
        try:
            fig = build()
        except Exception as exc:
            logger.warning("Figure '%s' non generee: %s", name, exc)
            continue
        path = os.path.join(str(output_dir), f"{prefix}_{name}.png")
        FigureCanvasAgg(fig)
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight", facecolor=fig.get_facecolor())
        written.append(path)
    return written


def _has_both_conditions(table: pd.DataFrame, before: str, after: str) -> bool:
    if "group" not in table.columns:
        return False
    keys = set(table["group"].dropna().astype(str).str.strip().str.upper())
    return before.upper() in keys and after.upper() in keys
