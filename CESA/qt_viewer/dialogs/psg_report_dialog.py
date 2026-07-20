"""Visual PSG report dialog with pie charts and episode tables."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from CESA.sleep_metrics import PsgSummary, export_psg_summary_csv

logger = logging.getLogger(__name__)

_STAGE_COLORS: Dict[str, str] = {
    "W": "#EBA0AC",
    "N1": "#89B4FA",
    "N2": "#74C7EC",
    "N3": "#89DCEB",
    "R": "#CBA6F7",
    "REM": "#CBA6F7",
    "U": "#6C7086",
}


def _stage_label(stage: str) -> str:
    if stage == "R":
        return "REM"
    return stage


class PsgReportDialog(QtWidgets.QDialog):
    """Polysomnography summary with pie charts and structured tables."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        summary: PsgSummary,
        *,
        title_suffix: str = "",
        annotations: Optional[List[Dict[str, Any]]] = None,
        scoring_epochs: Optional[List[Tuple[float, str]]] = None,
        epoch_seconds: float = 30.0,
    ) -> None:
        super().__init__(parent)
        self._summary = summary
        self._annotations = list(annotations or summary.annotations or [])
        self._scoring_epochs = list(scoring_epochs or [])
        self._epoch_seconds = float(epoch_seconds)
        suffix = f" — {title_suffix}" if title_suffix else ""
        self.setWindowTitle(f"Rapport PSG{suffix}")
        self.resize(1180, 900)
        self.setMinimumSize(900, 700)
        self._build_ui()
        self._populate()

    @classmethod
    def open_from_controller(
        cls,
        parent: Optional[QtWidgets.QWidget],
        ctrl,
        *,
        annotations: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional["PsgReportDialog"]:
        """Compute summary from AppController and show dialog."""
        if ctrl is None or ctrl.sleep_scoring_data is None or ctrl.sleep_scoring_data.empty:
            QtWidgets.QMessageBox.warning(
                parent, "Rapport PSG", "Aucun scoring disponible.",
            )
            return None
        from CESA.sleep_metrics import compute_psg_summary

        df = ctrl.sleep_scoring_data
        if df is None or df.empty:
            QtWidgets.QMessageBox.warning(
                parent, "Rapport PSG", "Aucune epoque dans la fenetre d'analyse.",
            )
            return None

        if annotations is None and parent is not None:
            store = getattr(parent, "annotation_store", None)
            if store is not None:
                annotations = store.to_event_list()

        summary = compute_psg_summary(
            df,
            ctrl.scoring_epoch_duration,
            night_start_s=ctrl.night_start_s,
            night_end_s=ctrl.night_end_s,
        )
        summary.annotations = list(annotations or [])

        scoring_epochs: List[Tuple[float, str]] = []
        work = df
        if ctrl.has_night_window:
            from CESA.sleep_metrics import _filter_window
            work = _filter_window(
                df,
                ctrl.scoring_epoch_duration,
                ctrl.night_start_s,
                ctrl.night_end_s,
            )
        for _, row in work.iterrows():
            scoring_epochs.append((
                float(row.get("time", 0.0)),
                str(row.get("stage", "U")).upper(),
            ))

        suffix = ""
        if ctrl.has_night_window:
            suffix = (
                f"fenetre {ctrl.night_start_s / 60.0:.0f}–{ctrl.night_end_s / 60.0:.0f} min"
            )
        dlg = cls(
            parent,
            summary,
            title_suffix=suffix,
            annotations=annotations,
            scoring_epochs=scoring_epochs,
            epoch_seconds=ctrl.scoring_epoch_duration,
        )
        dlg.show()
        return dlg

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self._window_banner = QtWidgets.QLabel()
        self._window_banner.setWordWrap(True)
        self._window_banner.setStyleSheet(
            "font-size: 12px; font-weight: 600; padding: 10px 12px; "
            "background: #1B4332; border: 2px solid #4CAF50; border-radius: 6px; color: #A5D6A7;"
        )
        self._window_banner.setVisible(False)
        root.addWidget(self._window_banner)

        header = QtWidgets.QLabel()
        header.setWordWrap(True)
        header.setStyleSheet(
            "font-size: 13px; font-weight: 600; padding: 8px; "
            "background: #313244; border-radius: 6px; color: #CDD6F4;"
        )
        self._header_label = header
        root.addWidget(header)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._timeline_fig = Figure(figsize=(10, 1.6), dpi=100)
        self._timeline_fig.set_facecolor("#1E1E2E")
        self._timeline_canvas = FigureCanvasQTAgg(self._timeline_fig)
        self._timeline_canvas.setFixedHeight(130)
        root.addWidget(self._timeline_canvas)

        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, stretch=1)

        left = QtWidgets.QVBoxLayout()
        metrics_box = QtWidgets.QGroupBox("Architecture du sommeil")
        metrics_box.setStyleSheet("QGroupBox { font-weight: bold; color: #89B4FA; }")
        form = QtWidgets.QFormLayout(metrics_box)
        form.setSpacing(8)
        self._metric_labels: Dict[str, QtWidgets.QLabel] = {}
        for key, label in (
            ("tib", "Temps au lit (TIB)"),
            ("tst", "Temps de sommeil (TST)"),
            ("se", "Efficacite (SE)"),
            ("sol", "Latence endormissement (SOL)"),
            ("waso", "Eveil intra-sommeil (WASO)"),
            ("rem_lat", "Latence REM"),
            ("awakenings", "Reveils"),
            ("ann_count", "Annotations"),
        ):
            lbl = QtWidgets.QLabel("--")
            lbl.setStyleSheet("color: #CDD6F4; font-size: 12px;")
            form.addRow(f"{label}:", lbl)
            self._metric_labels[key] = lbl
        left.addWidget(metrics_box)

        export_btn = QtWidgets.QPushButton("Exporter CSV…")
        export_btn.clicked.connect(self._export_csv)
        left.addWidget(export_btn)
        left.addStretch()
        body.addLayout(left, stretch=0)

        self._fig = Figure(figsize=(8, 5), dpi=100)
        self._fig.set_facecolor("#1E1E2E")
        self._canvas = FigureCanvasQTAgg(self._fig)
        body.addWidget(self._canvas, stretch=2)

        tables = QtWidgets.QTabWidget()
        tables.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #45475A; background: #1E1E2E; }"
            "QTabBar::tab { background: #313244; color: #CDD6F4; padding: 6px 12px; }"
            "QTabBar::tab:selected { background: #45475A; }"
        )

        self._stage_table = QtWidgets.QTableWidget(0, 5)
        self._stage_table.setHorizontalHeaderLabels([
            "Stade", "Duree (min)", "% TST", "Nb episodes", "Duree moy. episode (min)",
        ])
        self._stage_table.horizontalHeader().setStretchLastSection(True)
        self._stage_table.verticalHeader().setVisible(False)
        self._stage_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers,
        )
        self._stage_table.setAlternatingRowColors(True)
        tables.addTab(self._stage_table, "Synthese par stade")

        self._episode_table = QtWidgets.QTableWidget(0, 4)
        self._episode_table.setHorizontalHeaderLabels([
            "Debut (min)", "Fin (min)", "Stade", "Duree (min)",
        ])
        self._episode_table.horizontalHeader().setStretchLastSection(True)
        self._episode_table.verticalHeader().setVisible(False)
        self._episode_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers,
        )
        self._episode_table.setAlternatingRowColors(True)
        tables.addTab(self._episode_table, "Episodes detailles")

        self._annotation_table = QtWidgets.QTableWidget(0, 6)
        self._annotation_table.setHorizontalHeaderLabels([
            "Debut (min)", "Fin (min)", "Type", "Label", "Canal", "Source",
        ])
        self._annotation_table.horizontalHeader().setStretchLastSection(True)
        self._annotation_table.verticalHeader().setVisible(False)
        self._annotation_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers,
        )
        self._annotation_table.setAlternatingRowColors(True)
        tables.addTab(self._annotation_table, "Annotations")

        root.addWidget(tables, stretch=1)

        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.accept)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _populate(self) -> None:
        s = self._summary
        cm = s.clinical

        if s.night_start_s is not None and s.night_end_s is not None:
            self._window_banner.setVisible(True)
            self._window_banner.setText(
                f"Fenetre d'analyse active  ·  "
                f"Lights off {s.night_start_s / 60.0:.1f} min  →  "
                f"Lights on {s.night_end_s / 60.0:.1f} min  ·  "
                f"TIB {cm.time_in_bed_min:.1f} min  ·  "
                f"TST {cm.total_sleep_time_min:.1f} min  ·  "
                f"SE {cm.sleep_efficiency_pct:.1f} %"
            )
        else:
            self._window_banner.setVisible(False)

        window_txt = ""
        if s.night_start_s is not None and s.night_end_s is not None:
            window_txt = (
                f"Fenetre d'analyse : {s.night_start_s / 60.0:.1f} – "
                f"{s.night_end_s / 60.0:.1f} min  ·  "
            )
        self._header_label.setText(
            f"{window_txt}"
            f"{cm.n_epochs} epoques analysees  ·  "
            f"TST {cm.total_sleep_time_min:.1f} min  ·  "
            f"SE {cm.sleep_efficiency_pct:.1f}%  ·  "
            f"{len(self._annotations)} annotation(s)"
        )

        self._metric_labels["tib"].setText(f"{cm.time_in_bed_min:.1f} min")
        self._metric_labels["tst"].setText(f"{cm.total_sleep_time_min:.1f} min")
        self._metric_labels["se"].setText(f"{cm.sleep_efficiency_pct:.1f} %")
        self._metric_labels["sol"].setText(f"{cm.sleep_onset_latency_min:.1f} min")
        self._metric_labels["waso"].setText(f"{cm.waso_min:.1f} min")
        self._metric_labels["rem_lat"].setText(f"{cm.rem_latency_min:.1f} min")
        self._metric_labels["awakenings"].setText(str(s.awakenings))
        self._metric_labels["ann_count"].setText(str(len(self._annotations)))

        self._fill_stage_table(s)
        self._fill_episode_table(s)
        self._fill_annotation_table()
        self._draw_timeline()
        self._draw_pies(s)

    def _fill_stage_table(self, s: PsgSummary) -> None:
        rows: List[Tuple[str, float, float, int, float]] = []
        for stage in ("W", "N1", "N2", "N3", "R"):
            rows.append((
                _stage_label(stage),
                s.stage_durations_min.get(stage, 0.0),
                s.stage_pct_of_tst.get(stage, 0.0),
                s.episode_counts.get(stage, 0),
                s.episode_mean_duration_min.get(stage, 0.0),
            ))
        self._stage_table.setRowCount(len(rows))
        for i, (stage, dur, pct, n_ep, mean_d) in enumerate(rows):
            pct_txt = f"{pct:.1f} %" if stage != "W" else "—"
            for j, val in enumerate((stage, f"{dur:.1f}", pct_txt, str(n_ep), f"{mean_d:.1f}")):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self._stage_table.setItem(i, j, item)

    def _fill_episode_table(self, s: PsgSummary) -> None:
        self._episode_table.setRowCount(len(s.episodes))
        for i, ep in enumerate(s.episodes):
            for j, val in enumerate((
                f"{ep.start_min:.1f}",
                f"{ep.end_min:.1f}",
                _stage_label(ep.stage),
                f"{ep.duration_min:.1f}",
            )):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                color = _STAGE_COLORS.get(ep.stage.upper(), "#CDD6F4")
                item.setForeground(QtGui.QColor(color))
                self._episode_table.setItem(i, j, item)

    def _fill_annotation_table(self) -> None:
        rows: List[Tuple[float, float, str, str, str, str, str]] = []
        for ann in self._annotations:
            onset_s = float(ann.get("onset", 0.0))
            dur_s = float(ann.get("duration", 0.0))
            rows.append((
                onset_s / 60.0,
                (onset_s + dur_s) / 60.0,
                str(ann.get("type", "")),
                str(ann.get("label", ann.get("text", ""))),
                str(ann.get("channel", "")),
                str(ann.get("source", "manual")),
                str(ann.get("color", "#F9E2AF")),
            ))
        self._annotation_table.setRowCount(len(rows))
        for i, (start_m, end_m, atype, label, channel, source, color) in enumerate(rows):
            for j, val in enumerate((
                f"{start_m:.1f}",
                f"{end_m:.1f}",
                atype,
                label,
                channel or "—",
                source,
            )):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                item.setForeground(QtGui.QColor(color))
                self._annotation_table.setItem(i, j, item)

    def _draw_timeline(self) -> None:
        """Compact hypnogram + night window + annotations for the analysis view."""
        self._timeline_fig.clear()
        ax = self._timeline_fig.add_subplot(111)
        ax.set_facecolor("#1E1E2E")
        for spine in ax.spines.values():
            spine.set_color("#45475A")
        ax.tick_params(colors="#CDD6F4", labelsize=8)

        s = self._summary
        ep = self._epoch_seconds
        if not self._scoring_epochs:
            ax.text(
                0.5, 0.5, "Aucune epoque dans la fenetre",
                ha="center", va="center", color="#6C7086",
                transform=ax.transAxes,
            )
            ax.axis("off")
            self._timeline_canvas.draw_idle()
            return

        t_min = min(t for t, _ in self._scoring_epochs) / 60.0
        t_max = max(t for t, _ in self._scoring_epochs) / 60.0 + ep / 60.0
        if s.night_start_s is not None and s.night_end_s is not None:
            t_min = min(t_min, s.night_start_s / 60.0)
            t_max = max(t_max, s.night_end_s / 60.0)

        for t_s, stage in self._scoring_epochs:
            color = _STAGE_COLORS.get(stage.upper(), "#6C7086")
            ax.barh(
                0.5, ep / 60.0, left=t_s / 60.0, height=0.7,
                color=color, edgecolor="#1E1E2E", linewidth=0.4, alpha=0.95,
            )

        if s.night_start_s is not None and s.night_end_s is not None:
            ns = s.night_start_s / 60.0
            ne = s.night_end_s / 60.0
            ax.axvspan(ns, ne, color="#4CAF50", alpha=0.18, zorder=0)
            ax.axvline(ns, color="#66BB6A", linewidth=2.5, linestyle="-", label="Lights off")
            ax.axvline(ne, color="#EF5350", linewidth=2.5, linestyle="-", label="Lights on")

        for ann in self._annotations:
            onset_m = float(ann.get("onset", 0.0)) / 60.0
            dur_m = float(ann.get("duration", 0.0)) / 60.0
            color = str(ann.get("color") or "#F9E2AF")
            ax.barh(
                1.15, max(dur_m, 0.05), left=onset_m, height=0.35,
                color=color, edgecolor="#CDD6F4", linewidth=0.8, alpha=0.95,
            )
            label = str(ann.get("label") or ann.get("type") or "")[:14]
            if label:
                ax.text(
                    onset_m + 0.05, 1.32, label,
                    color="#CDD6F4", fontsize=7, va="bottom",
                )

        ax.set_xlim(t_min - 1, t_max + 1)
        ax.set_ylim(0, 1.6)
        ax.set_yticks([0.5, 1.32])
        ax.set_yticklabels(["Hypnogramme", "Annotations"], color="#CDD6F4")
        ax.set_xlabel("Temps (min)", color="#CDD6F4", fontsize=9)
        ax.set_title(
            "Vue d'analyse — hypnogramme et annotations",
            color="#CDD6F4", fontsize=10, fontweight="bold", pad=6,
        )
        if s.night_start_s is not None and s.night_end_s is not None:
            ax.legend(loc="upper right", fontsize=7, facecolor="#313244", labelcolor="#CDD6F4")
        self._timeline_fig.tight_layout(pad=1.2)
        self._timeline_canvas.draw_idle()

    def _draw_pies(self, s: PsgSummary) -> None:
        self._fig.clear()
        ax1 = self._fig.add_subplot(1, 2, 1)
        ax2 = self._fig.add_subplot(1, 2, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor("#1E1E2E")
            for spine in ax.spines.values():
                spine.set_color("#45475A")
            ax.tick_params(colors="#CDD6F4", labelsize=9)

        sleep_stages = ["N1", "N2", "N3", "R"]
        sleep_pcts = [s.stage_pct_of_tst.get(st, 0.0) for st in sleep_stages]
        sleep_labels = [_stage_label(st) for st in sleep_stages]
        sleep_colors = [_STAGE_COLORS[st] for st in sleep_stages]
        self._plot_pie(
            ax1,
            sleep_pcts,
            sleep_labels,
            sleep_colors,
            "Repartition du sommeil (% TST)",
        )

        cm = s.clinical
        tib = max(cm.time_in_bed_min, 1e-6)
        tib_stages = ["W", "N1", "N2", "N3", "R"]
        tib_pcts = [
            100.0 * s.stage_durations_min.get(st, 0.0) / tib for st in tib_stages
        ]
        tib_labels = [_stage_label(st) for st in tib_stages]
        tib_colors = [_STAGE_COLORS[st] for st in tib_stages]
        self._plot_pie(
            ax2,
            tib_pcts,
            tib_labels,
            tib_colors,
            "Repartition du temps au lit (% TIB)",
        )

        self._fig.tight_layout(pad=2.0)
        self._canvas.draw_idle()

    @staticmethod
    def _plot_pie(
        ax,
        values: List[float],
        labels: List[str],
        colors: List[str],
        title: str,
    ) -> None:
        pairs = [(v, lb, c) for v, lb, c in zip(values, labels, colors) if v > 0.05]
        if not pairs:
            ax.text(
                0.5, 0.5, "Aucune donnee",
                ha="center", va="center", color="#6C7086",
                transform=ax.transAxes, fontsize=12,
            )
            ax.set_title(title, color="#CDD6F4", fontsize=11, fontweight="bold", pad=10)
            ax.axis("off")
            return

        vals, labs, cols = zip(*pairs)
        _wedges, _texts, autotexts = ax.pie(
            vals,
            labels=labs,
            colors=cols,
            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"edgecolor": "#1E1E2E", "linewidth": 1.2},
            textprops={"color": "#CDD6F4", "fontsize": 9},
        )
        for t in autotexts:
            t.set_color("#1E1E2E")
            t.set_fontsize(8)
            t.set_fontweight("bold")
        ax.set_title(title, color="#CDD6F4", fontsize=11, fontweight="bold", pad=10)

    def _export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter rapport PSG", "", "CSV (*.csv);;Tous (*.*)",
        )
        if not path:
            return
        try:
            export_psg_summary_csv(
                self._summary, path, annotations=self._annotations,
            )
            QtWidgets.QMessageBox.information(
                self, "Export", f"Rapport exporte :\n{path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Export echoue : {exc}")
