"""Qt dialog: sleep-period analysis (SleepEEGpy-style).

Computes contiguous stage episodes and clinical PSG metrics from the active
scoring via :func:`CESA.sleep_metrics.compute_psg_summary`, and presents a
metrics summary, an episode table and a stage-duration chart. Replaces the Tk
``_analyze_sleep_periods`` window.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6 import QtCore, QtWidgets

logger = logging.getLogger(__name__)

_STAGE_COLORS = {
    "W": "#f38ba8", "N1": "#89b4fa", "N2": "#74c7ec",
    "N3": "#89dceb", "R": "#cba6f7",
}


class SleepPeriodsDialog(QtWidgets.QDialog):
    """Sleep-period episodes + clinical metrics for the active scoring."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], controller) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analyse des periodes de sommeil")
        self.resize(1000, 760)
        self._ctrl = controller
        self._summary = None
        self._build_ui()
        self._compute()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._summary_label = QtWidgets.QLabel("")
        self._summary_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        split = QtWidgets.QSplitter()
        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Debut (min)", "Fin (min)", "Stade", "Duree (min)"]
        )
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        split.addWidget(self._table)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        plot_w = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plot_w)
        self._fig = Figure(figsize=(5, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        pv.addWidget(self._canvas)
        split.addWidget(plot_w)
        split.setSizes([520, 460])
        layout.addWidget(split, stretch=1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch()
        exp = QtWidgets.QPushButton("Exporter rapport PSG (CSV)...")
        exp.clicked.connect(self._export)
        bottom.addWidget(exp)
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    def _compute(self) -> None:
        ctrl = self._ctrl
        try:
            df = ctrl.get_active_scoring_df()
        except Exception:
            df = getattr(ctrl, "sleep_scoring_data", None)
        if df is None or df.empty:
            QtWidgets.QMessageBox.warning(
                self, "Periodes de sommeil", "Aucun scoring disponible.")
            self.reject()
            return
        try:
            from CESA.sleep_metrics import compute_psg_summary

            self._summary = compute_psg_summary(
                df,
                float(getattr(ctrl, "scoring_epoch_duration", 30.0)),
                night_start_s=getattr(ctrl, "night_start_s", None),
                night_end_s=getattr(ctrl, "night_end_s", None),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Calcul echoue:\n{exc}")
            self.reject()
            return
        self._populate()

    def _populate(self) -> None:
        s = self._summary
        cm = s.clinical
        dur = s.stage_durations_min
        note = ""
        if s.night_start_s is not None and s.night_end_s is not None:
            note = (f" &nbsp; | &nbsp; Fenetre: {s.night_start_s / 60.0:.1f}"
                    f"–{s.night_end_s / 60.0:.1f} min")
        self._summary_label.setText(
            f"<b>TIB</b> {cm.time_in_bed_min:.1f} min &nbsp; "
            f"<b>TST</b> {cm.total_sleep_time_min:.1f} min &nbsp; "
            f"<b>SOL</b> {cm.sleep_onset_latency_min:.1f} min &nbsp; "
            f"<b>WASO</b> {cm.waso_min:.1f} min &nbsp; "
            f"<b>SE</b> {cm.sleep_efficiency_pct:.1f} % &nbsp; "
            f"<b>REM lat.</b> {cm.rem_latency_min:.1f} min &nbsp; "
            f"<b>Reveils</b> {s.awakenings} &nbsp; "
            f"<b>Cycles</b> {s.n_cycles}{note}<br>"
            + " &nbsp; ".join(
                f"<b>{st}</b> {dur.get(st, 0.0):.1f}" for st in ["W", "N1", "N2", "N3", "R"]
            )
        )

        self._table.setRowCount(len(s.episodes))
        for r, ep in enumerate(s.episodes):
            self._table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{ep.start_min:.2f}"))
            self._table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{ep.end_min:.2f}"))
            self._table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(ep.stage)))
            self._table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{ep.duration_min:.2f}"))

        self._fig.clear()
        ax1 = self._fig.add_subplot(221)
        stages = [st for st in ["W", "N1", "N2", "N3", "R"] if dur.get(st, 0.0) > 0.05]
        vals = [dur[st] for st in stages]
        colors = [_STAGE_COLORS.get(st, "#6c7086") for st in stages]
        if vals:
            ax1.pie(vals, labels=stages, colors=colors, autopct="%1.0f%%",
                    textprops={"fontsize": 8})
        ax1.set_title("Repartition des stades (TST)")

        ax2 = self._fig.add_subplot(222)
        ax2.bar(stages, vals, color=colors)
        ax2.set_ylabel("Duree (min)")
        ax2.set_title("Duree par stade")
        ax2.grid(True, alpha=0.3, axis="y")

        ax3 = self._fig.add_subplot(212)
        if s.transition_probs is not None and s.transition_probs.size:
            labs = list(s.transition_labels)
            im = ax3.imshow(s.transition_probs, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax3.set_xticks(range(len(labs)))
            ax3.set_yticks(range(len(labs)))
            ax3.set_xticklabels(labs)
            ax3.set_yticklabels(labs)
            ax3.set_xlabel("Vers")
            ax3.set_ylabel("Depuis")
            ax3.set_title(f"Transitions (probs) — {s.n_cycles} cycle(s)")
            self._fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        else:
            ax3.set_title("Pas de matrice de transitions")
            ax3.axis("off")
        self._fig.tight_layout()
        self._canvas.draw()

    def _export(self) -> None:
        if self._summary is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter rapport PSG", "", "CSV (*.csv)")
        if not path:
            return
        try:
            from CESA.sleep_metrics import export_psg_summary_csv

            export_psg_summary_csv(self._summary, path)
            QtWidgets.QMessageBox.information(self, "Export", f"Rapport exporte:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")
