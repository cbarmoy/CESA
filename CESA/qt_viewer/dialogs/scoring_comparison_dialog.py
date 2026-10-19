"""Qt dialog: detailed auto-vs-manual scoring comparison.

Given the concordance DataFrame produced by
``AppController.compare_scoring`` (columns ``epoch``, ``auto``, ``manual``,
``match``, plus a ``diagnostics`` entry in ``DataFrame.attrs``), this dialog
shows a summary tab (global agreement, Cohen's kappa, per-stage sensitivity /
precision / F1), a confusion-matrix heatmap and an overlaid hypnogram view.

The summary also surfaces the alignment diagnostics, in particular the case of
two strictly identical hypnograms: a 100 % agreement is then an artefact (the
manual scoring compared against itself) rather than a perfect model.

Replaces the Tk ``_show_detailed_comparison`` / ``_open_comparison_window``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from CESA.scoring_comparison import (
    STAGE_ORDER,
    cohens_kappa,
    confusion_matrix,
    kappa_label,
    macro_f1,
    present_stages,
    stage_metrics,
)
from CESA.scoring_report import (
    alignment_rows,
    interpretation_notes,
    merged_deep_sleep_accuracy,
    params_rows,
    top_confusions,
    write_report_csv,
)

logger = logging.getLogger(__name__)

_STAGE_ORDER = list(STAGE_ORDER)
_STAGE_Y = {"W": 5, "R": 4, "N1": 3, "N2": 2, "N3": 1, "U": 0}


class ScoringComparisonDialog(QtWidgets.QDialog):
    """Detailed comparison between automatic and manual scoring."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        merged: pd.DataFrame,
        epoch_len: float = 30.0,
        controller: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Comparaison auto vs manuel (detaillee)")
        self.resize(1000, 780)
        self._ctrl = controller
        self._set_data(merged, epoch_len)
        self._build_ui()

    # -- Data -----------------------------------------------------------

    def _set_data(self, merged: pd.DataFrame, epoch_len: Optional[float] = None) -> None:
        self._diagnostics: Dict[str, Any] = dict(merged.attrs.get("diagnostics") or {})
        self._merged = merged.reset_index(drop=True)
        if epoch_len is not None:
            self._epoch_len = float(epoch_len)
        self._epoch_len = float(
            self._diagnostics.get("comparison_epoch_seconds")
            or getattr(self, "_epoch_len", 30.0)
        )
        self._stages = present_stages(self._merged)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self._banner = QtWidgets.QLabel()
        self._banner.setWordWrap(True)
        self._banner.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._banner.setVisible(False)
        layout.addWidget(self._banner)

        self._alert = QtWidgets.QLabel()
        self._alert.setWordWrap(True)
        self._alert.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._alert.setVisible(False)
        self._alert.setStyleSheet(
            "background-color: #78350f; color: #fef3c7; padding: 8px; "
            "border-radius: 4px;"
        )
        layout.addWidget(self._alert)

        self._exclude_u = QtWidgets.QCheckBox(
            "Exclure les epoques non scorees / artefact (U) des deux cotes"
        )
        self._exclude_u.setChecked(bool(self._diagnostics.get("exclude_unscored", True)))
        self._exclude_u.setEnabled(self._ctrl is not None)
        if self._ctrl is None:
            self._exclude_u.setToolTip(
                "Recalcul indisponible : la comparaison a ete fournie sans controleur."
            )
        else:
            self._exclude_u.setToolTip(
                "Les epoques 'U' ne sont pas des stades de sommeil : les inclure fait "
                "chuter artificiellement la concordance."
            )
        self._exclude_u.toggled.connect(self._on_exclude_u_toggled)
        layout.addWidget(self._exclude_u)

        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        report_btn = QtWidgets.QPushButton("Exporter le rapport complet...")
        report_btn.setToolTip(
            "CSV avec les parametres du calcul, les metriques, la matrice de "
            "confusion, l'interpretation, puis le detail par epoque."
        )
        report_btn.clicked.connect(self._export_report)
        btns.addWidget(report_btn)
        save_btn = QtWidgets.QPushButton("Exporter les epoques (CSV)...")
        save_btn.clicked.connect(self._export_csv)
        btns.addWidget(save_btn)
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        self._refresh_tabs()

    def _refresh_tabs(self) -> None:
        current = self._tabs.currentIndex()
        self._tabs.clear()
        self._tabs.addTab(self._build_summary_tab(), "Resume")
        self._tabs.addTab(self._build_metrics_tab(), "Metriques par stade")
        self._tabs.addTab(self._build_confusion_tab(), "Matrice de confusion")
        self._tabs.addTab(self._build_hypno_tab(), "Hypnogrammes")
        self._tabs.addTab(self._build_params_tab(), "Parametres du calcul")
        if 0 <= current < self._tabs.count():
            self._tabs.setCurrentIndex(current)
        self._update_banner()

    def _on_exclude_u_toggled(self, checked: bool) -> None:
        if self._ctrl is None:
            return
        merged = self._ctrl.compare_scoring(exclude_unscored=checked)
        if merged is None or merged.empty:
            QtWidgets.QMessageBox.information(
                self, "Comparaison",
                "Aucune epoque comparable avec ce reglage.",
            )
            self._exclude_u.blockSignals(True)
            self._exclude_u.setChecked(not checked)
            self._exclude_u.blockSignals(False)
            return
        self._set_data(merged)
        self._refresh_tabs()

    # -- Banner ---------------------------------------------------------

    def _update_banner(self) -> None:
        diag = self._diagnostics
        messages: List[str] = []
        if diag.get("same_object"):
            messages.append(
                "Les deux scorings pointent sur le meme objet en memoire : "
                "la comparaison n'a aucune valeur."
            )
        elif diag.get("identical"):
            messages.append(
                "Les deux hypnogrammes sont <b>strictement identiques</b> : la "
                "concordance de 100 % est un artefact, pas une performance du modele. "
                "Relancez le scoring automatique (l'import d'un scoring manuel ne "
                "l'ecrase plus), ou verifiez que le scoring manuel a bien ete edite "
                "s'il a ete initialise depuis l'auto."
            )
        if diag.get("auto_is_fallback"):
            messages.append(
                "Aucun scoring automatique distinct n'etait en memoire : le scoring "
                "actif a ete utilise comme cote 'auto'. Relancez le scoring auto pour "
                "une comparaison fiable."
            )
        n_auto = diag.get("n_auto") or 0
        n_manual = diag.get("n_manual") or 0
        n_common = diag.get("n_common") or 0
        if n_auto and n_manual:
            coverage = n_common / max(n_auto, n_manual)
            if coverage < 0.8:
                messages.append(
                    f"Seules {n_common} epoques sont communes aux deux scorings "
                    f"(auto : {n_auto}, manuel : {n_manual}) : verifiez l'alignement "
                    "temporel et la duree d'epoque."
                )
        if messages:
            self._banner.setStyleSheet(
                "background-color: #7f1d1d; color: #fee2e2; padding: 8px; "
                "border-radius: 4px; font-weight: bold;"
            )
            self._banner.setText("Attention<br>- " + "<br>- ".join(messages))
            self._banner.setVisible(True)
        else:
            self._banner.setVisible(False)
        self._update_alert()

    def _update_alert(self) -> None:
        """Amber banner for a stage the model never produced.

        This is the single most useful signal when the agreement is low: it
        separates a signal or configuration problem from a plain disagreement.
        """
        never = list(self._diagnostics.get("never_predicted") or [])
        if not never:
            self._alert.setVisible(False)
            return
        detail = ", ".join(
            f"<b>{stage}</b> ({int((self._merged['manual'] == stage).sum())} epoques "
            "scorees a la main)"
            for stage in never
        )
        self._alert.setText(
            f"Stade(s) jamais predit(s) par le modele : {detail}.<br>"
            "Un stade totalement absent de la sortie n'est pas un desaccord de "
            "frontiere. Verifiez l'amplitude du signal (unites V / uV, "
            "calibration) et la derivation EEG choisie : N3 se definit par un "
            "critere d'amplitude (ondes lentes > 75 uV), un signal trop faible "
            "empeche sa detection. Voir l'onglet Parametres du calcul."
        )
        self._alert.setVisible(True)

    # -- Summary --------------------------------------------------------

    def _build_summary_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        n_total = len(self._merged)
        n_match = int(self._merged["match"].sum())
        acc = n_match / n_total * 100 if n_total else 0.0
        kappa = cohens_kappa(self._merged["auto"], self._merged["manual"])
        metrics = stage_metrics(self._merged, self._stages)

        deep = merged_deep_sleep_accuracy(self._merged)
        header = QtWidgets.QLabel(
            f"<h3>Concordance globale: {n_match}/{n_total} ({acc:.1f}%)</h3>"
            f"<p>Cohen's kappa: <b>{kappa:.3f}</b> ({kappa_label(kappa)})"
            f" &nbsp;|&nbsp; F1 macro: <b>{macro_f1(metrics):.3f}</b>"
            f" &nbsp;|&nbsp; N2+N3 fusionnes: <b>{deep:.1f}%</b></p>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(header)

        diag_box = QtWidgets.QGroupBox("Diagnostic de l'alignement")
        diag_lay = QtWidgets.QVBoxLayout(diag_box)
        diag_label = QtWidgets.QLabel(self._diagnostics_html())
        diag_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        diag_label.setWordWrap(True)
        diag_lay.addWidget(diag_label)
        lay.addWidget(diag_box)

        dist_box = QtWidgets.QGroupBox("Repartition des stades (epoques comparees)")
        dist_lay = QtWidgets.QVBoxLayout(dist_box)
        dist_table = QtWidgets.QTableWidget(len(self._stages), 3)
        dist_table.setHorizontalHeaderLabels(["Stade", "Manuel", "Auto"])
        dist_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        for r, stage in enumerate(self._stages):
            n_man = metrics[stage]["support"]
            n_aut = metrics[stage]["n_auto"]
            values = [
                stage,
                f"{int(n_man)} ({n_man / n_total * 100:.1f}%)" if n_total else "0",
                f"{int(n_aut)} ({n_aut / n_total * 100:.1f}%)" if n_total else "0",
            ]
            for c, val in enumerate(values):
                dist_table.setItem(r, c, QtWidgets.QTableWidgetItem(val))
        dist_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        dist_lay.addWidget(dist_table)
        lay.addWidget(dist_box, stretch=1)
        return w

    def _diagnostics_html(self) -> str:
        diag = self._diagnostics
        if not diag:
            return "<i>Aucun diagnostic disponible.</i>"
        method = diag.get("auto_method") or "inconnu"
        source = diag.get("manual_source") or "inconnue"
        rows = [
            f"Scoring auto : backend <b>{method}</b>, "
            f"{diag.get('n_auto', 0)} epoques de "
            f"{diag.get('auto_epoch_seconds', 30.0):.0f} s",
            f"Scoring manuel : source <b>{source}</b>, "
            f"{diag.get('n_manual', 0)} epoques de "
            f"{diag.get('manual_epoch_seconds', 30.0):.0f} s",
            f"Grille de comparaison : {diag.get('comparison_epoch_seconds', 30.0):.0f} s "
            f"&nbsp;|&nbsp; epoques communes : {diag.get('n_common', 0)} "
            f"&nbsp;|&nbsp; comparees : {diag.get('n_compared', 0)}",
        ]
        excluded = []
        if diag.get("n_excluded_u"):
            excluded.append(f"{diag['n_excluded_u']} epoques U exclues")
        if diag.get("n_excluded_night"):
            excluded.append(
                f"{diag['n_excluded_night']} epoques hors fenetre de nuit exclues"
            )
        if excluded:
            rows.append("Exclusions : " + ", ".join(excluded))
        return "<br>".join(rows)

    # -- Per-stage metrics ---------------------------------------------

    def _build_metrics_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel(
            "Le scoring manuel sert de reference. Sensibilite = epoques du stade "
            "retrouvees par le modele ; precision = justesse quand le modele "
            "annonce ce stade."
        ))

        headers = [
            "Stade", "Support (manuel)", "Sensibilite", "Precision",
            "Specificite", "F1", "VP", "FP", "FN",
        ]
        metrics = stage_metrics(self._merged, self._stages)
        table = QtWidgets.QTableWidget(len(self._stages), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )

        def fmt(value: float) -> str:
            return "-" if value != value else f"{value * 100:.1f}%"

        for r, stage in enumerate(self._stages):
            m = metrics[stage]
            values = [
                stage,
                str(int(m["support"])),
                fmt(m["recall"]),
                fmt(m["precision"]),
                fmt(m["specificity"]),
                fmt(m["f1"]),
                str(int(m["tp"])),
                str(int(m["fp"])),
                str(int(m["fn"])),
            ]
            for c, val in enumerate(values):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(val))
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(table, stretch=1)
        return w

    # -- Parameters of the run -----------------------------------------

    def _build_params_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        inner_lay = QtWidgets.QVBoxLayout(inner)

        params = params_rows(self._diagnostics)
        if params:
            inner_lay.addWidget(self._rows_group(
                "Parametres du scoring automatique", params
            ))
        else:
            missing = QtWidgets.QLabel(
                "Les parametres du run automatique ne sont pas disponibles : ce "
                "scoring a ete produit avant l'enregistrement des parametres, ou "
                "importe depuis un fichier. Relancez le scoring automatique pour "
                "renseigner cette section."
            )
            missing.setWordWrap(True)
            inner_lay.addWidget(missing)

        align = alignment_rows(self._diagnostics)
        if align:
            inner_lay.addWidget(self._rows_group(
                "Alignement des deux scorings", align
            ))

        confusions = top_confusions(self._merged, limit=5)
        if confusions:
            rows = [
                (
                    f"Manuel {c['manual']} classe {c['auto']}",
                    f"{c['n']} epoques ({c['pct_errors']:.1f}% des desaccords, "
                    f"{c['pct_total']:.1f}% du total)",
                )
                for c in confusions
            ]
            inner_lay.addWidget(self._rows_group("Principales confusions", rows))

        notes = interpretation_notes(self._merged, self._diagnostics)
        if notes:
            box = QtWidgets.QGroupBox("Interpretation")
            box_lay = QtWidgets.QVBoxLayout(box)
            label = QtWidgets.QLabel("<ul><li>" + "</li><li>".join(notes) + "</li></ul>")
            label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            label.setWordWrap(True)
            box_lay.addWidget(label)
            inner_lay.addWidget(box)

        inner_lay.addStretch()
        scroll.setWidget(inner)
        lay.addWidget(scroll)
        return w

    @staticmethod
    def _rows_group(title: str, rows: List[Any]) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(box)
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Parametre", "Valeur"])
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        table.verticalHeader().setVisible(False)
        for r, (label, value) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(label)))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(value)))
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.resizeRowsToContents()
        height = table.horizontalHeader().height() + 2
        for r in range(table.rowCount()):
            height += table.rowHeight(r)
        table.setMinimumHeight(min(height + 4, 420))
        lay.addWidget(table)
        return box

    # -- Confusion matrix ----------------------------------------------

    def _build_confusion_tab(self) -> QtWidgets.QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        fig = Figure(figsize=(7, 6), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(canvas, stretch=1)

        stages = self._stages or _STAGE_ORDER
        cm = confusion_matrix(self._merged, stages)

        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(stages)))
        ax.set_yticks(range(len(stages)))
        ax.set_xticklabels(stages)
        ax.set_yticklabels(stages)
        ax.set_xlabel("Manuel")
        ax.set_ylabel("Automatique")
        ax.set_title("Matrice de confusion (nb d'epoques)")
        thresh = cm.max() / 2.0 if cm.max() else 0.5
        for i in range(len(stages)):
            for j in range(len(stages)):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        canvas.draw()
        self._cm = cm
        return w

    # -- Hypnogram overlay ---------------------------------------------

    def _build_hypno_tab(self) -> QtWidgets.QWidget:
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        fig = Figure(figsize=(10, 5), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, w)
        lay.addWidget(toolbar)
        lay.addWidget(canvas, stretch=1)

        df = self._merged.sort_values("epoch")
        epochs = df["epoch"].to_numpy() * (self._epoch_len / 3600.0)
        auto_y = [_STAGE_Y.get(s, 0) for s in df["auto"]]
        manual_y = [_STAGE_Y.get(s, 0) for s in df["manual"]]

        ax = fig.add_subplot(111)
        ax.step(epochs, auto_y, where="post", label="Auto", color="#89b4fa", alpha=0.9)
        ax.step(epochs, manual_y, where="post", label="Manuel", color="#f38ba8",
                alpha=0.7, linestyle="--")
        mism = ~df["match"].to_numpy()
        if mism.any():
            ax.scatter(
                epochs[mism], np.array(auto_y)[mism], color="#fab387",
                s=12, zorder=3, label="Desaccords",
            )
        ax.set_yticks(list(_STAGE_Y.values()))
        ax.set_yticklabels(list(_STAGE_Y.keys()))
        ax.set_xlabel("Temps (h)")
        ax.set_title("Hypnogrammes superposes")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        canvas.draw()
        return w

    # -- Export ---------------------------------------------------------

    def _export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter comparaison", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._merged.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Export", f"Comparaison exportee:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")

    def _export_report(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter le rapport complet", "rapport_comparaison.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        try:
            write_report_csv(path, self._merged, self._diagnostics)
        except Exception as exc:
            logger.exception("Export du rapport de comparaison echoue")
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")
            return
        QtWidgets.QMessageBox.information(
            self, "Export",
            f"Rapport complet exporte:\n{path}\n\n"
            "Il contient les parametres du calcul, les metriques globales et par "
            "stade, la matrice de confusion, les principales confusions, "
            "l'interpretation et le detail par epoque.",
        )
