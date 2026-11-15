"""Qt dialog: circular and event-locked figures of the SO-spindle coupling."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from PySide6 import QtCore, QtWidgets

logger = logging.getLogger(__name__)

TYPE_PHASE = "phase"
TYPE_AVERAGE = "moyenne"
TYPE_TFR = "tfr"

_TYPE_LABELS = {
    TYPE_PHASE: "Phase preferee (rose)",
    TYPE_AVERAGE: "Moyenne fuseau-verrouillee",
    TYPE_TFR: "TFR creux onde lente",
}
_TYPE_TIPS = {
    TYPE_PHASE: (
        "Histogramme circulaire des phases de l'onde lente au pic des fuseaux. "
        "0° = up-state, ±180° = creux. La fleche est le vecteur resultant (R)."
    ),
    TYPE_AVERAGE: (
        "Moyenne de l'EEG (±2,5 s) alignee sur le pic des fuseaux, z-normalisee, "
        "avec l'onde lente superposee. Calcule maintenant pour le canal choisi."
    ),
    TYPE_TFR: (
        "Spectrogramme 5–30 Hz aligne sur chaque creux d'onde lente, "
        "z-score vs baseline −2,0 a −1,5 s. Calcule maintenant pour le canal choisi."
    ),
}


def phases_by_channel(events: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Coupled spindle phases of each channel, from the UI event dicts."""
    out: Dict[str, List[float]] = {}
    for event in events:
        meta = event.get("metadata") or {}
        if str(meta.get("detector") or event.get("detector") or "") == "yasa":
            continue
        phase = meta.get("so_phase_rad")
        if phase is None:
            continue
        channel = str(event.get("channel") or meta.get("channel") or "")
        if not channel:
            continue
        out.setdefault(channel, []).append(float(phase))
    return out


class CouplingPlotDialog(QtWidgets.QDialog):
    """Figures of the coupling, computed on demand for the selected channel."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        *,
        events: Sequence[Dict[str, Any]],
        coupling: Sequence[Dict[str, Any]] = (),
        enabled: Optional[Set[str]] = None,
        get_signal: Optional[Callable[[str], Tuple[Any, float]]] = None,
        min_coupled: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Couplage SO-fuseau — figures")
        self.resize(820, 760)

        from CESA.sleep_pipeline.so_spindle_coupling import MIN_COUPLED_EVENTS

        self._min_coupled = int(
            MIN_COUPLED_EVENTS if min_coupled is None else min_coupled
        )
        self._phases = phases_by_channel(events)
        self._events = list(events)
        self._stats = {
            str(row.get("channel", "")): dict(row) for row in (coupling or [])
        }
        self._enabled = set(enabled) if enabled else {TYPE_PHASE}
        self._get_signal = get_signal
        self._build_ui()
        self._draw()

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Canal:"))
        self._channel_combo = QtWidgets.QComboBox()
        channels = sorted(self._phases) or sorted(self._stats) or ["(aucun)"]
        self._channel_combo.addItems(channels)
        self._channel_combo.currentIndexChanged.connect(self._draw)
        top.addWidget(self._channel_combo)

        top.addWidget(QtWidgets.QLabel("Figure:"))
        self._type_combo = QtWidgets.QComboBox()
        for key in (TYPE_PHASE, TYPE_AVERAGE, TYPE_TFR):
            if key in self._enabled:
                self._type_combo.addItem(_TYPE_LABELS[key], key)
                idx = self._type_combo.count() - 1
                self._type_combo.setItemData(
                    idx, _TYPE_TIPS[key], QtCore.Qt.ItemDataRole.ToolTipRole,
                )
        if self._type_combo.count() == 0:
            self._type_combo.addItem(_TYPE_LABELS[TYPE_PHASE], TYPE_PHASE)
        self._type_combo.setToolTip(
            "Types coches avant la detection. Survolez une entree pour le detail."
        )
        self._type_combo.currentIndexChanged.connect(self._draw)
        top.addWidget(self._type_combo)

        bins_lbl = QtWidgets.QLabel("Secteurs:")
        bins_lbl.setToolTip(
            "Nombre de secteurs de la rose polaire (phase seulement). 18 = 20° par secteur."
        )
        top.addWidget(bins_lbl)
        self._bins_spin = QtWidgets.QSpinBox()
        self._bins_spin.setRange(6, 72)
        self._bins_spin.setSingleStep(6)
        self._bins_spin.setValue(18)
        self._bins_spin.setToolTip(bins_lbl.toolTip())
        self._bins_spin.valueChanged.connect(self._draw)
        top.addWidget(self._bins_spin)
        top.addStretch(1)
        layout.addLayout(top)

        self._fig = Figure(figsize=(6.4, 6.4), facecolor="#1E1E2E")
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas, 1)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

        buttons = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Enregistrer figure")
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        close = QtWidgets.QPushButton("Fermer")
        close.clicked.connect(self.accept)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _kind(self) -> str:
        data = self._type_combo.currentData()
        return str(data) if data else TYPE_PHASE

    def _draw(self) -> None:
        self._fig.clear()
        channel = self._channel_combo.currentText()
        kind = self._kind()
        self._bins_spin.setEnabled(kind == TYPE_PHASE)
        try:
            if kind == TYPE_PHASE:
                self._draw_phase(channel)
            elif kind == TYPE_AVERAGE:
                self._draw_event(channel, TYPE_AVERAGE)
            else:
                self._draw_event(channel, TYPE_TFR)
            self._status.setText("")
        except Exception as exc:
            logger.error("Coupling figure failed: %s", exc, exc_info=True)
            self._status.setText(str(exc))
        self._canvas.draw_idle()

    def _draw_phase(self, channel: str) -> None:
        from CESA.coupling_plots import plot_phase_rose

        phases = self._phases.get(channel) or []
        plot_phase_rose(
            phases, channel=channel, bins=self._bins_spin.value(),
            stats=self._stats.get(channel), min_coupled=self._min_coupled,
            dark=True, fig=self._fig,
        )

    def _draw_event(self, channel: str, kind: str) -> None:
        if kind not in self._enabled:
            raise RuntimeError(
                "Cette figure n'a pas ete demandee dans les parametres. "
                "Cochez-la et relancez la detection."
            )
        if self._get_signal is None:
            raise RuntimeError("Signal du canal indisponible.")
        signal, sfreq = self._get_signal(channel)
        from CESA.coupling_event_plots import (
            event_indices_from_dicts,
            so_trough_tfr,
            spindle_locked_average,
        )

        troughs, peaks = event_indices_from_dicts(self._events, sfreq, channel)
        if kind == TYPE_AVERAGE:
            spindle_locked_average(signal, sfreq, peaks, fig=self._fig)
        else:
            so_trough_tfr(signal, sfreq, troughs, fig=self._fig)

    def _save(self) -> None:
        path, selected = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer la figure", "coupling_figure.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        try:
            self._fig.savefig(path, dpi=200, bbox_inches="tight",
                              facecolor=self._fig.get_facecolor())
            self._status.setText(f"Enregistre : {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Erreur", str(exc))
