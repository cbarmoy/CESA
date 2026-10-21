"""Qt dialog: circular figure of the SO-spindle coupling of one recording.

Shows the distribution of the SO phases sampled at the spindle peaks, channel
by channel, with the mean resultant vector drawn only when enough spindles
were coupled.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from PySide6 import QtWidgets

logger = logging.getLogger(__name__)


def phases_by_channel(events: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Coupled spindle phases of each channel, from the UI event dicts."""
    out: Dict[str, List[float]] = {}
    for event in events:
        meta = event.get("metadata") or {}
        phase = meta.get("so_phase_rad")
        if phase is None:
            continue
        channel = str(event.get("channel") or meta.get("channel") or "")
        out.setdefault(channel, []).append(float(phase))
    return out


class CouplingPlotDialog(QtWidgets.QDialog):
    """Polar view of the preferred phase, with PNG/PDF/SVG export."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        *,
        events: Sequence[Dict[str, Any]],
        coupling: Sequence[Dict[str, Any]] = (),
        min_coupled: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Couplage SO-fuseau — phase preferee")
        self.resize(760, 760)

        from CESA.sleep_pipeline.so_spindle_coupling import MIN_COUPLED_EVENTS

        self._min_coupled = int(
            MIN_COUPLED_EVENTS if min_coupled is None else min_coupled
        )
        self._phases = phases_by_channel(events)
        self._stats = {
            str(row.get("channel", "")): dict(row) for row in (coupling or [])
        }
        self._build_ui()
        self._draw()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Canal:"))
        self._channel_combo = QtWidgets.QComboBox()
        self._channel_combo.addItems(sorted(self._phases) or ["(aucun)"])
        self._channel_combo.currentIndexChanged.connect(self._draw)
        top.addWidget(self._channel_combo)

        top.addWidget(QtWidgets.QLabel("Secteurs:"))
        self._bins_spin = QtWidgets.QSpinBox()
        self._bins_spin.setRange(6, 72)
        self._bins_spin.setSingleStep(6)
        self._bins_spin.setValue(18)
        self._bins_spin.valueChanged.connect(self._draw)
        top.addWidget(self._bins_spin)
        top.addStretch(1)
        layout.addLayout(top)

        self._fig = Figure(figsize=(7.0, 7.0), dpi=100)
        self._fig.set_facecolor("#1E1E2E")
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas, stretch=1)

        self._status = QtWidgets.QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        save_btn = QtWidgets.QPushButton("Enregistrer figure")
        save_btn.clicked.connect(self._save_figure)
        buttons.addWidget(save_btn)
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------

    def _draw(self) -> None:
        from CESA.coupling_plots import plot_phase_rose

        channel = self._channel_combo.currentText()
        phases = self._phases.get(channel) or []
        self._fig.clear()
        if not phases:
            self._status.setText(
                "Aucun fuseau couple sur ce canal : rien a tracer."
            )
            self._canvas.draw()
            return
        try:
            plot_phase_rose(
                phases,
                channel=channel,
                bins=int(self._bins_spin.value()),
                stats=self._stats.get(channel),
                min_coupled=self._min_coupled,
                dark=True,
                fig=self._fig,
            )
        except Exception as exc:
            logger.error("Coupling figure failed: %s", exc, exc_info=True)
            self._status.setText(f"Figure impossible: {exc}")
            self._canvas.draw()
            return
        self._status.setText(
            f"{len(phases)} fuseau(x) couple(s) sur {channel}. "
            "0 rad est le pic positif de l'onde lente."
        )
        self._canvas.draw()

    def _save_figure(self) -> None:
        channel = self._channel_combo.currentText().replace("/", "-")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Enregistrer figure",
            f"couplage_{channel}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        try:
            self._fig.savefig(
                path, dpi=200, bbox_inches="tight",
                facecolor=self._fig.get_facecolor(),
            )
        except Exception as exc:
            logger.error("Figure export failed: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Export echoue:\n{exc}")
            return
        self._status.setText(f"Figure ecrite: {path}")
