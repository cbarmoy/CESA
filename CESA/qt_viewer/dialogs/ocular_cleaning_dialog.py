"""Qt dialog: non-destructive EOG→EEG ocular cleaning with before/after preview.

Cleaning is computed in memory only. Apply/restore are delegated to callbacks
provided by the main window (never writes recording files).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from CESA.filters import detect_signal_type
from CESA.ocular_cleaning import (
    clean_channels,
    ocular_bandpower_ratio,
    slice_signal,
)

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]

ApplyCallback = Callable[[Dict[str, np.ndarray]], None]
RestoreCallback = Callable[[], None]


class OcularCleaningDialog(QtWidgets.QDialog):
    """Preview and apply EOG regression cleaning on EEG channels."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: Sequence[str],
        *,
        channel_types: Optional[Dict[str, str]] = None,
        preview_t0: float = 0.0,
        preview_duration: float = 30.0,
        has_active_cleaning: bool = False,
        on_apply: Optional[ApplyCallback] = None,
        on_restore: Optional[RestoreCallback] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nettoyage oculaire (EOG)")
        self.resize(1000, 720)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._types = dict(channel_types or {})
        for ch in self._channels:
            if ch not in self._types:
                self._types[ch] = detect_signal_type(ch)
        self._preview_t0 = max(0.0, float(preview_t0))
        self._preview_duration = max(1.0, float(preview_duration))
        self._on_apply = on_apply
        self._on_restore = on_restore
        self._preview_done = False
        self._last_cleaned: Dict[str, np.ndarray] = {}
        self._preview_eeg: Optional[str] = None
        self._build_ui()
        self._btn_restore.setEnabled(bool(has_active_cleaning))
        if not self._eog_names():
            self._status.setText(
                "Aucun canal EOG detecte. Selectionnez au moins une reference EOG."
            )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Regression EOG → EEG en memoire uniquement — le fichier d'origine "
            "n'est jamais modifie. Lancez un apercu avant d'appliquer."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        lists = QtWidgets.QHBoxLayout()
        eeg_box = QtWidgets.QGroupBox("Canaux EEG a nettoyer")
        eeg_l = QtWidgets.QVBoxLayout(eeg_box)
        self._eeg_list = QtWidgets.QListWidget()
        self._eeg_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        for ch in self._eeg_names():
            self._eeg_list.addItem(ch)
        self._eeg_list.selectAll()
        eeg_l.addWidget(self._eeg_list)
        lists.addWidget(eeg_box)

        eog_box = QtWidgets.QGroupBox("References EOG")
        eog_l = QtWidgets.QVBoxLayout(eog_box)
        self._eog_list = QtWidgets.QListWidget()
        self._eog_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        for ch in self._eog_names():
            self._eog_list.addItem(ch)
        self._eog_list.selectAll()
        # Also allow picking any channel as EOG if auto-detect missed some
        other = [c for c in self._channels if c not in self._eog_names()]
        for ch in other:
            item = QtWidgets.QListWidgetItem(f"{ch} (manuel)")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, ch)
            self._eog_list.addItem(item)
        eog_l.addWidget(self._eog_list)
        lists.addWidget(eog_box)
        layout.addLayout(lists)

        win = QtWidgets.QHBoxLayout()
        win.addWidget(QtWidgets.QLabel("Apercu — debut:"))
        self._t0 = QtWidgets.QDoubleSpinBox()
        self._t0.setRange(0.0, 1e7)
        self._t0.setDecimals(1)
        self._t0.setSuffix(" s")
        self._t0.setValue(self._preview_t0)
        win.addWidget(self._t0)
        win.addWidget(QtWidgets.QLabel("duree:"))
        self._dur = QtWidgets.QDoubleSpinBox()
        self._dur.setRange(1.0, 600.0)
        self._dur.setDecimals(1)
        self._dur.setSuffix(" s")
        self._dur.setValue(self._preview_duration)
        win.addWidget(self._dur)
        win.addWidget(QtWidgets.QLabel("EEG apercu:"))
        self._preview_ch = QtWidgets.QComboBox()
        self._preview_ch.addItems(self._eeg_names() or self._channels)
        win.addWidget(self._preview_ch)
        win.addStretch()
        layout.addLayout(win)

        plot_w = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plot_w)
        self._fig = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        pv.addWidget(NavigationToolbar2QT(self._canvas, plot_w))
        pv.addWidget(self._canvas, stretch=1)
        layout.addWidget(plot_w, stretch=1)

        self._metric = QtWidgets.QLabel("Puissance oculaire (0.3–4 Hz) : —")
        layout.addWidget(self._metric)

        btns = QtWidgets.QHBoxLayout()
        self._btn_preview = QtWidgets.QPushButton("Apercu")
        self._btn_preview.clicked.connect(self._run_preview)
        btns.addWidget(self._btn_preview)
        self._btn_apply = QtWidgets.QPushButton("Appliquer au viewer")
        self._btn_apply.setEnabled(False)
        self._btn_apply.clicked.connect(self._apply)
        btns.addWidget(self._btn_apply)
        self._btn_restore = QtWidgets.QPushButton("Restaurer l'original")
        self._btn_restore.clicked.connect(self._restore)
        btns.addWidget(self._btn_restore)
        btns.addStretch()
        close = QtWidgets.QPushButton("Fermer")
        close.clicked.connect(self.reject)
        btns.addWidget(close)
        layout.addLayout(btns)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------

    def _eeg_names(self) -> List[str]:
        eeg = [c for c in self._channels if self._types.get(c, "") == "eeg"]
        if eeg:
            return eeg
        # Fallback: anything that is not clearly EOG/EMG/ECG
        skip = {"eog", "emg", "ecg"}
        return [c for c in self._channels if self._types.get(c, "") not in skip]

    def _eog_names(self) -> List[str]:
        return [c for c in self._channels if self._types.get(c, "") == "eog"]

    def _selected_eeg(self) -> List[str]:
        return [i.text() for i in self._eeg_list.selectedItems()]

    def _selected_eog(self) -> List[str]:
        names: List[str] = []
        for item in self._eog_list.selectedItems():
            manual = item.data(QtCore.Qt.ItemDataRole.UserRole)
            names.append(str(manual) if manual else item.text())
        return names

    # ------------------------------------------------------------------
    # Preview / apply
    # ------------------------------------------------------------------

    def _run_preview(self) -> None:
        eeg_names = self._selected_eeg()
        eog_names = self._selected_eog()
        if not eog_names:
            self._status.setText("Selectionnez au moins un canal EOG.")
            self._btn_apply.setEnabled(False)
            return
        if not eeg_names:
            self._status.setText("Selectionnez au moins un canal EEG.")
            self._btn_apply.setEnabled(False)
            return

        preview_ch = self._preview_ch.currentText()
        if preview_ch not in eeg_names:
            preview_ch = eeg_names[0]
            idx = self._preview_ch.findText(preview_ch)
            if idx >= 0:
                self._preview_ch.setCurrentIndex(idx)

        t0 = float(self._t0.value())
        dur = float(self._dur.value())
        data_eeg, fs = self._signals[preview_ch]
        fs = float(fs)
        before = slice_signal(data_eeg, fs, t0, dur)
        if before.size < 8:
            self._status.setText("Fenetre d'apercu trop courte ou hors signal.")
            self._btn_apply.setEnabled(False)
            return

        # Build windowed signal dict for clean_channels
        win_signals: Dict[str, Signal] = {}
        for name in set(eeg_names) | set(eog_names):
            arr, ch_fs = self._signals[name]
            win_signals[name] = (slice_signal(arr, float(ch_fs), t0, dur), float(ch_fs))

        try:
            cleaned_win = clean_channels(win_signals, eeg_names, eog_names, sfreq=fs)
        except Exception as exc:
            logger.error("Ocular preview failed: %s", exc, exc_info=True)
            self._status.setText(f"Erreur apercu: {exc}")
            self._btn_apply.setEnabled(False)
            return

        after = cleaned_win.get(preview_ch)
        if after is None or after.size < 8:
            self._status.setText("Echec du nettoyage sur le canal d'apercu.")
            self._btn_apply.setEnabled(False)
            return

        ratio = ocular_bandpower_ratio(before, after, fs)
        pct = ratio * 100.0
        sign = "+" if pct >= 0 else ""
        self._metric.setText(
            f"Puissance oculaire (0.3–4 Hz) : {sign}{pct:.1f} % "
            f"(negatif = reduction)"
        )

        eog_plot = None
        eog_label = eog_names[0]
        if eog_label in win_signals:
            eog_plot = win_signals[eog_label][0]

        n = min(before.size, after.size)
        t = np.arange(n) / fs + t0
        self._fig.clear()
        ax0 = self._fig.add_subplot(311)
        ax1 = self._fig.add_subplot(312, sharex=ax0)
        ax2 = self._fig.add_subplot(313, sharex=ax0)
        ax0.plot(t, before[:n], color="#89b4fa", lw=0.7)
        ax0.set_ylabel("EEG avant")
        ax0.set_title(f"{preview_ch} — avant / apres nettoyage oculaire")
        ax0.grid(True, alpha=0.3)
        ax1.plot(t, after[:n], color="#a6e3a1", lw=0.7)
        ax1.set_ylabel("EEG apres")
        ax1.grid(True, alpha=0.3)
        if eog_plot is not None and eog_plot.size:
            ne = min(n, eog_plot.size)
            ax2.plot(t[:ne], eog_plot[:ne], color="#fab387", lw=0.7)
            ax2.set_ylabel(f"EOG ({eog_label})")
        else:
            ax2.set_ylabel("EOG")
        ax2.set_xlabel("Temps (s)")
        ax2.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._canvas.draw()

        # Full-night cleaned arrays ready for apply (same params)
        try:
            self._last_cleaned = clean_channels(
                self._signals, eeg_names, eog_names, sfreq=fs
            )
        except Exception as exc:
            logger.error("Full ocular clean failed: %s", exc, exc_info=True)
            self._status.setText(f"Erreur calcul complet: {exc}")
            self._btn_apply.setEnabled(False)
            return

        self._preview_done = True
        self._preview_eeg = preview_ch
        self._btn_apply.setEnabled(bool(self._last_cleaned) and self._on_apply is not None)
        self._status.setText(
            f"Apercu OK — {len(self._last_cleaned)} canal(aux) prets. "
            "Fichier d'origine non modifie."
        )

    def _apply(self) -> None:
        if not self._preview_done or not self._last_cleaned:
            self._status.setText("Lancez d'abord un apercu.")
            return
        if self._on_apply is None:
            return
        try:
            self._on_apply(dict(self._last_cleaned))
            self._btn_restore.setEnabled(True)
            self._status.setText(
                "Nettoyage applique au viewer (memoire) — fichier non modifie."
            )
            QtWidgets.QMessageBox.information(
                self,
                "Nettoyage oculaire",
                "Nettoyage applique en memoire.\n"
                "Le fichier d'enregistrement n'a pas ete modifie.",
            )
        except Exception as exc:
            logger.error("Apply ocular cleaning failed: %s", exc, exc_info=True)
            self._status.setText(f"Erreur application: {exc}")

    def _restore(self) -> None:
        if self._on_restore is None:
            return
        try:
            self._on_restore()
            self._btn_restore.setEnabled(False)
            self._preview_done = False
            self._last_cleaned = {}
            self._btn_apply.setEnabled(False)
            self._metric.setText("Puissance oculaire (0.3–4 Hz) : —")
            self._status.setText("Signaux originaux restaures (memoire).")
        except Exception as exc:
            logger.error("Restore ocular cleaning failed: %s", exc, exc_info=True)
            self._status.setText(f"Erreur restauration: {exc}")
