"""Qt dialog: non-destructive MNE ICA cleaning with preview / apply / restore."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from CESA.filters import detect_signal_type
from CESA.ica_cleaning import (
    IcaFitResult,
    component_labels,
    component_spectrum,
    component_stats,
    fit_ica,
    get_sources,
    preview_exclude,
)

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]
ApplyCallback = Callable[[Dict[str, np.ndarray]], None]
RestoreCallback = Callable[[], None]


class IcaCleaningDialog(QtWidgets.QDialog):
    """Fit ICA, select components to exclude, preview, apply (memory only)."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: Sequence[str],
        *,
        channel_types: Optional[Dict[str, str]] = None,
        sfreq: float = 200.0,
        preview_t0: float = 0.0,
        preview_duration: float = 30.0,
        has_active_cleaning: bool = False,
        filters_on: bool = True,
        ocular_cleaning_active: bool = False,
        on_apply: Optional[ApplyCallback] = None,
        on_restore: Optional[RestoreCallback] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nettoyage ICA")
        self.resize(1100, 780)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._types = dict(channel_types or {})
        for ch in self._channels:
            if ch not in self._types:
                self._types[ch] = detect_signal_type(ch)
        self._sfreq = float(sfreq)
        self._preview_t0 = max(0.0, float(preview_t0))
        self._preview_duration = max(1.0, float(preview_duration))
        self._on_apply = on_apply
        self._on_restore = on_restore
        self._filters_on = bool(filters_on)
        self._ocular_active = bool(ocular_cleaning_active)
        self._has_ica = bool(has_active_cleaning)
        self._fit: Optional[IcaFitResult] = None
        self._stats: List[dict] = []
        self._preview_done = False
        self._last_cleaned: Dict[str, np.ndarray] = {}
        self._build_ui()
        self._btn_restore.setEnabled(bool(has_active_cleaning))
        self._ic_list.currentRowChanged.connect(self._on_ic_selected)

    def _eeg_names(self) -> List[str]:
        eeg = [c for c in self._channels if self._types.get(c, "") == "eeg"]
        return eeg or list(self._channels)

    def _all_fit_candidates(self) -> List[str]:
        return list(self._channels)

    def _ref_channel(self, kind: str) -> Optional[np.ndarray]:
        """Return first channel matching type *kind* (eog/ecg), full length µV."""
        for ch in self._channels:
            if self._types.get(ch, "") == kind and ch in self._signals:
                return np.asarray(self._signals[ch][0], dtype=float)
        tokens = ("eog", "eye", "loc", "roc") if kind == "eog" else ("ecg", "ekg", "heart")
        for ch in self._channels:
            low = ch.lower()
            if any(t in low for t in tokens) and ch in self._signals:
                return np.asarray(self._signals[ch][0], dtype=float)
        return None

    def _sync_window_from_viewer(self) -> None:
        """Align t0/duree with the currently displayed viewer window."""
        parent = self.parent()
        viewer = getattr(parent, "_viewer", None) if parent is not None else None
        if viewer is None:
            return
        try:
            t0 = float(getattr(viewer, "start_s", self._t0.value()) or 0.0)
            dur = float(getattr(viewer, "duration_s", self._dur.value()) or 30.0)
        except Exception:
            return
        self._t0.blockSignals(True)
        self._dur.blockSignals(True)
        self._t0.setValue(max(0.0, t0))
        self._dur.setValue(max(1.0, dur))
        self._t0.blockSignals(False)
        self._dur.blockSignals(False)

    def _window_bounds(self) -> Tuple[float, float]:
        self._sync_window_from_viewer()
        t0 = float(self._t0.value())
        dur = float(self._dur.value())
        return t0, t0 + dur

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)

        self._banner = QtWidgets.QLabel()
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet(
            "font-size: 12px; padding: 8px 10px; background: #313244; "
            "border-radius: 6px; color: #CDD6F4;"
        )
        self._banner.setText(
            f"Signaux viewer : filtres {'ON' if self._filters_on else 'OFF'}  ·  "
            f"nettoyage oculaire actif : {'oui' if self._ocular_active else 'non'}  ·  "
            f"ICA deja appliquee : {'oui' if self._has_ica else 'non'}"
        )
        layout.addWidget(self._banner)

        info = QtWidgets.QLabel(
            "ICA MNE en memoire uniquement — le fichier n'est jamais modifie. "
            "Fit → inspecter IC (trace + spectre sur la fenetre viewer) → selection → "
            "Apercu EEG nettoye obligatoire avant Appliquer. "
            "Topographie non disponible (montage PSG)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        top = QtWidgets.QHBoxLayout()
        ch_box = QtWidgets.QGroupBox("Canaux pour le fit (tous listes)")
        ch_l = QtWidgets.QVBoxLayout(ch_box)
        self._ch_list = QtWidgets.QListWidget()
        self._ch_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        eeg_set = set(self._eeg_names())
        for ch in self._all_fit_candidates():
            ctype = self._types.get(ch, detect_signal_type(ch))
            item = QtWidgets.QListWidgetItem(f"{ch}  [{ctype}]")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, ch)
            self._ch_list.addItem(item)
            item.setSelected(ch in eeg_set)
        if not self._ch_list.selectedItems() and self._ch_list.count():
            for i in range(min(4, self._ch_list.count())):
                self._ch_list.item(i).setSelected(True)
        ch_l.addWidget(self._ch_list)
        top.addWidget(ch_box, 1)

        ic_box = QtWidgets.QGroupBox("Composantes (exclure = selection)")
        ic_l = QtWidgets.QVBoxLayout(ic_box)
        self._ic_list = QtWidgets.QListWidget()
        self._ic_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        ic_l.addWidget(self._ic_list)
        self._preview_mode = QtWidgets.QComboBox()
        self._preview_mode.addItem("Previsualiser IC seule", "ic")
        self._preview_mode.addItem("EEG apres exclusion", "eeg")
        self._preview_mode.currentIndexChanged.connect(self._on_preview_mode)
        ic_l.addWidget(self._preview_mode)
        top.addWidget(ic_box, 2)
        layout.addLayout(top)

        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("n_components:"))
        self._n_comp = QtWidgets.QSpinBox()
        self._n_comp.setRange(0, 64)
        self._n_comp.setSpecialValueText("auto")
        self._n_comp.setValue(0)
        opts.addWidget(self._n_comp)
        fit_btn = QtWidgets.QPushButton("Fit ICA")
        fit_btn.clicked.connect(self._run_fit)
        opts.addWidget(fit_btn)
        opts.addWidget(QtWidgets.QLabel("Apercu canal:"))
        self._preview_ch = QtWidgets.QComboBox()
        for ch in self._all_fit_candidates():
            self._preview_ch.addItem(ch)
        for i, ch in enumerate(self._all_fit_candidates()):
            if ch in eeg_set:
                self._preview_ch.setCurrentIndex(i)
                break
        opts.addWidget(self._preview_ch)
        opts.addWidget(QtWidgets.QLabel("t0:"))
        self._t0 = QtWidgets.QDoubleSpinBox()
        self._t0.setRange(0.0, 1e7)
        self._t0.setDecimals(2)
        self._t0.setValue(self._preview_t0)
        opts.addWidget(self._t0)
        opts.addWidget(QtWidgets.QLabel("duree:"))
        self._dur = QtWidgets.QDoubleSpinBox()
        self._dur.setRange(1.0, 600.0)
        self._dur.setDecimals(2)
        self._dur.setValue(self._preview_duration)
        opts.addWidget(self._dur)
        sync_btn = QtWidgets.QPushButton("Fenetre viewer")
        sync_btn.setToolTip("Reprendre t0/duree de la vue courante du viewer")
        sync_btn.clicked.connect(self._on_sync_window_clicked)
        opts.addWidget(sync_btn)
        opts.addStretch()
        layout.addLayout(opts)

        self._fig = Figure(figsize=(9, 4.5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(NavigationToolbar2QT(self._canvas, self))
        layout.addWidget(self._canvas, stretch=1)

        btns = QtWidgets.QHBoxLayout()
        self._btn_preview = QtWidgets.QPushButton("Apercu EEG nettoye")
        self._btn_preview.setEnabled(False)
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

        self._status = QtWidgets.QLabel("Pret.")
        layout.addWidget(self._status)
        self._sync_window_from_viewer()

    def _on_sync_window_clicked(self) -> None:
        self._sync_window_from_viewer()
        row = self._ic_list.currentRow()
        if self._preview_mode.currentData() == "eeg" and self._preview_done:
            self._plot_eeg_preview()
        elif row >= 0 and self._fit is not None:
            self._plot_ic(row)
        self._status.setText(
            f"Fenetre alignee: t0={self._t0.value():.1f}s, "
            f"duree={self._dur.value():.1f}s"
        )

    def _selected_channels(self) -> List[str]:
        names = []
        for item in self._ch_list.selectedItems():
            name = item.data(QtCore.Qt.ItemDataRole.UserRole)
            names.append(str(name) if name is not None else item.text())
        return names

    def _selected_exclude(self) -> List[int]:
        idxs = []
        for item in self._ic_list.selectedItems():
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if data is not None:
                idxs.append(int(data))
        return idxs

    def _run_fit(self) -> None:
        chs = self._selected_channels()
        if len(chs) < 2:
            QtWidgets.QMessageBox.warning(
                self, "ICA", "Selectionnez au moins 2 canaux EEG.",
            )
            return
        data = {}
        for ch in chs:
            if ch not in self._signals:
                continue
            data[ch] = np.asarray(self._signals[ch][0], dtype=float)
            self._sfreq = float(self._signals[ch][1])
        n_comp = self._n_comp.value()
        progress = QtWidgets.QProgressDialog("Fit ICA...", None, 0, 0, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()
        QtWidgets.QApplication.processEvents()
        try:
            self._fit = fit_ica(
                data,
                self._sfreq,
                n_components=None if n_comp == 0 else n_comp,
            )
            self._stats = component_stats(
                self._fit,
                eog_uv=self._ref_channel("eog"),
                ecg_uv=self._ref_channel("ecg"),
            )
        except Exception as exc:
            progress.close()
            logger.error("ICA fit failed: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(self, "ICA", f"Fit echoue:\n{exc}")
            return
        progress.close()
        self._ic_list.clear()
        for i, label in enumerate(component_labels(self._fit, self._stats)):
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            self._ic_list.addItem(item)
        self._btn_preview.setEnabled(True)
        self._preview_done = False
        self._btn_apply.setEnabled(False)
        self._status.setText(
            f"ICA fit OK — {self._fit.n_components} composantes sur {len(chs)} canaux. "
            "Cliquez une IC pour voir trace + spectre."
        )
        if self._ic_list.count():
            self._ic_list.setCurrentRow(0)

    def _on_preview_mode(self) -> None:
        mode = self._preview_mode.currentData()
        if mode == "ic":
            row = self._ic_list.currentRow()
            if row >= 0:
                self._plot_ic(row)
        elif mode == "eeg" and self._preview_done:
            self._plot_eeg_preview()

    def _on_ic_selected(self, row: int) -> None:
        if row < 0 or self._fit is None:
            return
        if self._preview_mode.currentData() == "ic":
            self._plot_ic(row)

    def _plot_ic(self, idx: int) -> None:
        if self._fit is None:
            return
        t0, t1 = self._window_bounds()
        fs = float(self._fit.sfreq)
        try:
            sources = get_sources(self._fit, tmin=t0, tmax=t1)
        except Exception as exc:
            logger.error("get_sources failed: %s", exc, exc_info=True)
            return
        if idx < 0 or idx >= sources.shape[0]:
            return
        seg = sources[idx]
        if seg.size == 0:
            return
        t = t0 + np.arange(seg.size) / fs
        # Spectrum on the same window
        freqs, psd = component_spectrum(seg, fs)
        st = next((s for s in self._stats if int(s["index"]) == idx), None)
        title_extra = ""
        if st:
            title_extra = (
                f"  kurt={float(st['kurtosis']):.1f}  "
                f"|r_EOG|={float(st['corr_eog']):.2f}  "
                f"|r_ECG|={float(st['corr_ecg']):.2f}  "
                f"hint={st.get('label_hint', 'other')}"
            )

        self._fig.clear()
        ax1 = self._fig.add_subplot(211)
        ax1.plot(t, seg, color="#89B4FA", linewidth=0.8)
        ax1.set_ylabel("IC (a.u.)")
        ax1.set_title(
            f"IC{idx}{title_extra}  [{t0:.1f}–{t1:.1f} s]", fontsize=10,
        )
        ax1.grid(True, alpha=0.3)
        ax2 = self._fig.add_subplot(212)
        ax2.semilogy(freqs, np.maximum(psd, 1e-20), color="#A6E3A1")
        ax2.set_xlabel("Frequence (Hz)")
        ax2.set_ylabel("PSD")
        ax2.set_xlim(0, min(50.0, fs / 2.0))
        ax2.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._canvas.draw()
        self._status.setText(
            f"Inspection IC{idx} sur fenetre {t0:.1f}–{t1:.1f} s."
        )

    def _run_preview(self) -> None:
        if self._fit is None:
            return
        self._sync_window_from_viewer()
        exclude = self._selected_exclude()
        try:
            cleaned = preview_exclude(self._fit, exclude)
        except Exception as exc:
            logger.error("ICA preview failed: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.warning(self, "ICA", f"Apercu echoue:\n{exc}")
            return
        self._last_cleaned = cleaned
        self._preview_done = True
        self._btn_apply.setEnabled(True)
        self._preview_mode.setCurrentIndex(
            self._preview_mode.findData("eeg")
        )
        self._plot_eeg_preview()
        self._status.setText(
            f"Apercu EEG OK — {len(cleaned)} canaux, "
            f"{len(exclude)} composante(s) exclue(s) "
            f"(fenetre {self._t0.value():.1f}–"
            f"{self._t0.value() + self._dur.value():.1f} s)."
        )

    def _plot_eeg_preview(self) -> None:
        cleaned = self._last_cleaned
        if not cleaned:
            return
        t0, t1 = self._window_bounds()
        exclude = self._selected_exclude()
        ch = self._preview_ch.currentText()
        if ch not in self._signals or ch not in cleaned:
            ch = next(iter(cleaned.keys()), "")
        if not ch:
            return
        fs = float(self._signals[ch][1]) if ch in self._signals else self._sfreq
        i0 = int(t0 * fs)
        i1 = int(t1 * fs)
        raw = np.asarray(self._signals[ch][0], dtype=float)
        cln = np.asarray(cleaned[ch], dtype=float)
        i1 = min(i1, len(raw), len(cln))
        i0 = max(0, min(i0, i1 - 1))
        t = np.arange(i0, i1) / fs
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.plot(t, raw[i0:i1], label="avant", alpha=0.7)
        ax.plot(t, cln[i0:i1], label="apres ICA", alpha=0.9)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("uV")
        ax.set_title(f"{ch} — exclude={exclude or 'aucune'} [{t0:.1f}–{t1:.1f} s]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._canvas.draw()

    def _apply(self) -> None:
        if not self._preview_done or not self._last_cleaned:
            QtWidgets.QMessageBox.information(
                self, "ICA", "Lancez un apercu EEG nettoye avant d'appliquer.",
            )
            return
        if self._on_apply:
            self._on_apply(dict(self._last_cleaned))
        self._btn_restore.setEnabled(True)
        self._has_ica = True
        self._banner.setText(
            f"Signaux viewer : filtres {'ON' if self._filters_on else 'OFF'}  ·  "
            f"nettoyage oculaire actif : {'oui' if self._ocular_active else 'non'}  ·  "
            f"ICA deja appliquee : oui"
        )
        self._status.setText("ICA appliquee au viewer (memoire).")

    def _restore(self) -> None:
        if self._on_restore:
            self._on_restore()
        self._btn_restore.setEnabled(False)
        self._has_ica = False
        self._banner.setText(
            f"Signaux viewer : filtres {'ON' if self._filters_on else 'OFF'}  ·  "
            f"nettoyage oculaire actif : {'oui' if self._ocular_active else 'non'}  ·  "
            f"ICA deja appliquee : non"
        )
        self._status.setText("Signaux restaures depuis le backup ICA.")
