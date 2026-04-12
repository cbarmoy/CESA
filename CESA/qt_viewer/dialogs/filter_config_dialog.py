"""Qt filter configuration dialog for CESA.

Provides per-channel filter pipeline editing with:
* Channel list with type badges and search
* Pipeline editor: add / remove / reorder filter stages
* Per-filter parameter sliders
* Live signal + frequency-response preview (matplotlib)
* Preset dropdown (load from PresetLibrary)
* Global enable toggle
* Apply / Cancel
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]

from CESA.filter_engine import (
    BaseFilter,
    BandpassFilter,
    FilterPipeline,
    HighpassFilter,
    LowpassFilter,
    NotchFilter,
    PresetLibrary,
    SmoothingFilter,
    filter_from_dict,
)

_FILTER_TYPES: Dict[str, type] = {
    "Passe-bande": BandpassFilter,
    "Passe-haut": HighpassFilter,
    "Passe-bas": LowpassFilter,
    "Notch (rejet)": NotchFilter,
    "Lissage": SmoothingFilter,
}

_DEFAULT_PRESETS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "filter_presets.json"


def _short_label(filt: BaseFilter) -> str:
    if isinstance(filt, BandpassFilter):
        return f"BP {filt.low_hz:.1f}-{filt.high_hz:.1f} Hz"
    if isinstance(filt, HighpassFilter):
        return f"HP {filt.cutoff_hz:.1f} Hz"
    if isinstance(filt, LowpassFilter):
        return f"LP {filt.cutoff_hz:.1f} Hz"
    if isinstance(filt, NotchFilter):
        return f"Notch {filt.freq_hz:.0f} Hz"
    if isinstance(filt, SmoothingFilter):
        return f"Smooth {filt.method}"
    return type(filt).__name__


# ---------------------------------------------------------------------------
# Single-filter parameter widget
# ---------------------------------------------------------------------------

class _FilterCardWidget(QtWidgets.QGroupBox):
    """Editable card for one filter in the pipeline."""

    changed = QtCore.Signal()
    delete_requested = QtCore.Signal()
    move_up = QtCore.Signal()
    move_down = QtCore.Signal()

    def __init__(self, filt: BaseFilter, index: int, parent=None):
        super().__init__(parent)
        self.filt = filt
        self.setTitle(f"{index + 1}. {_short_label(filt)}")
        self._build(filt)

    def _build(self, f: BaseFilter) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 10, 6, 6)

        # Header row: enabled + action buttons
        hdr = QtWidgets.QHBoxLayout()
        self._enabled_cb = QtWidgets.QCheckBox("Actif")
        self._enabled_cb.setChecked(f.enabled)
        self._enabled_cb.toggled.connect(self._on_enabled)
        hdr.addWidget(self._enabled_cb)
        hdr.addStretch()
        btn_up = QtWidgets.QPushButton("\u25B2")
        btn_up.setFixedWidth(28)
        btn_up.clicked.connect(self.move_up.emit)
        btn_down = QtWidgets.QPushButton("\u25BC")
        btn_down.setFixedWidth(28)
        btn_down.clicked.connect(self.move_down.emit)
        btn_del = QtWidgets.QPushButton("\u2715")
        btn_del.setFixedWidth(28)
        btn_del.clicked.connect(self.delete_requested.emit)
        hdr.addWidget(btn_up)
        hdr.addWidget(btn_down)
        hdr.addWidget(btn_del)
        layout.addLayout(hdr)

        # Parameter rows
        if isinstance(f, BandpassFilter):
            self._add_spin(layout, "Bas (Hz)", 0.01, 200.0, f.low_hz, 0.1,
                           lambda v: self._set("low_hz", v))
            self._add_spin(layout, "Haut (Hz)", 0.1, 500.0, f.high_hz, 0.1,
                           lambda v: self._set("high_hz", v))
            self._add_int_spin(layout, "Ordre", 1, 12, f.order,
                               lambda v: self._set("order", v))
        elif isinstance(f, HighpassFilter):
            self._add_spin(layout, "Coupure (Hz)", 0.01, 200.0, f.cutoff_hz, 0.1,
                           lambda v: self._set("cutoff_hz", v))
            self._add_int_spin(layout, "Ordre", 1, 12, f.order,
                               lambda v: self._set("order", v))
        elif isinstance(f, LowpassFilter):
            self._add_spin(layout, "Coupure (Hz)", 0.1, 500.0, f.cutoff_hz, 0.1,
                           lambda v: self._set("cutoff_hz", v))
            self._add_int_spin(layout, "Ordre", 1, 12, f.order,
                               lambda v: self._set("order", v))
        elif isinstance(f, NotchFilter):
            self._add_spin(layout, "Frequence (Hz)", 1.0, 500.0, f.freq_hz, 1.0,
                           lambda v: self._set("freq_hz", v))
            self._add_spin(layout, "Facteur Q", 1.0, 100.0, f.quality_factor, 1.0,
                           lambda v: self._set("quality_factor", v))
            self._add_int_spin(layout, "Harmoniques", 1, 5, f.harmonics,
                               lambda v: self._set("harmonics", v))
        elif isinstance(f, SmoothingFilter):
            self._add_combo(layout, "Methode",
                            ["moving_average", "savgol", "gaussian"],
                            f.method, lambda v: self._set("method", v))
            self._add_int_spin(layout, "Fenetre", 3, 201, f.window_size,
                               lambda v: self._set("window_size", v))
            self._add_int_spin(layout, "Polynome", 0, 10, f.poly_order,
                               lambda v: self._set("poly_order", v))

    def _on_enabled(self, checked: bool) -> None:
        self.filt.enabled = checked
        self.changed.emit()

    def _set(self, attr: str, value: Any) -> None:
        setattr(self.filt, attr, value)
        self.setTitle(f"{self.title().split('.')[0]}. {_short_label(self.filt)}")
        self.changed.emit()

    def _add_spin(self, layout, label, lo, hi, val, step, setter):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(2)
        sp.setValue(val)
        sp.valueChanged.connect(setter)
        row.addWidget(sp)
        layout.addLayout(row)

    def _add_int_spin(self, layout, label, lo, hi, val, setter):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        sp = QtWidgets.QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        sp.valueChanged.connect(setter)
        row.addWidget(sp)
        layout.addLayout(row)

    def _add_combo(self, layout, label, options, val, setter):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        cb = QtWidgets.QComboBox()
        cb.addItems(options)
        cb.setCurrentText(val)
        cb.currentTextChanged.connect(setter)
        row.addWidget(cb)
        layout.addLayout(row)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class FilterConfigDialog(QtWidgets.QDialog):
    """Qt filter configuration dialog.

    Parameters
    ----------
    channel_names : list of str
    sfreq : float
    channel_pipelines : dict mapping channel name -> FilterPipeline
    channel_types : dict mapping channel name -> type string
    signals : dict mapping channel name -> (data_array, sfreq)
    global_enabled : bool
    on_apply : callback(pipelines_dict, global_enabled_bool)
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        channel_names: List[str],
        sfreq: float,
        *,
        channel_pipelines: Optional[Dict[str, FilterPipeline]] = None,
        channel_types: Optional[Dict[str, str]] = None,
        signals: Optional[Dict[str, Signal]] = None,
        global_enabled: bool = True,
        on_apply: Optional[Callable[[Dict[str, FilterPipeline], bool], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configuration des Filtres - CESA")
        self.resize(1200, 750)
        self.setMinimumSize(800, 500)

        self._channel_names = list(channel_names)
        self._sfreq = sfreq
        self._channel_types = channel_types or {}
        self._signals = signals or {}
        self._on_apply_cb = on_apply
        self._global_enabled = global_enabled

        self._pipelines: Dict[str, FilterPipeline] = {}
        for ch in self._channel_names:
            if channel_pipelines and ch in channel_pipelines:
                self._pipelines[ch] = channel_pipelines[ch].deep_copy()
            else:
                self._pipelines[ch] = FilterPipeline()

        try:
            self._preset_lib = PresetLibrary(_DEFAULT_PRESETS_PATH)
        except Exception:
            self._preset_lib = None

        self._selected_channel: Optional[str] = None
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(200)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()

        if self._channel_names:
            self._ch_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)

        # Top bar: global toggle + preset
        top = QtWidgets.QHBoxLayout()
        self._global_cb = QtWidgets.QCheckBox("Filtrage actif")
        self._global_cb.setChecked(self._global_enabled)
        self._global_cb.toggled.connect(self._schedule_preview)
        top.addWidget(self._global_cb)

        top.addSpacing(16)
        top.addWidget(QtWidgets.QLabel("Preset:"))
        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.setMinimumWidth(180)
        self._refresh_preset_list()
        top.addWidget(self._preset_combo)
        btn_apply_preset = QtWidgets.QPushButton("Appliquer preset")
        btn_apply_preset.clicked.connect(self._apply_preset)
        top.addWidget(btn_apply_preset)
        top.addStretch()
        main_layout.addLayout(top)

        # Splitter: channel list | pipeline editor | preview
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left: channel list
        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText("Rechercher...")
        self._search_edit.textChanged.connect(self._filter_channel_list)
        left_lay.addWidget(self._search_edit)

        self._ch_list = QtWidgets.QListWidget()
        self._ch_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._populate_channel_list()
        self._ch_list.currentRowChanged.connect(self._on_channel_selected)
        left_lay.addWidget(self._ch_list)

        # Batch buttons
        batch_box = QtWidgets.QGroupBox("Batch")
        batch_lay = QtWidgets.QVBoxLayout(batch_box)
        btn_copy = QtWidgets.QPushButton("Copier vers tous")
        btn_copy.clicked.connect(self._batch_copy)
        batch_lay.addWidget(btn_copy)
        btn_reset = QtWidgets.QPushButton("Reinitialiser tout")
        btn_reset.clicked.connect(self._batch_reset)
        batch_lay.addWidget(btn_reset)
        left_lay.addWidget(batch_box)

        left.setMaximumWidth(280)
        splitter.addWidget(left)

        # Centre: pipeline editor
        centre = QtWidgets.QWidget()
        self._centre_layout = QtWidgets.QVBoxLayout(centre)
        self._centre_layout.setContentsMargins(4, 0, 4, 0)

        self._pipeline_scroll = QtWidgets.QScrollArea()
        self._pipeline_scroll.setWidgetResizable(True)
        self._pipeline_content = QtWidgets.QWidget()
        self._pipeline_vbox = QtWidgets.QVBoxLayout(self._pipeline_content)
        self._pipeline_vbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self._pipeline_scroll.setWidget(self._pipeline_content)
        self._centre_layout.addWidget(self._pipeline_scroll, stretch=1)

        # Add filter row
        add_row = QtWidgets.QHBoxLayout()
        self._add_type_combo = QtWidgets.QComboBox()
        self._add_type_combo.addItems(list(_FILTER_TYPES.keys()))
        add_row.addWidget(self._add_type_combo)
        btn_add = QtWidgets.QPushButton("+ Ajouter filtre")
        btn_add.clicked.connect(self._add_filter)
        add_row.addWidget(btn_add)
        add_row.addStretch()
        self._centre_layout.addLayout(add_row)

        splitter.addWidget(centre)

        # Right: preview (matplotlib)
        right = QtWidgets.QWidget()
        self._right_layout = QtWidgets.QVBoxLayout(right)
        self._right_layout.setContentsMargins(0, 0, 0, 0)
        self._build_preview(right)
        right.setMinimumWidth(300)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)
        main_layout.addWidget(splitter, stretch=1)

        # Bottom bar
        bottom = QtWidgets.QHBoxLayout()
        self._status_label = QtWidgets.QLabel("")
        bottom.addWidget(self._status_label)
        bottom.addStretch()
        btn_cancel = QtWidgets.QPushButton("Annuler")
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        btn_apply = QtWidgets.QPushButton("Appliquer et Fermer")
        btn_apply.setDefault(True)
        btn_apply.clicked.connect(self._on_apply)
        bottom.addWidget(btn_apply)
        main_layout.addLayout(bottom)

        self._update_status()

    def _build_preview(self, parent: QtWidgets.QWidget) -> None:
        self._has_preview = False
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(5, 6), dpi=90)
            self._fig.set_facecolor('#1E1E2E')
            gs = self._fig.add_gridspec(2, 1, hspace=0.4)
            self._ax_signal = self._fig.add_subplot(gs[0])
            self._ax_freq = self._fig.add_subplot(gs[1])
            for ax in (self._ax_signal, self._ax_freq):
                ax.set_facecolor('#1E1E2E')
                ax.tick_params(colors='#CDD6F4', labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color('#45475A')

            self._canvas_mpl = FigureCanvasQTAgg(self._fig)
            self._right_layout.addWidget(self._canvas_mpl, stretch=1)
            self._has_preview = True
        except Exception:
            self._right_layout.addWidget(QtWidgets.QLabel("(matplotlib indisponible)"))

    # ------------------------------------------------------------------
    # Channel list
    # ------------------------------------------------------------------

    def _populate_channel_list(self, query: str = "") -> None:
        self._ch_list.blockSignals(True)
        self._ch_list.clear()
        q = query.strip().lower()
        for ch in self._channel_names:
            if q and q not in ch.lower():
                continue
            ctype = self._channel_types.get(ch, "?").upper()[:3]
            pipe = self._pipelines.get(ch, FilterPipeline())
            n_active = sum(1 for f in pipe.filters if f.enabled) if pipe.enabled else 0
            badge = f" [{n_active}F]" if n_active else ""
            self._ch_list.addItem(f"[{ctype}] {ch}{badge}")
        self._ch_list.blockSignals(False)

    def _filter_channel_list(self, text: str) -> None:
        prev = self._selected_channel
        self._populate_channel_list(text)
        if prev:
            for i in range(self._ch_list.count()):
                if prev in (self._ch_list.item(i).text() or ""):
                    self._ch_list.setCurrentRow(i)
                    return

    def _ch_name_from_row(self, row: int) -> Optional[str]:
        if row < 0:
            return None
        item = self._ch_list.item(row)
        if item is None:
            return None
        text = item.text()
        # Format: "[TYP] name [nF]"  -- extract name between "] " and end/next "["
        try:
            after_bracket = text.split("] ", 1)[1]
            name = after_bracket.split(" [")[0].strip()
            return name if name in self._pipelines else None
        except Exception:
            return None

    def _on_channel_selected(self, row: int) -> None:
        ch = self._ch_name_from_row(row)
        if ch is None:
            return
        self._selected_channel = ch
        self._rebuild_pipeline_editor()
        self._schedule_preview()

    # ------------------------------------------------------------------
    # Pipeline editor
    # ------------------------------------------------------------------

    def _rebuild_pipeline_editor(self) -> None:
        # Clear existing cards
        while self._pipeline_vbox.count():
            item = self._pipeline_vbox.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        ch = self._selected_channel
        if not ch:
            lbl = QtWidgets.QLabel("Selectionnez un canal")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._pipeline_vbox.addWidget(lbl)
            return

        pipeline = self._pipelines.get(ch, FilterPipeline())

        # Pipeline enabled toggle
        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(f"Pipeline : {ch}")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        hdr.addWidget(title)
        hdr.addStretch()
        pe_cb = QtWidgets.QCheckBox("Pipeline actif")
        pe_cb.setChecked(pipeline.enabled)
        pe_cb.toggled.connect(lambda v: self._toggle_pipeline(v))
        hdr.addWidget(pe_cb)
        hdr_w = QtWidgets.QWidget()
        hdr_w.setLayout(hdr)
        self._pipeline_vbox.addWidget(hdr_w)

        for i, filt in enumerate(pipeline.filters):
            card = _FilterCardWidget(filt, i)
            card.changed.connect(self._on_filter_changed)
            card.delete_requested.connect(lambda idx=i: self._delete_filter(idx))
            card.move_up.connect(lambda idx=i: self._move_filter(idx, idx - 1))
            card.move_down.connect(lambda idx=i: self._move_filter(idx, idx + 1))
            self._pipeline_vbox.addWidget(card)

        if not pipeline.filters:
            empty = QtWidgets.QLabel("Aucun filtre. Utilisez '+ Ajouter filtre' ci-dessous.")
            empty.setStyleSheet("color: #6C7086; padding: 20px;")
            empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._pipeline_vbox.addWidget(empty)

    def _toggle_pipeline(self, enabled: bool) -> None:
        ch = self._selected_channel
        if ch and ch in self._pipelines:
            self._pipelines[ch].enabled = enabled
            self._schedule_preview()
            self._refresh_channel_badges()

    def _on_filter_changed(self) -> None:
        self._schedule_preview()
        self._refresh_channel_badges()

    def _add_filter(self) -> None:
        ch = self._selected_channel
        if not ch:
            return
        type_label = self._add_type_combo.currentText()
        cls = _FILTER_TYPES.get(type_label)
        if cls is None:
            return
        pipeline = self._pipelines.setdefault(ch, FilterPipeline())
        pipeline.filters.append(cls())
        self._rebuild_pipeline_editor()
        self._schedule_preview()
        self._refresh_channel_badges()

    def _delete_filter(self, idx: int) -> None:
        ch = self._selected_channel
        if not ch:
            return
        pipeline = self._pipelines.get(ch)
        if pipeline and 0 <= idx < len(pipeline.filters):
            pipeline.filters.pop(idx)
        self._rebuild_pipeline_editor()
        self._schedule_preview()
        self._refresh_channel_badges()

    def _move_filter(self, from_idx: int, to_idx: int) -> None:
        ch = self._selected_channel
        if not ch:
            return
        pipeline = self._pipelines.get(ch)
        if not pipeline:
            return
        n = len(pipeline.filters)
        if 0 <= from_idx < n and 0 <= to_idx < n:
            pipeline.filters[from_idx], pipeline.filters[to_idx] = (
                pipeline.filters[to_idx], pipeline.filters[from_idx]
            )
        self._rebuild_pipeline_editor()
        self._schedule_preview()

    def _refresh_channel_badges(self) -> None:
        query = self._search_edit.text()
        cur_row = self._ch_list.currentRow()
        self._populate_channel_list(query)
        if 0 <= cur_row < self._ch_list.count():
            self._ch_list.setCurrentRow(cur_row)
        self._update_status()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _schedule_preview(self) -> None:
        self._preview_timer.start()

    def _update_preview(self) -> None:
        if not self._has_preview:
            return
        ch = self._selected_channel
        if not ch:
            return

        pipeline = self._pipelines.get(ch, FilterPipeline())
        duration_s = 5.0

        snippet: Optional[np.ndarray] = None
        fs = self._sfreq
        sig = self._signals.get(ch)
        if sig is not None:
            data_arr, fs = sig
            n = min(len(data_arr), int(fs * duration_s))
            snippet = data_arr[:n]

        if snippet is None or len(snippet) < 2:
            n_pts = int(fs * duration_s)
            tt = np.arange(n_pts) / fs
            snippet = (
                30 * np.sin(2 * np.pi * 10 * tt)
                + 10 * np.sin(2 * np.pi * 0.5 * tt)
                + 5 * np.sin(2 * np.pi * 50 * tt)
                + np.random.default_rng(42).normal(0, 3, n_pts)
            )

        t_arr = np.arange(len(snippet)) / fs
        filtered = None
        if pipeline.enabled and pipeline.filters and self._global_cb.isChecked():
            try:
                filtered = pipeline.apply(snippet, fs)
            except Exception:
                filtered = None

        # Signal plot
        ax = self._ax_signal
        ax.clear()
        ax.plot(t_arr, snippet, color='#6C7086', linewidth=0.6, alpha=0.6, label='Brut')
        if filtered is not None:
            ax.plot(t_arr, filtered, color='#89B4FA', linewidth=1.0, label='Filtre')
        ax.set_title(f"Signal : {ch}", fontsize=9, color='#CDD6F4')
        ax.set_xlabel("Temps (s)", fontsize=8, color='#CDD6F4')
        ax.set_ylabel("\u00b5V", fontsize=8, color='#CDD6F4')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

        # Frequency response
        ax2 = self._ax_freq
        ax2.clear()
        try:
            freqs, mag_db = pipeline.frequency_response(fs, n_points=512)
            ax2.plot(freqs, mag_db, color='#A6E3A1', linewidth=1.0)
            ax2.set_xlim(0, min(fs / 2, 150))
            ax2.set_ylim(-60, 5)
            ax2.axhline(-3, color='#F38BA8', linestyle='--', linewidth=0.5, label='-3 dB')
            ax2.legend(fontsize=7)
        except Exception:
            pass
        ax2.set_title("Reponse en frequence", fontsize=9, color='#CDD6F4')
        ax2.set_xlabel("Frequence (Hz)", fontsize=8, color='#CDD6F4')
        ax2.set_ylabel("dB", fontsize=8, color='#CDD6F4')
        ax2.grid(True, alpha=0.2)

        for a in (ax, ax2):
            a.set_facecolor('#1E1E2E')
            a.tick_params(colors='#CDD6F4', labelsize=7)
            for spine in a.spines.values():
                spine.set_color('#45475A')

        self._fig.tight_layout(pad=1.5)
        try:
            self._canvas_mpl.draw_idle()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _refresh_preset_list(self) -> None:
        self._preset_combo.clear()
        if self._preset_lib is None:
            return
        try:
            names = self._preset_lib.list_names()
            self._preset_combo.addItems(names)
        except Exception:
            pass

    def _apply_preset(self) -> None:
        if not self._preset_lib or not self._selected_channel:
            return
        name = self._preset_combo.currentText()
        preset = self._preset_lib.get(name)
        if preset is None:
            return
        self._pipelines[self._selected_channel] = preset.pipeline.deep_copy()
        self._rebuild_pipeline_editor()
        self._schedule_preview()
        self._refresh_channel_badges()

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def _batch_copy(self) -> None:
        ch = self._selected_channel
        if not ch:
            return
        src = self._pipelines.get(ch, FilterPipeline())
        count = 0
        for other in self._channel_names:
            if other != ch:
                self._pipelines[other] = src.deep_copy()
                count += 1
        self._refresh_channel_badges()
        QtWidgets.QMessageBox.information(
            self, "Batch", f"Pipeline copie vers {count} canal(aux).")

    def _batch_reset(self) -> None:
        for ch in self._channel_names:
            self._pipelines[ch] = FilterPipeline()
        self._rebuild_pipeline_editor()
        self._schedule_preview()
        self._refresh_channel_badges()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        active = sum(1 for p in self._pipelines.values()
                     if p.enabled and p.filters)
        total = len(self._pipelines)
        self._status_label.setText(f"{active}/{total} canaux filtres")

    # ------------------------------------------------------------------
    # Apply / result
    # ------------------------------------------------------------------

    def _on_apply(self) -> None:
        if self._on_apply_cb:
            self._on_apply_cb(self._pipelines, self._global_cb.isChecked())
        self.accept()

    @property
    def pipelines(self) -> Dict[str, FilterPipeline]:
        return self._pipelines

    @property
    def global_enabled(self) -> bool:
        return self._global_cb.isChecked()
