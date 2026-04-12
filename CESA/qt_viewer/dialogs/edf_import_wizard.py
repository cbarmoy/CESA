"""Four-step EDF import wizard with drag-and-drop, preview, and progressive loading.

Pages
-----
1. File selection  -- drag-and-drop zone + browse button, instant validation
2. Metadata preview -- channel list with type colours, summary, mini signal viewer
3. Channel config   -- multi-select table, auto-detected types, gain spinboxes
4. Loading          -- threaded ``AppController.load_recording`` with progress bar
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg

from ..downsampler import downsample_minmax
from ..edf_metadata_loader import EDFMetadataLoader
from ..edf_session import DEFAULT_GAINS, ChannelInfo, EDFImportResult, EDFSession
from ..import_profile_store import ImportProfileStore
from ..themes import CHANNEL_TYPE_COLORS, DARK, THEMES, ThemePalette

logger = logging.getLogger(__name__)

# Friendly labels for signal types
_TYPE_LABELS: Dict[str, str] = {
    "eeg": "EEG",
    "eog": "EOG",
    "emg": "EMG",
    "ecg": "ECG",
    "other": "Autre",
}
_TYPE_KEYS = list(_TYPE_LABELS.keys())


def _theme_qss(t: ThemePalette) -> str:
    """Generate a QSS stylesheet from the viewer theme palette."""
    return (
        f"QDialog, QWidget {{ background: {t['background']}; color: {t['foreground']}; }}\n"
        f"QGroupBox {{ border: 1px solid {t['border']}; border-radius: 6px; "
        f"margin-top: 12px; padding-top: 14px; font-weight: bold; color: {t['foreground']}; }}\n"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}\n"
        f"QPushButton {{ background: {t['surface']}; color: {t['foreground']}; "
        f"border: 1px solid {t['border']}; border-radius: 4px; padding: 6px 16px; }}\n"
        f"QPushButton:hover {{ background: {t['accent']}; color: {t['background']}; }}\n"
        f"QPushButton:disabled {{ color: {t['text_dim']}; }}\n"
        f"QPushButton#primary {{ background: {t['accent']}; color: {t['background']}; "
        f"font-weight: bold; }}\n"
        f"QPushButton#primary:hover {{ background: {t['highlight']}; }}\n"
        f"QLabel {{ color: {t['foreground']}; }}\n"
        f"QLabel#dim {{ color: {t['text_dim']}; }}\n"
        f"QLabel#title {{ font-size: 15px; font-weight: bold; color: {t['accent']}; }}\n"
        f"QLabel#subtitle {{ font-size: 12px; color: {t['text_dim']}; }}\n"
        f"QLineEdit {{ background: {t['surface']}; color: {t['foreground']}; "
        f"border: 1px solid {t['border']}; border-radius: 4px; padding: 5px 8px; }}\n"
        f"QProgressBar {{ background: {t['surface']}; border: 1px solid {t['border']}; "
        f"border-radius: 4px; text-align: center; color: {t['foreground']}; }}\n"
        f"QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}\n"
        f"QTableWidget {{ background: {t['surface']}; color: {t['foreground']}; "
        f"gridline-color: {t['border']}; border: 1px solid {t['border']}; "
        f"selection-background-color: {t['accent_dim']}; }}\n"
        f"QHeaderView::section {{ background: {t['surface_alt']}; color: {t['foreground']}; "
        f"border: 1px solid {t['border']}; padding: 4px; }}\n"
        f"QComboBox {{ background: {t['surface']}; color: {t['foreground']}; "
        f"border: 1px solid {t['border']}; border-radius: 4px; padding: 4px 8px; }}\n"
        f"QComboBox QAbstractItemView {{ background: {t['surface']}; color: {t['foreground']}; "
        f"selection-background-color: {t['accent_dim']}; }}\n"
        f"QSpinBox, QDoubleSpinBox {{ background: {t['surface']}; color: {t['foreground']}; "
        f"border: 1px solid {t['border']}; border-radius: 4px; padding: 3px; }}\n"
        f"QCheckBox {{ color: {t['foreground']}; spacing: 6px; }}\n"
        f"QCheckBox::indicator {{ width: 16px; height: 16px; }}\n"
        f"QRadioButton {{ color: {t['foreground']}; }}\n"
        f"QScrollArea {{ border: none; }}\n"
        f"QListWidget {{ background: {t['surface']}; color: {t['foreground']}; "
        f"border: 1px solid {t['border']}; border-radius: 4px; outline: none; }}\n"
        f"QListWidget::item {{ padding: 4px 8px; }}\n"
        f"QListWidget::item:selected {{ background: {t['accent_dim']}; }}\n"
    )


# ======================================================================
# Page 1 -- File Selection
# ======================================================================

class _DropZone(QtWidgets.QLabel):
    """Large drop target that accepts recording files."""

    file_dropped = QtCore.Signal(str)

    def __init__(self, theme: ThemePalette, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setAcceptDrops(True)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(140)
        self._set_idle()

    def _set_idle(self) -> None:
        t = self._theme
        self.setText("Glissez-deposez un fichier EDF ici\nou cliquez sur Parcourir")
        self.setStyleSheet(
            f"QLabel {{ background: {t['surface']}; border: 2px dashed {t['border']}; "
            f"border-radius: 10px; color: {t['text_dim']}; font-size: 13px; padding: 20px; }}"
        )

    def _set_hover(self) -> None:
        t = self._theme
        self.setStyleSheet(
            f"QLabel {{ background: {t['surface_alt']}; border: 2px dashed {t['accent']}; "
            f"border-radius: 10px; color: {t['accent']}; font-size: 13px; padding: 20px; }}"
        )

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent) -> None:
        self._set_idle()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        self._set_idle()
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.file_dropped.emit(path)


class _PageFileSelect(QtWidgets.QWidget):
    """Step 1: file selection + drag-and-drop + validation."""

    file_validated = QtCore.Signal(str)  # emitted with valid path

    def __init__(self, theme: ThemePalette, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)

        title = QtWidgets.QLabel("Importer un enregistrement")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Selectionnez un fichier EDF, BDF, FIF, BrainVision ou EEGLAB."
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        layout.addSpacing(12)

        self._drop = _DropZone(self._theme)
        self._drop.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self._drop)
        layout.addSpacing(8)

        row = QtWidgets.QHBoxLayout()
        self._path_edit = QtWidgets.QLineEdit()
        self._path_edit.setPlaceholderText("Chemin vers le fichier...")
        self._path_edit.textChanged.connect(self._on_text_changed)
        row.addWidget(self._path_edit, stretch=1)
        browse = QtWidgets.QPushButton("Parcourir...")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)

        self._status_icon = QtWidgets.QLabel()
        self._status_msg = QtWidgets.QLabel()
        self._status_msg.setObjectName("dim")
        self._status_msg.setWordWrap(True)
        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(self._status_icon)
        status_row.addWidget(self._status_msg, stretch=1)
        layout.addLayout(status_row)

        # Mode selection (raw / precomputed)
        mode_group = QtWidgets.QGroupBox("Mode de lecture")
        mode_lay = QtWidgets.QVBoxLayout(mode_group)
        self._mode_raw = QtWidgets.QRadioButton("Mode standard (lecture directe)")
        self._mode_raw.setChecked(True)
        self._mode_precomp = QtWidgets.QRadioButton(
            "Navigation rapide (fichier pre-calcule Zarr)"
        )
        mode_lay.addWidget(self._mode_raw)
        mode_lay.addWidget(self._mode_precomp)

        self._precomp_frame = QtWidgets.QWidget()
        pf = QtWidgets.QVBoxLayout(self._precomp_frame)
        pf.setContentsMargins(20, 4, 0, 0)
        pf.addWidget(QtWidgets.QLabel("Dossier Zarr :"))
        ms_row = QtWidgets.QHBoxLayout()
        self._ms_edit = QtWidgets.QLineEdit()
        ms_row.addWidget(self._ms_edit, stretch=1)
        ms_browse = QtWidgets.QPushButton("Selectionner...")
        ms_browse.clicked.connect(self._browse_ms)
        ms_row.addWidget(ms_browse)
        pf.addLayout(ms_row)
        self._action_build = QtWidgets.QRadioButton("Creer / mettre a jour")
        self._action_build.setChecked(True)
        self._action_existing = QtWidgets.QRadioButton("Utiliser existant")
        pf.addWidget(self._action_build)
        pf.addWidget(self._action_existing)
        mode_lay.addWidget(self._precomp_frame)
        self._precomp_frame.setVisible(False)
        self._mode_precomp.toggled.connect(self._precomp_frame.setVisible)
        layout.addWidget(mode_group)

        layout.addStretch()

        self._set_status(None, "")

    # -- File validation ---------------------------------------------------

    def _on_file_dropped(self, path: str) -> None:
        self._path_edit.setText(path)

    def _on_text_changed(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            self._set_status(None, "")
            return
        ok, msg = EDFMetadataLoader.validate_file(path)
        if ok:
            self._set_status(True, msg)
            # auto-fill ms path
            if not self._ms_edit.text().strip():
                self._ms_edit.setText(str(Path(path).with_suffix("")) + "_ms")
            self.file_validated.emit(path)
        else:
            self._set_status(False, msg)

    def _set_status(self, ok: Optional[bool], msg: str) -> None:
        if ok is None:
            self._status_icon.setText("")
            self._status_msg.setText("")
        elif ok:
            self._status_icon.setText("  ✓ ")
            self._status_icon.setStyleSheet(f"color: #22c55e; font-size: 16px; font-weight: bold;")
            self._status_msg.setText(msg)
            self._status_msg.setStyleSheet(f"color: #22c55e;")
        else:
            self._status_icon.setText("  ✗ ")
            self._status_icon.setStyleSheet(f"color: #ef4444; font-size: 16px; font-weight: bold;")
            self._status_msg.setText(msg)
            self._status_msg.setStyleSheet(f"color: #ef4444;")

    def _browse(self) -> None:
        from core.raw_loader import SUPPORTED_RECORDING_EXTENSIONS

        exts = " ".join(f"*{e}" for e in SUPPORTED_RECORDING_EXTENSIONS)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Selectionner un enregistrement",
            "",
            f"Fichiers enregistrement ({exts});;Tous les fichiers (*.*)",
        )
        if path:
            self._path_edit.setText(path)

    def _browse_ms(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Dossier de navigation rapide"
        )
        if path:
            self._ms_edit.setText(path)

    # -- Value accessors ---------------------------------------------------

    def file_path(self) -> str:
        return self._path_edit.text().strip()

    def set_file_path(self, path: str) -> None:
        self._path_edit.setText(path)

    def mode(self) -> str:
        return "precomputed" if self._mode_precomp.isChecked() else "raw"

    def ms_path(self) -> Optional[str]:
        if self._mode_precomp.isChecked():
            return self._ms_edit.text().strip() or None
        return None

    def precompute_action(self) -> str:
        return "build" if self._action_build.isChecked() else "existing"

    def is_valid(self) -> bool:
        ok, _ = EDFMetadataLoader.validate_file(self.file_path())
        return ok


# ======================================================================
# Page 2 -- Metadata Preview
# ======================================================================

class _PagePreview(QtWidgets.QWidget):
    """Step 2: metadata summary + mini signal viewer."""

    def __init__(self, theme: ThemePalette, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._session: Optional[EDFSession] = None
        self._loader: Optional[EDFMetadataLoader] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)

        title = QtWidgets.QLabel("Apercu de l'enregistrement")
        title.setObjectName("title")
        layout.addWidget(title)

        body = QtWidgets.QHBoxLayout()

        # Left: channel list
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Canaux detectes :"))
        self._ch_list = QtWidgets.QListWidget()
        self._ch_list.currentRowChanged.connect(self._on_channel_selected)
        left.addWidget(self._ch_list, stretch=1)
        body.addLayout(left, stretch=2)

        # Right: file metadata + per-channel detail panel
        right = QtWidgets.QVBoxLayout()

        self._meta_labels: Dict[str, QtWidgets.QLabel] = {}
        meta_group = QtWidgets.QGroupBox("Fichier")
        mg = QtWidgets.QFormLayout(meta_group)
        for key, label in [
            ("file", "Fichier"),
            ("duration", "Duree"),
            ("sfreq", "Freq. echantillonnage"),
            ("n_channels", "Nombre de canaux"),
            ("date", "Date d'enregistrement"),
            ("patient", "Patient"),
            ("types", "Types detectes"),
        ]:
            val_label = QtWidgets.QLabel("—")
            val_label.setWordWrap(True)
            self._meta_labels[key] = val_label
            mg.addRow(f"{label} :", val_label)
        right.addWidget(meta_group)

        body.addLayout(right, stretch=3)

        layout.addLayout(body, stretch=2)

        # Bottom: signal preview
        preview_label = QtWidgets.QLabel("Apercu signal (cliquez sur un canal) :")
        preview_label.setObjectName("subtitle")
        layout.addWidget(preview_label)

        self._plot = pg.PlotWidget()
        self._plot.setMinimumHeight(180)
        self._apply_plot_theme()
        self._plot.setLabel("bottom", "Temps", units="s")
        self._plot.setLabel("left", "Amplitude", units="µV")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot, stretch=2)

    def _apply_plot_theme(self) -> None:
        t = self._theme
        self._plot.setBackground(t["surface"])
        for axis in ("bottom", "left"):
            ax = self._plot.getAxis(axis)
            ax.setPen(pg.mkPen(t["foreground"], width=1))
            ax.setTextPen(pg.mkPen(t["foreground"]))

    def set_session(self, session: EDFSession, loader: EDFMetadataLoader) -> None:
        self._session = session
        self._loader = loader
        self._populate_metadata()
        self._plot.clear()
        self._populate_channel_list()

    def _populate_metadata(self) -> None:
        s = self._session
        if s is None:
            return
        self._meta_labels["file"].setText(s.file_path.name)
        self._meta_labels["duration"].setText(s.duration_hms)
        self._meta_labels["sfreq"].setText(f"{s.sfreq:.1f} Hz")
        self._meta_labels["n_channels"].setText(str(len(s.channels)))

        if s.recording_date:
            self._meta_labels["date"].setText(s.recording_date.strftime("%Y-%m-%d %H:%M"))
        else:
            self._meta_labels["date"].setText("Non disponible")

        if s.patient_info:
            parts = [f"{k}: {v}" for k, v in s.patient_info.items()]
            self._meta_labels["patient"].setText("; ".join(parts))
        else:
            self._meta_labels["patient"].setText("Non disponible")

        type_counts: Dict[str, int] = {}
        for ch in s.channels:
            type_counts[ch.signal_type] = type_counts.get(ch.signal_type, 0) + 1
        parts = [f"{_TYPE_LABELS.get(k, k)} ({v})" for k, v in type_counts.items()]
        self._meta_labels["types"].setText(", ".join(parts))

    def _populate_channel_list(self) -> None:
        self._ch_list.clear()
        if self._session is None:
            return
        t = self._theme
        for ch in self._session.channels:
            color_key = CHANNEL_TYPE_COLORS.get(ch.signal_type, "foreground")
            color = t.get(color_key, t["foreground"])
            item = QtWidgets.QListWidgetItem(
                f"  {ch.name}  ({_TYPE_LABELS.get(ch.signal_type, ch.signal_type)})"
            )
            item.setForeground(QtGui.QColor(color))
            self._ch_list.addItem(item)
        if self._ch_list.count() > 0:
            self._ch_list.setCurrentRow(0)

    def _on_channel_selected(self, row: int) -> None:
        if self._session is None or self._loader is None:
            return
        if row < 0 or row >= len(self._session.channels):
            return
        ch = self._session.channels[row]

        try:
            preview = self._loader.load_preview_chunk([ch.name], 0.0, 10.0)
        except Exception:
            logger.error("Preview chunk error for %s", ch.name, exc_info=True)
            preview = {}

        self._plot.clear()

        if ch.name not in preview or len(preview[ch.name][1]) == 0:
            self._plot.setTitle(f"{ch.name} — aucune donnee disponible")
            logger.warning("[PREVIEW] No data returned for channel %s", ch.name)
            return

        try:
            chunk = preview[ch.name]
            times, data = chunk[0], chunk[1]
            data = np.asarray(data, dtype=np.float64)
            times = np.asarray(times, dtype=np.float64)

            if len(data) > 2000:
                _, data_ds = downsample_minmax(data, 2000)
                times_ds = np.linspace(float(times[0]), float(times[-1]), len(data_ds))
            else:
                data_ds, times_ds = data, times

            color_key = CHANNEL_TYPE_COLORS.get(ch.signal_type, "foreground")
            color = self._theme.get(color_key, self._theme["foreground"])
            self._plot.plot(times_ds, data_ds, pen=pg.mkPen(color, width=1.5))

            ds_min = float(np.nanmin(data_ds))
            ds_max = float(np.nanmax(data_ds))
            ds_range = ds_max - ds_min
            if ds_range < 1e-9:
                center = (ds_min + ds_max) / 2.0
                pad = max(abs(center) * 0.1, 1.0)
                ds_min, ds_max = center - pad, center + pad
            else:
                pad = ds_range * 0.08
                ds_min -= pad
                ds_max += pad

            self._plot.setYRange(ds_min, ds_max, padding=0)
            self._plot.setXRange(float(times_ds[0]), float(times_ds[-1]), padding=0.02)
            self._plot.setTitle(
                f"{ch.name}  —  {_TYPE_LABELS.get(ch.signal_type, '?').upper()}  —  "
                f"{ch.sfreq:.0f} Hz  —  {ch.unit}"
            )
        except Exception:
            logger.error("Preview plot error for %s", ch.name, exc_info=True)
            self._plot.setTitle(f"{ch.name} — erreur d'affichage")


# ======================================================================
# Page 3 -- Channel Configuration
# ======================================================================

class _PageChannelConfig(QtWidgets.QWidget):
    """Step 3: channel selection, type mapping, gain, rename, profile save/load."""

    def __init__(self, theme: ThemePalette, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._session: Optional[EDFSession] = None
        self._profile_store = ImportProfileStore()
        self._wizard_ref = None  # set by EDFImportWizard after construction
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)

        title = QtWidgets.QLabel("Configuration des canaux")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Selectionnez les canaux, renommez-les, verifiez le type et ajustez le gain."
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        layout.addSpacing(8)

        # Preset + profile buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Tout selectionner")
        btn_all.clicked.connect(self._select_all)
        btn_none = QtWidgets.QPushButton("Tout deselectionner")
        btn_none.clicked.connect(self._select_none)
        btn_psg = QtWidgets.QPushButton("PSG standard")
        btn_psg.clicked.connect(self._select_psg)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addWidget(btn_psg)
        btn_row.addStretch()

        btn_load = QtWidgets.QPushButton("Charger profil...")
        btn_load.clicked.connect(self._load_profile)
        btn_save = QtWidgets.QPushButton("Sauvegarder profil...")
        btn_save.clicked.connect(self._save_profile)
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

        # Auto-match info bar
        self._profile_info = QtWidgets.QLabel("")
        self._profile_info.setObjectName("dim")
        self._profile_info.setWordWrap(True)
        layout.addWidget(self._profile_info)

        # Channel table
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Canal", "Nom affiche", "Type", "Gain (µV)", "Freq (Hz)"]
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self._table, stretch=1)

        # Warning area
        self._warn_label = QtWidgets.QLabel()
        self._warn_label.setWordWrap(True)
        self._warn_label.setStyleSheet("color: #f59e0b; font-style: italic;")
        layout.addWidget(self._warn_label)

    def set_session(self, session: EDFSession) -> None:
        self._session = session
        self._auto_apply_profile()
        self._populate_table()

    def _populate_table(self) -> None:
        s = self._session
        if s is None:
            return
        self._table.setRowCount(len(s.channels))
        for row, ch in enumerate(s.channels):
            # Col 0: Checkbox + original name
            cb = QtWidgets.QCheckBox(ch.name)
            cb.setChecked(ch.selected)
            cb.stateChanged.connect(lambda state, r=row: self._on_check(r, state))
            self._table.setCellWidget(row, 0, cb)

            # Col 1: Editable alias (display name)
            alias_edit = QtWidgets.QLineEdit()
            alias_edit.setPlaceholderText(ch.name)
            alias_edit.setText(ch.alias)
            alias_edit.textChanged.connect(
                lambda text, r=row: self._on_alias_changed(r, text)
            )
            self._table.setCellWidget(row, 1, alias_edit)

            # Col 2: Type combo
            combo = QtWidgets.QComboBox()
            combo.addItems(list(_TYPE_LABELS.values()))
            idx = _TYPE_KEYS.index(ch.signal_type) if ch.signal_type in _TYPE_KEYS else len(_TYPE_KEYS) - 1
            combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(
                lambda i, r=row: self._on_type_changed(r, i)
            )
            self._table.setCellWidget(row, 2, combo)

            # Col 3: Gain spinbox
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(1.0, 10000.0)
            spin.setSuffix(" µV")
            spin.setDecimals(0)
            spin.setValue(ch.gain)
            spin.valueChanged.connect(lambda v, r=row: self._on_gain_changed(r, v))
            self._table.setCellWidget(row, 3, spin)

            # Col 4: Sfreq (read-only)
            freq_item = QtWidgets.QTableWidgetItem(f"{ch.sfreq:.0f}")
            freq_item.setFlags(freq_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 4, freq_item)

        self._update_warnings()

    def _on_check(self, row: int, state: int) -> None:
        if self._session:
            self._session.channels[row].selected = bool(state)
            self._update_warnings()

    def _on_alias_changed(self, row: int, text: str) -> None:
        if self._session:
            self._session.channels[row].alias = text.strip()

    def _on_type_changed(self, row: int, idx: int) -> None:
        if self._session and 0 <= idx < len(_TYPE_KEYS):
            new_type = _TYPE_KEYS[idx]
            self._session.channels[row].signal_type = new_type
            self._session.channels[row].gain = DEFAULT_GAINS.get(
                new_type, DEFAULT_GAINS["other"]
            )
            spin = self._table.cellWidget(row, 3)
            if isinstance(spin, QtWidgets.QDoubleSpinBox):
                spin.blockSignals(True)
                spin.setValue(self._session.channels[row].gain)
                spin.blockSignals(False)

    def _on_gain_changed(self, row: int, value: float) -> None:
        if self._session:
            self._session.channels[row].gain = value

    def _select_all(self) -> None:
        self._set_all_checked(True)

    def _select_none(self) -> None:
        self._set_all_checked(False)

    def _select_psg(self) -> None:
        """Select only channels whose type is eeg/eog/emg/ecg."""
        if not self._session:
            return
        for row, ch in enumerate(self._session.channels):
            checked = ch.signal_type in ("eeg", "eog", "emg", "ecg")
            ch.selected = checked
            cb = self._table.cellWidget(row, 0)
            if isinstance(cb, QtWidgets.QCheckBox):
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
        self._update_warnings()

    def _set_all_checked(self, checked: bool) -> None:
        if not self._session:
            return
        for row, ch in enumerate(self._session.channels):
            ch.selected = checked
            cb = self._table.cellWidget(row, 0)
            if isinstance(cb, QtWidgets.QCheckBox):
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
        self._update_warnings()

    def _update_warnings(self) -> None:
        if not self._session:
            return
        selected = [ch for ch in self._session.channels if ch.selected]
        if not selected:
            self._warn_label.setText(
                "Attention : aucun canal selectionne. "
                "Selectionnez au moins un canal pour continuer."
            )
            return
        has_eeg = any(ch.signal_type == "eeg" for ch in selected)
        if not has_eeg:
            self._warn_label.setText(
                "Info : aucun canal EEG detecte parmi la selection. "
                "Verifiez les types si necessaire."
            )
            return
        self._warn_label.setText("")

    # -- Profile management --------------------------------------------------

    def _auto_apply_profile(self) -> None:
        """Try to find and apply a matching profile automatically."""
        if not self._session:
            return
        match = self._profile_store.find_matching(self._session.channel_names)
        if match:
            profile_data = self._profile_store.load(match)
            if profile_data:
                ch_list = profile_data.get("channels", profile_data) if isinstance(profile_data, dict) else profile_data
                n = ImportProfileStore.apply_profile_to_channels(
                    self._session.channels, ch_list
                )
                if isinstance(profile_data, dict):
                    gfe = profile_data.get("global_filter_enabled")
                    if gfe is not None:
                        self._session.global_filter_enabled = gfe
                self._profile_info.setText(
                    f"Profil « {match} » applique automatiquement ({n} canaux mis a jour)."
                )
                self._profile_info.setStyleSheet("color: #22c55e; font-style: italic;")
                logger.info("[PROFILE] Auto-applied '%s' (%d channels)", match, n)
                return
        self._profile_info.setText("")

    def _save_profile(self) -> None:
        """Save current channel configuration + filter pipelines as a profile."""
        if not self._session:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Sauvegarder le profil",
            "Nom du profil :",
            text=Path(self._session.file_path).stem,
        )
        if not ok or not name.strip():
            return

        # Grab current filter pipelines from the controller (if any)
        ctrl = self._wizard_ref._controller if self._wizard_ref else None
        filter_pipes: dict = {}
        global_filt = True
        if ctrl is not None:
            filter_pipes = getattr(ctrl, "channel_filter_pipelines", {})
            global_filt = getattr(ctrl, "filter_enabled", True)

        dicts = ImportProfileStore.channels_to_dicts(
            self._session.channels,
            filter_pipelines=filter_pipes,
        )
        path = self._profile_store.save(
            name.strip(), dicts, global_filter_enabled=global_filt,
        )
        QtWidgets.QMessageBox.information(
            self,
            "Profil sauvegarde",
            f"Profil « {name.strip()} » sauvegarde.\n({path.name})",
        )

    def _load_profile(self) -> None:
        """Let the user pick a profile to apply (channels + filters)."""
        if not self._session:
            return
        profiles = self._profile_store.list_profiles()
        if not profiles:
            QtWidgets.QMessageBox.information(
                self,
                "Aucun profil",
                "Aucun profil sauvegarde.\n"
                "Configurez vos canaux puis cliquez « Sauvegarder profil ».",
            )
            return
        name, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Charger un profil",
            "Profil :",
            profiles,
            editable=False,
        )
        if not ok or not name:
            return
        profile_data = self._profile_store.load(name)
        if profile_data:
            ch_list = profile_data.get("channels", profile_data) if isinstance(profile_data, dict) else profile_data
            n = ImportProfileStore.apply_profile_to_channels(
                self._session.channels, ch_list
            )
            if isinstance(profile_data, dict):
                gfe = profile_data.get("global_filter_enabled")
                if gfe is not None:
                    self._session.global_filter_enabled = gfe
            self._populate_table()
            self._profile_info.setText(
                f"Profil « {name} » charge ({n} canaux mis a jour)."
            )
            self._profile_info.setStyleSheet("color: #22c55e; font-style: italic;")
            logger.info("[PROFILE] Loaded '%s' (%d channels)", name, n)

    def has_selection(self) -> bool:
        if not self._session:
            return False
        return any(ch.selected for ch in self._session.channels)


# ======================================================================
# Page 4 -- Loading
# ======================================================================

class _LoadWorker(QtCore.QThread):
    """Background thread for ``AppController.load_recording``."""

    progress = QtCore.Signal(str)
    finished_ok = QtCore.Signal(bool)

    def __init__(
        self,
        controller,
        file_path: str,
        mode: str,
        ms_path: Optional[str],
        precompute_action: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._file_path = file_path
        self._mode = mode
        self._ms_path = ms_path
        self._precompute_action = precompute_action
        self._ok = False

    def run(self) -> None:
        try:
            self._ok = self._ctrl.load_recording(
                self._file_path,
                mode=self._mode,
                ms_path=self._ms_path,
                precompute_action=self._precompute_action,
                progress_callback=lambda msg: self.progress.emit(msg),
            )
        except Exception as exc:
            logger.error("Load worker error: %s", exc, exc_info=True)
            self._ok = False
        self.finished_ok.emit(self._ok)


class _PageLoading(QtWidgets.QWidget):
    """Step 4: progressive loading with progress bar."""

    load_complete = QtCore.Signal(bool)

    def __init__(self, theme: ThemePalette, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._worker: Optional[_LoadWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)

        title = QtWidgets.QLabel("Chargement en cours")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addSpacing(20)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setMinimumHeight(24)
        layout.addWidget(self._progress)
        layout.addSpacing(12)

        self._status = QtWidgets.QLabel("Preparation...")
        self._status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._status.setObjectName("subtitle")
        layout.addWidget(self._status)

        self._detail = QtWidgets.QLabel("")
        self._detail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._detail.setObjectName("dim")
        self._detail.setWordWrap(True)
        layout.addWidget(self._detail)

        layout.addStretch()

    def start_loading(
        self,
        controller,
        file_path: str,
        mode: str,
        ms_path: Optional[str],
        precompute_action: str,
    ) -> None:
        self._progress.setRange(0, 0)
        self._status.setText("Lecture du fichier...")
        self._detail.setText(Path(file_path).name)

        self._worker = _LoadWorker(
            controller, file_path, mode, ms_path, precompute_action, parent=self
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        self._status.setText(msg)

    def _on_finished(self, ok: bool) -> None:
        if ok:
            self._progress.setRange(0, 100)
            self._progress.setValue(100)
            self._status.setText("Chargement termine !")
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._status.setText("Erreur lors du chargement")
            self._status.setStyleSheet("color: #ef4444; font-weight: bold;")
        self.load_complete.emit(ok)


# ======================================================================
# Main Wizard Dialog
# ======================================================================

class EDFImportWizard(QtWidgets.QDialog):
    """Four-step EDF import wizard.

    Usage::

        wizard = EDFImportWizard(theme_name="dark", parent=main_window)
        result = wizard.run()  # blocking; returns EDFImportResult or None

    For pre-filling via drag-and-drop from the main window::

        wizard.set_initial_path("/path/to/file.edf")
    """

    def __init__(
        self,
        theme_name: str = "dark",
        controller=None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import EDF — CESA")
        self.resize(1020, 720)
        self.setMinimumSize(800, 560)

        self._controller = controller
        self._theme_name = theme_name
        self._theme: ThemePalette = dict(THEMES.get(theme_name, DARK))
        self._loader = EDFMetadataLoader()
        self._session: Optional[EDFSession] = None
        self._result: Optional[EDFImportResult] = None

        self._build_ui()
        self._apply_theme()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Step indicator
        self._step_bar = QtWidgets.QWidget()
        self._step_bar.setFixedHeight(50)
        self._step_labels: List[QtWidgets.QLabel] = []
        sb_lay = QtWidgets.QHBoxLayout(self._step_bar)
        sb_lay.setContentsMargins(24, 8, 24, 8)
        step_names = ["Selection", "Apercu", "Configuration", "Chargement"]
        for i, name in enumerate(step_names):
            lbl = QtWidgets.QLabel(f"  {i + 1}. {name}  ")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._step_labels.append(lbl)
            sb_lay.addWidget(lbl)
            if i < len(step_names) - 1:
                sep = QtWidgets.QLabel("›")
                sep.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                sep.setObjectName("dim")
                sb_lay.addWidget(sep)
        root.addWidget(self._step_bar)

        # Stacked pages
        self._stack = QtWidgets.QStackedWidget()
        self._page1 = _PageFileSelect(self._theme)
        self._page2 = _PagePreview(self._theme)
        self._page3 = _PageChannelConfig(self._theme)
        self._page3._wizard_ref = self
        self._page4 = _PageLoading(self._theme)
        self._stack.addWidget(self._page1)
        self._stack.addWidget(self._page2)
        self._stack.addWidget(self._page3)
        self._stack.addWidget(self._page4)
        root.addWidget(self._stack, stretch=1)

        # Navigation buttons
        nav = QtWidgets.QWidget()
        nav.setFixedHeight(56)
        nav_lay = QtWidgets.QHBoxLayout(nav)
        nav_lay.setContentsMargins(24, 8, 24, 8)
        self._btn_cancel = QtWidgets.QPushButton("Annuler")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_back = QtWidgets.QPushButton("Precedent")
        self._btn_back.clicked.connect(self._go_back)
        self._btn_next = QtWidgets.QPushButton("Suivant")
        self._btn_next.setObjectName("primary")
        self._btn_next.clicked.connect(self._go_next)
        nav_lay.addWidget(self._btn_cancel)
        nav_lay.addStretch()
        nav_lay.addWidget(self._btn_back)
        nav_lay.addWidget(self._btn_next)
        root.addWidget(nav)

        # Connections
        self._page4.load_complete.connect(self._on_load_complete)

        self._update_nav_buttons()
        self._update_step_indicator()

    # -- Theme -------------------------------------------------------------

    def _apply_theme(self) -> None:
        self.setStyleSheet(_theme_qss(self._theme))
        t = self._theme
        self._step_bar.setStyleSheet(
            f"background: {t['surface_alt']}; border-bottom: 1px solid {t['border']};"
        )
        self._update_step_indicator()

    def _update_step_indicator(self) -> None:
        t = self._theme
        current = self._stack.currentIndex()
        for i, lbl in enumerate(self._step_labels):
            if i == current:
                lbl.setStyleSheet(
                    f"color: {t['accent']}; font-weight: bold; "
                    f"background: {t['surface']}; border-radius: 4px;"
                )
            elif i < current:
                lbl.setStyleSheet(f"color: {t['foreground']}; font-weight: normal;")
            else:
                lbl.setStyleSheet(f"color: {t['text_dim']}; font-weight: normal;")

    # -- Navigation --------------------------------------------------------

    def _update_nav_buttons(self) -> None:
        idx = self._stack.currentIndex()
        self._btn_back.setVisible(idx > 0 and idx < 3)
        if idx == 2:
            self._btn_next.setText("Charger")
        elif idx == 3:
            self._btn_next.setVisible(False)
        else:
            self._btn_next.setText("Suivant")
            self._btn_next.setVisible(True)

    def _go_back(self) -> None:
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
            self._update_nav_buttons()
            self._update_step_indicator()

    def _go_next(self) -> None:
        idx = self._stack.currentIndex()

        if idx == 0:
            if not self._page1.is_valid():
                QtWidgets.QMessageBox.warning(
                    self, "Fichier invalide",
                    "Veuillez selectionner un fichier valide avant de continuer.",
                )
                return
            self._load_header()
            if self._session is None:
                QtWidgets.QMessageBox.warning(
                    self, "Erreur de lecture",
                    "Impossible de lire les metadonnees du fichier.",
                )
                return
            self._page2.set_session(self._session, self._loader)
            self._stack.setCurrentIndex(1)

        elif idx == 1:
            if self._session:
                self._page3.set_session(self._session)
            self._stack.setCurrentIndex(2)

        elif idx == 2:
            if not self._page3.has_selection():
                QtWidgets.QMessageBox.warning(
                    self, "Aucun canal",
                    "Selectionnez au moins un canal pour continuer.",
                )
                return
            self._stack.setCurrentIndex(3)
            self._start_loading()

        self._update_nav_buttons()
        self._update_step_indicator()

    # -- Header loading (fast, header-only) --------------------------------

    def _load_header(self) -> None:
        path = self._page1.file_path()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.setWindowTitle("Import EDF — CESA  (chargement...)")
        QtWidgets.QApplication.processEvents()
        try:
            self._session = self._loader.read_header(path)
        finally:
            self.setWindowTitle("Import EDF — CESA")
            QtWidgets.QApplication.restoreOverrideCursor()

    # -- Full loading (threaded) -------------------------------------------

    def _start_loading(self) -> None:
        if self._controller is None:
            logger.error("No AppController set -- cannot load")
            self._on_load_complete(False)
            return
        self._btn_cancel.setEnabled(False)
        self._page4.start_loading(
            controller=self._controller,
            file_path=self._page1.file_path(),
            mode=self._page1.mode(),
            ms_path=self._page1.ms_path(),
            precompute_action=self._page1.precompute_action(),
        )

    def _on_load_complete(self, ok: bool) -> None:
        self._btn_cancel.setEnabled(True)
        if ok and self._session is not None:
            self._result = EDFImportResult(
                session=self._session,
                mode=self._page1.mode(),
                ms_path=self._page1.ms_path(),
                precompute_action=self._page1.precompute_action(),
            )
            # Small delay so the user sees "100%" before closing
            QtCore.QTimer.singleShot(600, self.accept)
        else:
            self._btn_next.setVisible(True)
            self._btn_next.setText("Reessayer")
            self._btn_back.setVisible(True)
            self._stack.setCurrentIndex(0)
            self._update_nav_buttons()
            self._update_step_indicator()

    # -- Public API --------------------------------------------------------

    def set_initial_path(self, path: str) -> None:
        """Pre-fill file path (e.g. from drag-and-drop on main window)."""
        self._page1.set_file_path(path)

    def run(self) -> Optional[EDFImportResult]:
        """Execute the wizard modally. Returns result or ``None``."""
        if self.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return self._result
        self._loader.close()
        return None

    def closeEvent(self, event) -> None:
        self._loader.close()
        super().closeEvent(event)
