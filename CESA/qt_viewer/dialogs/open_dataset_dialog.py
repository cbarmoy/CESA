"""Qt dialog to select a recording file and choose the loading mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets

from core.raw_loader import SUPPORTED_RECORDING_EXTENSIONS


@dataclass
class OpenDatasetSelection:
    edf_path: str
    mode: str            # "raw" or "precomputed"
    ms_path: Optional[str]
    precompute_action: str  # "build" or "existing"


class OpenDatasetDialog(QtWidgets.QDialog):
    """Gather recording path, navigation mode, and optional multiscale settings."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ouvrir un enregistrement")
        self.setMinimumWidth(580)
        self.result: Optional[OpenDatasetSelection] = None
        self._ms_path_user_override = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # -- Recording selection --
        edf_group = QtWidgets.QGroupBox("Fichier d'enregistrement")
        edf_lay = QtWidgets.QHBoxLayout(edf_group)
        self._edf_edit = QtWidgets.QLineEdit()
        self._edf_edit.setPlaceholderText("Chemin vers le fichier EDF/BDF/FIF...")
        edf_lay.addWidget(self._edf_edit, stretch=1)
        browse_btn = QtWidgets.QPushButton("Parcourir...")
        browse_btn.clicked.connect(self._browse_edf)
        edf_lay.addWidget(browse_btn)
        layout.addWidget(edf_group)

        info = QtWidgets.QLabel("Choisissez le fichier de signal a analyser (EDF/BDF/FIF...).")
        info.setStyleSheet("color: #888;")
        layout.addWidget(info)

        # -- Mode selection --
        mode_group = QtWidgets.QGroupBox("Mode de lecture")
        mode_lay = QtWidgets.QVBoxLayout(mode_group)
        self._mode_raw = QtWidgets.QRadioButton("Mode standard (lecture directe)")
        self._mode_raw.setChecked(True)
        self._mode_precomp = QtWidgets.QRadioButton("Navigation rapide (fichier pre-calcule)")
        mode_lay.addWidget(self._mode_raw)
        mode_lay.addWidget(self._mode_precomp)

        # Precomputed options (initially hidden)
        self._precomp_frame = QtWidgets.QWidget()
        pf_lay = QtWidgets.QVBoxLayout(self._precomp_frame)
        pf_lay.setContentsMargins(20, 4, 0, 0)
        pf_lay.addWidget(QtWidgets.QLabel("Dossier de navigation rapide (Zarr) :"))
        ms_row = QtWidgets.QHBoxLayout()
        self._ms_edit = QtWidgets.QLineEdit()
        ms_row.addWidget(self._ms_edit, stretch=1)
        ms_browse = QtWidgets.QPushButton("Selectionner...")
        ms_browse.clicked.connect(self._browse_ms)
        ms_row.addWidget(ms_browse)
        pf_lay.addLayout(ms_row)
        self._action_build = QtWidgets.QRadioButton("Creer / mettre a jour automatiquement")
        self._action_build.setChecked(True)
        self._action_existing = QtWidgets.QRadioButton("Utiliser un fichier existant")
        pf_lay.addWidget(self._action_build)
        pf_lay.addWidget(self._action_existing)
        helper = QtWidgets.QLabel(
            "Le fichier sera genere si besoin (quelques minutes selon la taille)."
        )
        helper.setStyleSheet("color: #888;")
        pf_lay.addWidget(helper)
        mode_lay.addWidget(self._precomp_frame)
        self._precomp_frame.setVisible(False)
        layout.addWidget(mode_group)

        self._mode_precomp.toggled.connect(self._toggle_mode)
        self._edf_edit.textChanged.connect(self._on_edf_change)

        # -- Buttons --
        btn_lay = QtWidgets.QHBoxLayout()
        btn_lay.addStretch()
        open_btn = QtWidgets.QPushButton("Ouvrir")
        open_btn.setDefault(True)
        open_btn.clicked.connect(self._confirm)
        cancel_btn = QtWidgets.QPushButton("Annuler")
        cancel_btn.clicked.connect(self.reject)
        btn_lay.addWidget(open_btn)
        btn_lay.addWidget(cancel_btn)
        layout.addLayout(btn_lay)

    # ------------------------------------------------------------------

    def _toggle_mode(self, checked: bool) -> None:
        self._precomp_frame.setVisible(checked)
        if checked:
            self._on_edf_change()

    def _browse_edf(self) -> None:
        exts = " ".join(f"*{e}" for e in SUPPORTED_RECORDING_EXTENSIONS)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Selectionner un enregistrement",
            "",
            f"Fichiers enregistrement ({exts});;Tous les fichiers (*.*)",
        )
        if path:
            self._edf_edit.setText(path)

    def _browse_ms(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Selectionner un dossier de navigation rapide",
        )
        if path:
            self._ms_path_user_override = True
            self._ms_edit.setText(path)

    def _on_edf_change(self) -> None:
        if not self._mode_precomp.isChecked():
            return
        if self._ms_path_user_override:
            return
        edf = self._edf_edit.text().strip()
        if edf:
            default_ms = str(Path(edf).with_suffix("")) + "_ms"
            self._ms_edit.setText(default_ms)

    def _confirm(self) -> None:
        edf_path = self._edf_edit.text().strip()
        if not edf_path:
            QtWidgets.QMessageBox.warning(self, "Fichier manquant", "Veuillez selectionner un fichier.")
            return
        if not Path(edf_path).exists():
            QtWidgets.QMessageBox.warning(self, "Fichier introuvable", "Le fichier selectionne n'existe pas.")
            return

        mode = "precomputed" if self._mode_precomp.isChecked() else "raw"
        ms_path: Optional[str] = None
        precompute_action = "existing"

        if mode == "precomputed":
            ms_candidate = self._ms_edit.text().strip()
            if not ms_candidate:
                QtWidgets.QMessageBox.warning(
                    self, "Chemin requis",
                    "Indiquez un dossier pour le fichier de navigation rapide.",
                )
                return
            ms_path = ms_candidate
            precompute_action = "build" if self._action_build.isChecked() else "existing"

        self.result = OpenDatasetSelection(
            edf_path=edf_path,
            mode=mode,
            ms_path=ms_path,
            precompute_action=precompute_action,
        )
        self.accept()

    def get_result(self) -> Optional[OpenDatasetSelection]:
        if self.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return self.result
        return None
