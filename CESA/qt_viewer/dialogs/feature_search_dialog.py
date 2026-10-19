"""Search dialog to locate a menu feature by name, synonym or shortcut."""

from __future__ import annotations

import unicodedata
from typing import List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

# Extra keywords (French + English) so a search still works after menu moves.
_ALIASES = {
    "ouvrir un enregistrement": "edf edf+ charger fichier import psg",
    "rapport html": "export html rapport",
    "rapport pdf": "export pdf rapport",
    "annotations json": "export annotations",
    "segment edf (plage temporelle)": "export edf extraire crop",
    "theme sombre": "dark mode nuit",
    "theme clair": "light mode jour",
    "afficher signal brut": "raw underlay non filtre",
    "normaliser les amplitudes (ctrl+n)": "gain amplitude scale",
    "aller au temps": "goto jump hh:mm:ss",
    "inspection": "canal details hover",
    "dashboard": "tableau de bord snr qc resume",
    "scaling": "gain uV amplitude echelle",
    "layout": "espacement preset canaux",
    "spectrogramme": "ondelette morlet tfr bandeau nuit fenetre",
    "configuration des filtres": "bandpass highpass lowpass notch filtre",
    "nettoyage oculaire": "eog yeux artefact blink",
    "nettoyage ica": "ica composantes independantes artefact",
    "scoring automatique": "auto score yasa hypnogramme stades",
    "editeur de scoring manuel": "corriger stades manuel",
    "depuis fichier (csv/excel/json)": "import scoring excel csv",
    "depuis annotations edf+": "import scoring edf annotations",
    "sauvegarder scoring (csv/json)": "export scoring",
    "auto-avancer après scoring manuel": "auto advance epoque suivante",
    "fft scoring (époque ciblée)": "fft welch psd epoque scoring",
    "definir": "fenetre analyse night lights off debut",
    "effacer": "fenetre analyse night lights on reset",
    "exporter csv": "rapport psg csv metriques",
    "graphiques": "rapport psg graphiques figures",
    "comparer auto vs manuel": "agreement kappa comparaison scoring",
    "informations scoring": "stats hypnogramme resume",
    "detecter micro-architecture (spindles/so/rem)": (
        "spindle fuseaux k-complexe slow oscillation rem burst"
    ),
    "detecter les arousals": "arousal micro-eveil",
    "detecter apnees / hypopnees": "apnee hypopnee sahos respiration",
    "detecter desaturations spo2": "spo2 desat oxymetrie",
    "nouvelle annotation": "marquer evenement note",
    "marqueurs temporels": "markers curseur labels",
    "analyse spectrale (psd welch)": "psd fft welch spectre puissance",
    "psd par stade (fft - analyse_spectrale)": "psd fft stade bande",
    "psd par stade (welch) avant/apres": "psd welch avant apres",
    "specparam (fooof / 1/f)": "fooof aperiodic 1/f specparam",
    "spectrogramme ondelettes (avant/apres)": "morlet cwt wavelet tfr",
    "qc nuit": "qualite snr plat bruit secteur",
    "detection d'artefacts": "artefact mauvais canaux",
    "entropie (mse / renormalisee)": "mse entropy complexite",
    "analyse temporelle (rms, kurtosis...)": "rms kurtosis variance",
    "connectivite (coherence, correlation, plv)": "coherence plv correlation",
    "micro-etats (k-means)": "microstates kmeans cartes",
    "periodes de sommeil (sleepeegpy)": "sleepeegpy periodes",
    "automatisation en lot (fft / spaghetti / psg)": "batch lot multi fichiers spaghetti",
    "analyse groupe (mse/ren/hrv)": "groupe hrv mse statistique",
    "raccourcis clavier": "hotkeys shortcuts clavier",
    "documentation en ligne": "docs github aide",
}


def fold_text(text: str) -> str:
    """Lowercase ASCII fold: 'Époque' -> 'epoque'."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    cleaned = stripped.replace("&", " ").replace("...", " ")
    return " ".join(cleaned.casefold().split())


_ALIASES_FOLDED = {fold_text(key): value for key, value in _ALIASES.items()}


def action_search_haystack(label: str, path: str, shortcut: str) -> str:
    extra = _ALIASES_FOLDED.get(fold_text(label), "")
    return fold_text(f"{label} {path} {shortcut} {extra}")


def query_matches(query: str, haystack: str) -> bool:
    """True if every query word appears in the folded haystack."""
    words = fold_text(query).split()
    if not words:
        return True
    return all(word in haystack for word in words)


def collect_menu_features(
    menubar: QtWidgets.QMenuBar,
    *,
    skip_object_names: Tuple[str, ...] = (),
) -> List[Tuple[QtGui.QAction, str, str, str]]:
    """Walk the menu bar. Returns (action, label, path, shortcut)."""
    out: List[Tuple[QtGui.QAction, str, str, str]] = []

    def _walk(menu: QtWidgets.QMenu, prefix: str) -> None:
        for action in menu.actions():
            if action.isSeparator():
                continue
            submenu = action.menu()
            label = (action.text() or "").replace("&", "").strip()
            if not label:
                continue
            path = f"{prefix} > {label}" if prefix else label
            if submenu is not None:
                _walk(submenu, path)
                continue
            if action.objectName() in skip_object_names:
                continue
            shortcut = action.shortcut().toString() if action.shortcut() else ""
            out.append((action, label, prefix, shortcut))

    for top in menubar.actions():
        menu = top.menu()
        if menu is None:
            continue
        title = (top.text() or "").replace("&", "").strip()
        _walk(menu, title)
    return out


class FeatureSearchDialog(QtWidgets.QDialog):
    """Type-ahead search over every menu action."""

    def __init__(self, parent: QtWidgets.QMainWindow) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rechercher une fonctionnalite")
        self.resize(720, 460)
        self._parent_window = parent
        self._rows: List[Tuple[QtGui.QAction, str, str, str]] = []
        self._build_ui()
        self._reload()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "Tapez un mot (ex. spindle, filtre, PSD, rapport) : "
            "la liste se filtre, Entree lance la fonction."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #A6ADC8;")
        layout.addWidget(hint)

        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText("Rechercher une fonctionnalite...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        self._search.returnPressed.connect(self._activate_selected)
        layout.addWidget(self._search)

        self._table = QtWidgets.QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(
            ["Fonctionnalite", "Emplacement", "Raccourci"]
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.doubleClicked.connect(lambda *_: self._activate_selected())
        layout.addWidget(self._table, stretch=1)

        bottom = QtWidgets.QHBoxLayout()
        self._count = QtWidgets.QLabel("")
        self._count.setStyleSheet("color: #A6ADC8;")
        bottom.addWidget(self._count)
        bottom.addStretch(1)
        open_btn = QtWidgets.QPushButton("Ouvrir")
        open_btn.clicked.connect(self._activate_selected)
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.reject)
        bottom.addWidget(open_btn)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    def _reload(self) -> None:
        self._rows = collect_menu_features(
            self._parent_window.menuBar(),
            skip_object_names=("featureSearch",),
        )
        self._apply_filter()

    def _apply_filter(self) -> None:
        query = self._search.text()
        self._table.setRowCount(0)
        matches = 0
        for action, label, path, shortcut in self._rows:
            hay = action_search_haystack(label, path, shortcut)
            if not query_matches(query, hay):
                continue
            row = self._table.rowCount()
            self._table.insertRow(row)
            items = (
                QtWidgets.QTableWidgetItem(label),
                QtWidgets.QTableWidgetItem(path),
                QtWidgets.QTableWidgetItem(shortcut),
            )
            dim = not action.isEnabled()
            for col, item in enumerate(items):
                item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
                if dim:
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#6C7086")))
                self._table.setItem(row, col, item)
            self._table.item(row, 0).setData(QtCore.Qt.ItemDataRole.UserRole, action)
            matches += 1
        if matches:
            self._table.selectRow(0)
        self._count.setText(f"{matches} resultat{'s' if matches != 1 else ''}")

    def _selected_action(self) -> Optional[QtGui.QAction]:
        items = self._table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        item = self._table.item(row, 0)
        if item is None:
            return None
        action = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return action if isinstance(action, QtGui.QAction) else None

    def _activate_selected(self) -> None:
        action = self._selected_action()
        if action is None:
            return
        path_item = self._table.item(self._table.currentRow(), 1)
        path = path_item.text() if path_item is not None else ""
        if not action.isEnabled():
            QtWidgets.QMessageBox.information(
                self,
                "Fonctionnalite indisponible",
                "Cette fonctionnalite est desactivee pour l'instant "
                "(chargez un enregistrement si besoin).\n\n"
                f"Emplacement : {path}",
            )
            return
        self.accept()
        action.trigger()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self._search.setFocus()
        self._search.selectAll()
