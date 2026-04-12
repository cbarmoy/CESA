"""Interactive annotation system for the EEG viewer.

Provides structured annotation storage, rendering as coloured regions
on the EEG plot, and create / edit / delete interaction.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .themes import DARK, ThemePalette

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------

ANNOTATION_TYPES: Dict[str, str] = {
    "artifact": "#F38BA8",
    "arousal": "#F9E2AF",
    "apnea": "#FAB387",
    "hypopnea": "#FAB387",
    "rem_burst": "#CBA6F7",
    "spindle": "#89B4FA",
    "k_complex": "#A6E3A1",
    "note": "#94E2D5",
    "custom": "#6C7086",
}


@dataclass
class Annotation:
    """Single annotation on the EEG recording."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    onset_s: float = 0.0
    duration_s: float = 1.0
    ann_type: str = "artifact"
    channel: str = ""  # empty = global (all channels)
    text: str = ""
    color: str = ""  # override; empty = use type default
    confidence: float = 1.0
    source: str = "manual"  # "manual", "auto", "ml"

    def effective_color(self) -> str:
        return self.color or ANNOTATION_TYPES.get(self.ann_type, "#6C7086")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Annotation":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class AnnotationStore:
    """In-memory store with change notification and JSON persistence."""

    changed = None  # set to a QtCore.Signal instance by the widget

    def __init__(self) -> None:
        self._items: Dict[str, Annotation] = {}
        self._listeners: List[Any] = []

    def add(self, ann: Annotation) -> str:
        self._items[ann.id] = ann
        self._notify()
        return ann.id

    def remove(self, ann_id: str) -> Optional[Annotation]:
        removed = self._items.pop(ann_id, None)
        if removed:
            self._notify()
        return removed

    def update(self, ann_id: str, **kwargs) -> Optional[Annotation]:
        ann = self._items.get(ann_id)
        if ann is None:
            return None
        for k, v in kwargs.items():
            if hasattr(ann, k):
                setattr(ann, k, v)
        self._notify()
        return ann

    def get(self, ann_id: str) -> Optional[Annotation]:
        return self._items.get(ann_id)

    def all(self) -> List[Annotation]:
        return sorted(self._items.values(), key=lambda a: a.onset_s)

    def in_range(self, start_s: float, end_s: float) -> List[Annotation]:
        return [
            a for a in self._items.values()
            if a.onset_s + a.duration_s > start_s and a.onset_s < end_s
        ]

    def by_type(self, ann_type: str) -> List[Annotation]:
        return sorted(
            [a for a in self._items.values() if a.ann_type == ann_type],
            key=lambda a: a.onset_s,
        )

    def next_of_type(self, ann_type: str, after_s: float) -> Optional[Annotation]:
        for a in self.by_type(ann_type):
            if a.onset_s > after_s + 0.01:
                return a
        return None

    def prev_of_type(self, ann_type: str, before_s: float) -> Optional[Annotation]:
        candidates = [a for a in self.by_type(ann_type) if a.onset_s < before_s - 0.01]
        return candidates[-1] if candidates else None

    def clear(self) -> None:
        self._items.clear()
        self._notify()

    def __len__(self) -> int:
        return len(self._items)

    def add_listener(self, cb) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in self._listeners:
            try:
                cb()
            except Exception:
                pass

    def save_json(self, path: str) -> None:
        data = [a.to_dict() for a in self.all()]
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_json(self, path: str) -> int:
        text = Path(path).read_text(encoding="utf-8")
        items = json.loads(text)
        count = 0
        for d in items:
            try:
                self.add(Annotation.from_dict(d))
                count += 1
            except Exception:
                pass
        return count

    def to_event_list(self) -> List[Dict[str, Any]]:
        """Convert to the event list format used by EventsBar."""
        return [
            {
                "onset": a.onset_s,
                "duration": a.duration_s,
                "type": a.ann_type,
                "label": a.text or a.ann_type,
                "id": a.id,
            }
            for a in self.all()
        ]


# -----------------------------------------------------------------------
# Overlay renderer
# -----------------------------------------------------------------------

class AnnotationOverlay:
    """Draws annotation regions on a PlotItem."""

    def __init__(self, plot_item: pg.PlotItem, store: AnnotationStore) -> None:
        self._plot = plot_item
        self._store = store
        self._items: Dict[str, pg.LinearRegionItem] = {}
        self._labels: Dict[str, pg.TextItem] = {}
        store.add_listener(self._on_store_changed)

    def update_visible(self, start_s: float, end_s: float) -> None:
        visible = self._store.in_range(start_s, end_s)
        visible_ids = {a.id for a in visible}

        # Remove items no longer visible
        for aid in list(self._items.keys()):
            if aid not in visible_ids:
                self._plot.removeItem(self._items.pop(aid))
                lbl = self._labels.pop(aid, None)
                if lbl:
                    self._plot.removeItem(lbl)

        # Add / update visible
        for ann in visible:
            if ann.id not in self._items:
                color = ann.effective_color()
                brush = pg.mkBrush(color + "30")
                region = pg.LinearRegionItem(
                    values=[ann.onset_s, ann.onset_s + ann.duration_s],
                    movable=False,
                    brush=brush,
                    pen=pg.mkPen(color, width=1),
                )
                region.setZValue(-5)
                self._plot.addItem(region)
                self._items[ann.id] = region

                if ann.text:
                    lbl = pg.TextItem(text=ann.text[:20], color=color, anchor=(0, 1))
                    lbl.setFont(QtGui.QFont("Segoe UI", 7))
                    lbl.setPos(ann.onset_s, 0)
                    self._plot.addItem(lbl)
                    self._labels[ann.id] = lbl
            else:
                self._items[ann.id].setRegion(
                    [ann.onset_s, ann.onset_s + ann.duration_s]
                )

    def clear(self) -> None:
        for item in self._items.values():
            self._plot.removeItem(item)
        for lbl in self._labels.values():
            self._plot.removeItem(lbl)
        self._items.clear()
        self._labels.clear()

    def _on_store_changed(self) -> None:
        pass  # will be refreshed on next update_visible call


# -----------------------------------------------------------------------
# Create annotation dialog
# -----------------------------------------------------------------------

class AnnotationDialog(QtWidgets.QDialog):
    """Modal dialog for creating / editing an annotation."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        annotation: Optional[Annotation] = None,
        onset_s: float = 0.0,
        duration_s: float = 1.0,
        channel: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Annotation" if annotation is None else "Modifier annotation")
        self.setMinimumWidth(360)
        self._result: Optional[Annotation] = None
        editing = annotation is not None
        ann = annotation or Annotation(
            onset_s=onset_s, duration_s=duration_s, channel=channel,
        )

        form = QtWidgets.QFormLayout(self)

        self._type_combo = QtWidgets.QComboBox()
        self._type_combo.addItems(list(ANNOTATION_TYPES.keys()))
        self._type_combo.setCurrentText(ann.ann_type)
        form.addRow("Type:", self._type_combo)

        self._onset = QtWidgets.QDoubleSpinBox()
        self._onset.setRange(0, 1e6)
        self._onset.setDecimals(2)
        self._onset.setSuffix(" s")
        self._onset.setValue(ann.onset_s)
        form.addRow("Debut:", self._onset)

        self._duration = QtWidgets.QDoubleSpinBox()
        self._duration.setRange(0.1, 3600)
        self._duration.setDecimals(2)
        self._duration.setSuffix(" s")
        self._duration.setValue(ann.duration_s)
        form.addRow("Duree:", self._duration)

        self._channel = QtWidgets.QLineEdit(ann.channel)
        self._channel.setPlaceholderText("(vide = global)")
        form.addRow("Canal:", self._channel)

        self._text = QtWidgets.QLineEdit(ann.text)
        self._text.setPlaceholderText("Note optionnelle")
        form.addRow("Texte:", self._text)

        self._source = QtWidgets.QComboBox()
        self._source.addItems(["manual", "auto", "ml"])
        self._source.setCurrentText(ann.source)
        form.addRow("Source:", self._source)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        if editing:
            del_btn = QtWidgets.QPushButton("Supprimer")
            del_btn.setStyleSheet("color: #F38BA8;")
            del_btn.clicked.connect(lambda: self.done(2))
            form.addRow(del_btn)

        self._ann_id = ann.id if editing else None

    def _accept(self) -> None:
        self._result = Annotation(
            id=self._ann_id or uuid.uuid4().hex[:12],
            onset_s=self._onset.value(),
            duration_s=self._duration.value(),
            ann_type=self._type_combo.currentText(),
            channel=self._channel.text().strip(),
            text=self._text.text().strip(),
            source=self._source.currentText(),
        )
        self.accept()

    @property
    def annotation(self) -> Optional[Annotation]:
        return self._result
