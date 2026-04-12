"""Intelligent navigation: jump to events, stage changes, search.

Extends the navigation bar with targeted jump commands and a search
system for quickly locating events, stages, and annotations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


class SmartNavigator(QtCore.QObject):
    """Jump-to logic for events, stage transitions, and annotations."""

    navigate_to = QtCore.Signal(float)  # time in seconds

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._events: List[Dict[str, Any]] = []
        self._hypnogram: List[str] = []
        self._epoch_len: float = 30.0
        self._current_time: float = 0.0
        self._annotations: List[Any] = []

    # ---- data setters -------------------------------------------------

    def set_events(self, events: List[Dict[str, Any]]) -> None:
        self._events = sorted(events, key=lambda e: e.get("onset", 0))

    def set_hypnogram(self, labels: List[str], epoch_len: float) -> None:
        self._hypnogram = list(labels)
        self._epoch_len = epoch_len

    def set_annotations(self, annotations: List[Any]) -> None:
        self._annotations = annotations

    def set_current_time(self, t: float) -> None:
        self._current_time = t

    # ---- jump methods -------------------------------------------------

    def next_event(self, event_type: Optional[str] = None) -> Optional[float]:
        """Jump to the next event of given type (or any event)."""
        for e in self._events:
            onset = e.get("onset", 0.0)
            if onset <= self._current_time + 0.5:
                continue
            if event_type and e.get("type", "").lower() != event_type.lower():
                continue
            self.navigate_to.emit(onset)
            return onset
        return None

    def prev_event(self, event_type: Optional[str] = None) -> Optional[float]:
        """Jump to the previous event of given type."""
        candidates = []
        for e in self._events:
            onset = e.get("onset", 0.0)
            if onset >= self._current_time - 0.5:
                break
            if event_type and e.get("type", "").lower() != event_type.lower():
                continue
            candidates.append(onset)
        if candidates:
            t = candidates[-1]
            self.navigate_to.emit(t)
            return t
        return None

    def next_stage_change(self) -> Optional[float]:
        """Jump to the next hypnogram stage transition."""
        if not self._hypnogram:
            return None
        current_idx = int(self._current_time / self._epoch_len)
        for i in range(current_idx + 1, len(self._hypnogram)):
            if i > 0 and self._hypnogram[i] != self._hypnogram[i - 1]:
                t = i * self._epoch_len
                self.navigate_to.emit(t)
                return t
        return None

    def prev_stage_change(self) -> Optional[float]:
        if not self._hypnogram:
            return None
        current_idx = int(self._current_time / self._epoch_len)
        for i in range(min(current_idx, len(self._hypnogram) - 1), 0, -1):
            if self._hypnogram[i] != self._hypnogram[i - 1]:
                t = i * self._epoch_len
                self.navigate_to.emit(t)
                return t
        return None

    def jump_to_stage(self, stage: str) -> Optional[float]:
        """Jump to the next epoch of a specific stage."""
        if not self._hypnogram:
            return None
        current_idx = int(self._current_time / self._epoch_len)
        s_upper = stage.upper()
        for i in range(current_idx + 1, len(self._hypnogram)):
            if self._hypnogram[i].upper() in (s_upper, stage):
                t = i * self._epoch_len
                self.navigate_to.emit(t)
                return t
        return None

    def search(self, query: str) -> List[Tuple[float, str]]:
        """Search events, stages, and annotations by text query.

        Returns list of (time_s, description) matches.
        """
        q = query.strip().lower()
        results: List[Tuple[float, str]] = []

        # Search events
        for e in self._events:
            etype = e.get("type", "")
            label = e.get("label", "")
            if q in etype.lower() or q in label.lower():
                results.append((e.get("onset", 0.0), f"{etype}: {label}"))

        # Search hypnogram stages
        s_upper = q.upper()
        for i, s in enumerate(self._hypnogram):
            if s.upper() == s_upper or s.upper() == f"N{q}" or q in s.lower():
                results.append((i * self._epoch_len, f"Stage {s}"))

        # Search annotations
        for ann in self._annotations:
            txt = getattr(ann, "text", "") or ""
            atype = getattr(ann, "ann_type", "") or ""
            if q in txt.lower() or q in atype.lower():
                results.append((getattr(ann, "onset_s", 0.0), f"Ann: {atype} {txt}"))

        results.sort(key=lambda x: x[0])
        # Deduplicate consecutive stages
        if len(results) > 200:
            results = results[:200]
        return results


class SmartNavWidget(QtWidgets.QWidget):
    """Toolbar widget with smart navigation controls and search."""

    navigate_to = QtCore.Signal(float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._navigator = SmartNavigator(self)
        self._navigator.navigate_to.connect(self.navigate_to)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Event navigation
        self._event_type = QtWidgets.QComboBox()
        self._event_type.addItems([
            "Tout", "arousal", "apnea", "hypopnea",
            "artifact", "rem_burst", "spindle",
        ])
        self._event_type.setFixedWidth(100)
        layout.addWidget(self._event_type)

        btn_prev_ev = QtWidgets.QPushButton("< Evt")
        btn_prev_ev.setToolTip("Evenement precedent")
        btn_prev_ev.clicked.connect(self._prev_event)
        layout.addWidget(btn_prev_ev)

        btn_next_ev = QtWidgets.QPushButton("Evt >")
        btn_next_ev.setToolTip("Evenement suivant")
        btn_next_ev.clicked.connect(self._next_event)
        layout.addWidget(btn_next_ev)

        layout.addWidget(self._vsep())

        # Stage transition navigation
        btn_prev_trans = QtWidgets.QPushButton("< Trans")
        btn_prev_trans.setToolTip("Transition de stade precedente")
        btn_prev_trans.clicked.connect(
            lambda: self._navigator.prev_stage_change()
        )
        layout.addWidget(btn_prev_trans)

        btn_next_trans = QtWidgets.QPushButton("Trans >")
        btn_next_trans.setToolTip("Transition de stade suivante")
        btn_next_trans.clicked.connect(
            lambda: self._navigator.next_stage_change()
        )
        layout.addWidget(btn_next_trans)

        layout.addWidget(self._vsep())

        # Search
        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText("Rechercher (REM, N3, apnea...)")
        self._search.setFixedWidth(200)
        self._search.returnPressed.connect(self._do_search)
        layout.addWidget(self._search)

        self._search_results = QtWidgets.QComboBox()
        self._search_results.setFixedWidth(180)
        self._search_results.activated.connect(self._on_result_selected)
        layout.addWidget(self._search_results)

        layout.addStretch()

        self._search_data: List[Tuple[float, str]] = []

    @property
    def navigator(self) -> SmartNavigator:
        return self._navigator

    def set_current_time(self, t: float) -> None:
        self._navigator.set_current_time(t)

    def _prev_event(self) -> None:
        et = self._event_type.currentText()
        self._navigator.prev_event(None if et == "Tout" else et)

    def _next_event(self) -> None:
        et = self._event_type.currentText()
        self._navigator.next_event(None if et == "Tout" else et)

    def _do_search(self) -> None:
        query = self._search.text().strip()
        if not query:
            return
        results = self._navigator.search(query)
        self._search_data = results
        self._search_results.clear()
        for t, desc in results[:50]:
            h, m = int(t // 3600), int((t % 3600) // 60)
            s = int(t % 60)
            self._search_results.addItem(f"{h:02d}:{m:02d}:{s:02d} - {desc}")
        if results:
            self.navigate_to.emit(results[0][0])

    def _on_result_selected(self, idx: int) -> None:
        if 0 <= idx < len(self._search_data):
            self.navigate_to.emit(self._search_data[idx][0])

    @staticmethod
    def _vsep() -> QtWidgets.QFrame:
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setFixedWidth(2)
        return sep
