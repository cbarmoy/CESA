"""Display-only channel traces (clones) for the EEG viewer.

A physical recording channel (``source_name``) can appear multiple times
with independent label, colour, filter pipeline and gain.  Trace identity
is ``trace_id``; the first view of a channel keeps ``trace_id == source_name``
so existing profiles and scoring paths remain compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DisplayTrace:
    """One visual strip sharing source data with optional independent settings."""

    trace_id: str
    source_name: str
    label: str
    color: Optional[str] = None  # None → theme colour by signal type
    gain: Optional[float] = None  # None → type / scaling defaults
    visible: bool = True

    @property
    def is_clone(self) -> bool:
        return self.trace_id != self.source_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "source_name": self.source_name,
            "label": self.label,
            "color": self.color,
            "gain": self.gain,
            "visible": bool(self.visible),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisplayTrace":
        source = str(data.get("source_name", "") or "").strip()
        tid = str(data.get("trace_id", "") or source).strip() or source
        label = str(data.get("label", "") or tid).strip() or tid
        gain_raw = data.get("gain")
        gain = float(gain_raw) if gain_raw is not None else None
        return cls(
            trace_id=tid,
            source_name=source or tid,
            label=label,
            color=(str(data["color"]).strip() if data.get("color") else None) or None,
            gain=gain,
            visible=bool(data.get("visible", True)),
        )


def default_traces_for_sources(source_names: List[str]) -> List[DisplayTrace]:
    """One primary trace per physical channel (``trace_id == source_name``)."""
    return [
        DisplayTrace(trace_id=n, source_name=n, label=n)
        for n in source_names
    ]


def next_clone_id(source_name: str, existing_ids: List[str]) -> str:
    """Allocate ``source::2``, ``source::3``, … unique among *existing_ids*."""
    n = 2
    while True:
        candidate = f"{source_name}::{n}"
        if candidate not in existing_ids:
            return candidate
        n += 1


def next_clone_label(source_name: str, existing_labels: List[str]) -> str:
    """Human label ``source (2)``, ``source (3)``, …"""
    n = 2
    while True:
        candidate = f"{source_name} ({n})"
        if candidate not in existing_labels:
            return candidate
        n += 1


def traces_to_dict_list(traces: List[DisplayTrace]) -> List[Dict[str, Any]]:
    return [t.to_dict() for t in traces]


def traces_from_dict_list(items: List[Dict[str, Any]]) -> List[DisplayTrace]:
    out: List[DisplayTrace] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            out.append(DisplayTrace.from_dict(item))
        except Exception:
            continue
    return out
