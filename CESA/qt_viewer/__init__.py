"""CESA PyQtGraph-based EEG/PSG Viewer.

High-performance signal viewer using PyQtGraph, designed as the sole
GUI for the CESA application (no Tkinter dependency).

Public API
----------
``launch_viewer``
    Opens the Qt viewer window.
``ViewerBridge``
    Adapter exposing the same surface as ``PSGPlotter`` so existing
    code can drive the viewer without changes.
``Annotation``, ``AnnotationStore``
    Interactive annotation data model with persistence.
``ReportBuilder``
    Publication-ready HTML/PDF report generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .annotations import Annotation, AnnotationStore  # noqa: F401
    from .report_export import ReportBuilder  # noqa: F401
    from .viewer_bridge import ViewerBridge  # noqa: F401

__all__ = [
    "launch_viewer",
    "ViewerBridge",
    "Annotation",
    "AnnotationStore",
    "ReportBuilder",
]


def launch_viewer(**kwargs) -> "ViewerBridge":
    """Create and show the Qt EEG viewer, returning a bridge handle.

    Handles ``QApplication`` lifecycle automatically.
    Parameters match :class:`ViewerBridge` / ``EEGViewerMainWindow``.
    """
    from .viewer_bridge import create_and_launch

    return create_and_launch(**kwargs)
