"""Custom PyQtGraph axis that displays time as HH:MM:SS."""

from __future__ import annotations

import pyqtgraph as pg


class TimeAxisItem(pg.AxisItem):
    """Bottom-axis that converts seconds into ``HH:MM:SS`` labels.

    Designed for PSG recordings where the X values are in seconds from
    recording start.
    """

    def __init__(self, orientation: str = "bottom", **kw):
        super().__init__(orientation, **kw)

    # ------------------------------------------------------------------
    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        """Override to format tick values as ``HH:MM:SS``."""
        return [self._fmt(v * scale) for v in values]

    @staticmethod
    def _fmt(seconds: float) -> str:
        s = max(0.0, float(seconds))
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"
