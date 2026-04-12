"""Filter pipeline metrics for the Qt viewer (no Qt imports)."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def count_effective_filter_channels(pipelines: Dict[str, Any]) -> int:
    """Channels with pipeline enabled and at least one enabled filter stage."""
    n = 0
    for p in pipelines.values():
        if p is None:
            continue
        try:
            if not bool(getattr(p, "enabled", True)):
                continue
            filters = getattr(p, "filters", None) or []
            if not filters:
                continue
            if any(bool(getattr(f, "enabled", True)) for f in filters):
                n += 1
        except Exception:
            logger.debug("count_effective_filter_channels: skip entry", exc_info=True)
    return n
