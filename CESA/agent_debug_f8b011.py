"""Session debug NDJSON (f8b011) — retirer après validation."""
from __future__ import annotations

import json
import time
from pathlib import Path

_LOG = Path(__file__).resolve().parents[1] / "debug-f8b011.log"
_SESSION = "f8b011"


def log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
    *,
    run_id: str = "run1",
) -> None:
    try:
        rec = {
            "sessionId": _SESSION,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
