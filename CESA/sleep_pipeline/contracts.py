"""Canonical data types for the CESA sleep-scoring pipeline.

Every module in ``sleep_pipeline`` communicates through the types defined
here so that each stage of the pipeline can be developed, tested and
replaced independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stage labels
# ---------------------------------------------------------------------------

class StageLabel(str, Enum):
    """AASM sleep-stage labels plus a technical *Undefined* marker."""

    W = "W"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    R = "R"
    U = "U"  # technical only -- not a clinical stage

    @classmethod
    def from_string(cls, value: str) -> "StageLabel":
        """Robust normalisation of free-text stage labels."""
        _MAP = {
            "w": cls.W, "wake": cls.W, "eveil": cls.W, "éveil": cls.W,
            "veille": cls.W,
            "n1": cls.N1,
            "n2": cls.N2,
            "n3": cls.N3, "n4": cls.N3, "sws": cls.N3, "slow wave": cls.N3,
            "r": cls.R, "rem": cls.R, "paradoxal": cls.R,
            "u": cls.U, "unk": cls.U, "unknown": cls.U,
            "artefact": cls.U, "artifact": cls.U,
            "movement": cls.U, "mt": cls.U, "mvt": cls.U,
        }
        key = str(value).strip().lower()
        return _MAP.get(key, cls.U)

    @property
    def is_sleep(self) -> bool:
        return self in (StageLabel.N1, StageLabel.N2, StageLabel.N3, StageLabel.R)


CLINICAL_STAGES: List[StageLabel] = [
    StageLabel.W, StageLabel.N1, StageLabel.N2, StageLabel.N3, StageLabel.R,
]
CLINICAL_STAGE_STRINGS: List[str] = [s.value for s in CLINICAL_STAGES]


# ---------------------------------------------------------------------------
# Event labels
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Respiratory and EEG events scored during a PSG."""

    AROUSAL = "arousal"
    APNEA_OBSTRUCTIVE = "apnea_obstructive"
    APNEA_CENTRAL = "apnea_central"
    APNEA_MIXED = "apnea_mixed"
    HYPOPNEA = "hypopnea"
    DESAT = "desaturation"


@dataclass(frozen=True)
class ScoredEvent:
    """Single scored event with onset / duration / confidence."""

    event_type: EventType
    onset_s: float
    duration_s: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

@dataclass
class Epoch:
    """One 30-s epoch with its features, label and provenance."""

    index: int
    start_s: float
    duration_s: float = 30.0
    features: Optional[Dict[str, float]] = None
    stage: StageLabel = StageLabel.U
    confidence: float = 0.0
    decision_reason: str = ""


# ---------------------------------------------------------------------------
# Full scoring result
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Complete output of one scoring pass."""

    epochs: List[Epoch] = field(default_factory=list)
    events: List[ScoredEvent] = field(default_factory=list)
    epoch_duration_s: float = 30.0
    backend: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- Convenience helpers ------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert epoch list to a legacy-compatible DataFrame(time, stage)."""
        rows = [
            {
                "time": ep.start_s,
                "stage": ep.stage.value,
                "confidence": ep.confidence,
                "reason": ep.decision_reason,
            }
            for ep in self.epochs
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["time", "stage", "confidence", "reason"]
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        epoch_duration_s: float = 30.0,
        backend: str = "legacy",
    ) -> "ScoringResult":
        """Build a ScoringResult from a legacy DataFrame(time, stage)."""
        epochs: List[Epoch] = []
        for idx, row in df.iterrows():
            stage = StageLabel.from_string(str(row.get("stage", "U")))
            conf = float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else 0.0
            epochs.append(Epoch(
                index=int(idx),
                start_s=float(row["time"]),
                duration_s=epoch_duration_s,
                stage=stage,
                confidence=conf,
            ))
        return cls(epochs=epochs, epoch_duration_s=epoch_duration_s, backend=backend)

    @property
    def stages(self) -> List[str]:
        return [ep.stage.value for ep in self.epochs]

    @property
    def times(self) -> np.ndarray:
        return np.array([ep.start_s for ep in self.epochs], dtype=float)

    def stage_array(self) -> np.ndarray:
        """Numeric array: W=0, N1=1, N2=2, N3=3, R=4, U=-1."""
        _map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "U": -1}
        return np.array([_map.get(ep.stage.value, -1) for ep in self.epochs], dtype=int)
