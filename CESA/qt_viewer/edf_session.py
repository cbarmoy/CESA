"""Data model for an opened EDF/BDF/FIF recording session.

``EDFSession`` is the single object exchanged between the import wizard
and the viewer.  Signals are kept as ``None`` until the user confirms
the import (lazy-loading strategy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ChannelInfo:
    """Metadata for a single recording channel (header-only, no signal data)."""

    name: str
    signal_type: str  # eeg | eog | emg | ecg | other
    sfreq: float
    physical_min: float
    physical_max: float
    unit: str  # uV, mV, ...
    n_samples: int
    selected: bool = True
    gain: float = 150.0  # default display gain in µV
    alias: str = ""  # user-defined display name (empty = use original name)
    filter_pipeline_dict: Optional[Dict] = field(default=None, repr=False)

    @property
    def display_name(self) -> str:
        """Return alias if set, otherwise original name."""
        return self.alias if self.alias else self.name


# Default gain per signal type (µV full-scale for the viewer)
DEFAULT_GAINS: Dict[str, float] = {
    "eeg": 150.0,
    "eog": 200.0,
    "emg": 50.0,
    "ecg": 500.0,
    "other": 200.0,
}


@dataclass
class EDFSession:
    """Complete session metadata for an EDF-family recording.

    ``signals`` stays ``None`` until the full file is loaded (step 4 of the
    wizard).  Preview chunks are fetched separately via
    ``EDFMetadataLoader.load_preview_chunk``.
    """

    file_path: Path
    channels: List[ChannelInfo]
    sfreq: float
    duration_s: float
    n_samples: int
    patient_info: Dict[str, str] = field(default_factory=dict)
    recording_date: Optional[datetime] = None
    signals: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    global_filter_enabled: Optional[bool] = None

    # --- Convenience helpers ------------------------------------------------

    @property
    def channel_names(self) -> List[str]:
        return [ch.name for ch in self.channels]

    @property
    def selected_channels(self) -> List[ChannelInfo]:
        return [ch for ch in self.channels if ch.selected]

    @property
    def selected_channel_names(self) -> List[str]:
        return [ch.name for ch in self.channels if ch.selected]

    @property
    def duration_hms(self) -> str:
        """Human-readable ``HH:MM:SS`` duration string."""
        total = int(self.duration_s)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def channel_type_mapping(self) -> Dict[str, str]:
        """Return ``{channel_name: signal_type}`` for selected channels."""
        return {ch.name: ch.signal_type for ch in self.channels if ch.selected}

    @property
    def rename_mapping(self) -> Dict[str, str]:
        """Return ``{original_name: alias}`` for channels that have an alias."""
        return {ch.name: ch.alias for ch in self.channels if ch.alias and ch.selected}


@dataclass
class EDFImportResult:
    """Value returned by the import wizard to the main window."""

    session: EDFSession
    mode: str = "raw"  # raw | precomputed
    ms_path: Optional[str] = None
    precompute_action: str = "existing"  # build | existing
