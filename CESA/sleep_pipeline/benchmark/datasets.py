"""Public dataset loaders for Sleep-EDF Expanded and MASS.

Each loader returns a list of :class:`DatasetRecord` objects that
encapsulate one recording (subject ID, EDF path, reference hypnogram).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sleep-EDF annotation mapping (R&K to AASM 5-class)
_SLEEPEDF_STAGE_MAP: Dict[str, str] = {
    "W": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",  # R&K S4 -> AASM N3
    "R": "R",
    "M": "W",   # Movement -> Wake
    "?": "U",
}

# MASS annotation variants
_MASS_STAGE_MAP: Dict[str, str] = {
    "1": "N1", "2": "N2", "3": "N3", "4": "N3",
    "5": "R", "R": "R", "W": "W", "0": "W", "?": "U",
}


@dataclass
class DatasetRecord:
    """One polysomnography recording with reference scoring."""

    subject_id: str
    night: int = 1
    edf_path: str = ""
    hypnogram_path: str = ""
    labels: List[str] = field(default_factory=list)
    epoch_duration_s: float = 30.0
    recording_duration_s: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------
# Sleep-EDF Expanded
# ---------------------------------------------------------------------

def load_sleep_edf(
    *,
    n_subjects: int = -1,
    data_path: str = "",
    age_group: str = "age",
    epoch_duration_s: float = 30.0,
) -> List[DatasetRecord]:
    """Load the Sleep-EDF Expanded dataset via MNE.

    Parameters
    ----------
    n_subjects : int
        Max subjects to load (-1 = all available).
    data_path : str
        Override download/cache directory (empty = MNE default).
    age_group : str
        ``"age"`` for the age study or ``"temazepam"`` for the
        temazepam study.
    epoch_duration_s : float
        Epoch length in seconds (standard 30 s).

    Returns
    -------
    list of DatasetRecord
    """
    import mne
    from mne.datasets.sleep_physionet import age as sp_age, temazepam as sp_tem

    fetcher = sp_age if age_group == "age" else sp_tem

    # Discover available subjects
    try:
        if age_group == "age":
            all_subjects = list(range(83))
        else:
            all_subjects = list(range(22))
    except Exception:
        all_subjects = list(range(20))

    if 0 < n_subjects < len(all_subjects):
        all_subjects = all_subjects[:n_subjects]

    records: List[DatasetRecord] = []

    for subj in all_subjects:
        for night_idx in (1, 2):
            try:
                fetch_kw = {"subjects": [subj], "recording": [night_idx]}
                if data_path:
                    fetch_kw["path"] = data_path
                paths = fetcher.fetch_data(**fetch_kw)
            except Exception:
                continue

            if not paths:
                continue

            for psg_path, hyp_path in paths:
                psg_path = str(psg_path)
                hyp_path = str(hyp_path)

                try:
                    labels = _parse_sleepedf_hypnogram(
                        hyp_path, epoch_duration_s,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to parse hypnogram for subject %d night %d: %s",
                        subj, night_idx, exc,
                    )
                    continue

                if not labels:
                    continue

                # Trim trailing Movement/Unknown epochs
                labels = _trim_trailing(labels)

                raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
                dur_s = raw.times[-1] if len(raw.times) > 0 else 0.0

                records.append(DatasetRecord(
                    subject_id=f"SC{subj:02d}",
                    night=night_idx,
                    edf_path=psg_path,
                    hypnogram_path=hyp_path,
                    labels=labels,
                    epoch_duration_s=epoch_duration_s,
                    recording_duration_s=dur_s,
                    metadata={"dataset": "sleep_edf", "age_group": age_group},
                ))

    logger.info("Loaded %d Sleep-EDF records from %d subjects",
                len(records), len({r.subject_id for r in records}))
    return records


def _parse_sleepedf_hypnogram(
    hyp_path: str,
    epoch_duration_s: float = 30.0,
) -> List[str]:
    """Parse a Sleep-EDF ``*-Hypnogram.edf`` file to a label list."""
    import mne

    ann = mne.read_annotations(hyp_path)
    labels: List[str] = []

    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        desc_str = str(desc).strip()
        if "Sleep stage" not in desc_str:
            continue
        code = desc_str.split()[-1]
        stage = _SLEEPEDF_STAGE_MAP.get(code, "U")
        n_epochs = max(1, int(round(duration / epoch_duration_s)))
        labels.extend([stage] * n_epochs)

    return labels


def _trim_trailing(labels: List[str]) -> List[str]:
    """Remove trailing U/Movement-only epochs."""
    while labels and labels[-1] == "U":
        labels.pop()
    return labels


# ---------------------------------------------------------------------
# MASS dataset
# ---------------------------------------------------------------------

def load_mass(
    *,
    data_path: str,
    subset: str = "SS3",
    n_subjects: int = -1,
    epoch_duration_s: float = 30.0,
) -> List[DatasetRecord]:
    """Load MASS dataset from a local directory.

    The MASS dataset is not freely downloadable (requires license).
    This function expects the data to be pre-downloaded.

    Parameters
    ----------
    data_path : str
        Root directory containing MASS data with EDF and annotation files.
    subset : str
        MASS subset (``"SS1"``..``"SS5"``).
    n_subjects : int
        Max subjects (-1 = all).
    epoch_duration_s : float
        Epoch length.
    """
    root = Path(data_path)
    if not root.is_dir():
        raise FileNotFoundError(f"MASS data_path not found: {data_path}")

    edf_files = sorted(root.glob("**/*PSG*.edf")) + sorted(root.glob("**/*psg*.edf"))
    if not edf_files:
        edf_files = sorted(root.glob("**/*.edf"))

    records: List[DatasetRecord] = []
    seen_subjects = set()

    for edf_path in edf_files:
        # Extract subject ID from filename
        match = re.search(r"(\d{2}-\d{4}|\d{4})", edf_path.stem)
        subj_id = match.group(0) if match else edf_path.stem

        if 0 < n_subjects <= len(seen_subjects) and subj_id not in seen_subjects:
            continue
        seen_subjects.add(subj_id)

        # Find matching annotation file
        ann_candidates = [
            edf_path.with_suffix(".txt"),
            edf_path.parent / (edf_path.stem.replace("PSG", "Base") + ".edf"),
            edf_path.parent / (edf_path.stem + "_annotations.txt"),
        ]

        labels: List[str] = []
        for ann_path in ann_candidates:
            if ann_path.exists():
                try:
                    labels = _parse_mass_annotations(str(ann_path), epoch_duration_s)
                except Exception:
                    continue
                if labels:
                    break

        if not labels:
            logger.warning("No annotations found for MASS subject %s", subj_id)
            continue

        import mne
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
            dur_s = raw.times[-1]
        except Exception:
            dur_s = len(labels) * epoch_duration_s

        records.append(DatasetRecord(
            subject_id=subj_id,
            night=1,
            edf_path=str(edf_path),
            hypnogram_path=str(ann_candidates[0]) if ann_candidates else "",
            labels=labels,
            epoch_duration_s=epoch_duration_s,
            recording_duration_s=dur_s,
            metadata={"dataset": "mass", "subset": subset},
        ))

    logger.info("Loaded %d MASS records", len(records))
    return records


def _parse_mass_annotations(
    ann_path: str,
    epoch_duration_s: float = 30.0,
) -> List[str]:
    """Parse MASS text-based annotation file."""
    p = Path(ann_path)
    labels: List[str] = []

    if p.suffix == ".edf":
        import mne
        ann = mne.read_annotations(ann_path)
        for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
            code = str(desc).strip().split()[-1]
            stage = _MASS_STAGE_MAP.get(code, "U")
            n = max(1, int(round(duration / epoch_duration_s)))
            labels.extend([stage] * n)
    else:
        for line in p.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            code = parts[-1] if parts else "?"
            labels.append(_MASS_STAGE_MAP.get(code, "U"))

    return labels


# ---------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    *,
    data_path: str = "",
    n_subjects: int = -1,
    epoch_duration_s: float = 30.0,
    **kwargs,
) -> List[DatasetRecord]:
    """Load a dataset by name.

    Parameters
    ----------
    dataset_name : str
        ``"sleep_edf"`` or ``"mass"``.
    """
    if dataset_name == "sleep_edf":
        return load_sleep_edf(
            n_subjects=n_subjects,
            data_path=data_path,
            epoch_duration_s=epoch_duration_s,
            **kwargs,
        )
    elif dataset_name == "mass":
        if not data_path:
            raise ValueError("MASS requires data_path (licensed dataset)")
        return load_mass(
            data_path=data_path,
            n_subjects=n_subjects,
            epoch_duration_s=epoch_duration_s,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
