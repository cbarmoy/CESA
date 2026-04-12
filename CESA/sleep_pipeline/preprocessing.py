"""Signal preprocessing and 30-s epoching for the sleep pipeline.

Responsibilities
----------------
* Select and validate EEG / EOG / EMG channels from a Raw object.
* Resample to a common target frequency.
* Segment the continuous recording into fixed 30-s epochs.
* Apply basic artifact rejection per epoch.
* Optionally use ``FilterPipeline`` objects from ``CESA.filter_engine``
  to replace the hardcoded bandpass step (when provided via *filter_overrides*).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .contracts import Epoch, StageLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel selection helpers
# ---------------------------------------------------------------------------

# Common clinical names per modality (priority order)
EEG_CANDIDATES = [
    "C4-M1", "C3-M2", "Fpz-Cz", "Pz-Oz", "F4-M1", "F3-M2",
    "C4", "C3", "Fpz", "Pz",
]
EOG_CANDIDATES = [
    "E1-M2", "E2-M1", "EOG L", "EOG R", "EOG LEFT", "EOG RIGHT",
    "EOG GAUCHE", "EOG DROIT",
]
EMG_CANDIDATES = [
    "Chin1-Chin2", "Chin1-Chin3", "EMG Chin", "CHIN", "MENTON",
]


def pick_channel(
    raw: mne.io.BaseRaw,
    candidates: Sequence[str],
    *,
    kind: str = "EEG",
    required: bool = True,
) -> Optional[str]:
    """Return the first available channel from *candidates* (case-insensitive)."""
    available_upper = {ch.upper(): ch for ch in raw.ch_names}
    for name in candidates:
        resolved = available_upper.get(name.upper().strip())
        if resolved is not None:
            return resolved
    if required:
        raise RuntimeError(
            f"No {kind} channel found among candidates: {candidates}"
        )
    return None


# ---------------------------------------------------------------------------
# Preprocessing config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """All tuneable knobs for the preprocessing step.

    The optional *filter_overrides* dict maps a channel modality tag
    (``"eeg"``, ``"eog"``, ``"emg"``) to a ``FilterPipeline`` from
    ``CESA.filter_engine``.  When present the pipeline is used **instead**
    of the default ``_bandpass`` call for that modality.
    """

    target_sfreq: float = 100.0
    epoch_duration_s: float = 30.0
    eeg_bandpass: Tuple[float, float] = (0.3, 35.0)
    eog_bandpass: Tuple[float, float] = (0.3, 10.0)
    emg_bandpass: Tuple[float, float] = (10.0, 100.0)
    filter_order: int = 4
    artifact_uv_threshold: float = 500.0  # reject epoch if peak-to-peak > this
    eeg_candidates: Sequence[str] = ()
    eog_candidates: Sequence[str] = ()
    emg_candidates: Sequence[str] = ()
    filter_overrides: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Epoched output container
# ---------------------------------------------------------------------------

@dataclass
class EpochedSignals:
    """Holds the result of preprocessing: per-epoch signal arrays + metadata."""

    eeg_epochs: np.ndarray  # (n_epochs, n_samples)
    eog_epochs: Optional[np.ndarray] = None
    emg_epochs: Optional[np.ndarray] = None
    sfreq: float = 100.0
    epoch_duration_s: float = 30.0
    n_epochs: int = 0
    eeg_channel: str = ""
    eog_channel: Optional[str] = None
    emg_channel: Optional[str] = None
    rejected_mask: Optional[np.ndarray] = None  # bool, True = artifact

    @property
    def epoch_samples(self) -> int:
        return int(round(self.sfreq * self.epoch_duration_s))


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------

def _bandpass(data: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = sfreq / 2.0
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 0.9999)
    if lo >= hi:
        return data
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float64)


def preprocess(
    raw: mne.io.BaseRaw,
    config: Optional[PreprocessingConfig] = None,
) -> EpochedSignals:
    """Full preprocessing: channel selection, resample, filter, epoch, artifact check.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object (loaded or lazy).
    config : PreprocessingConfig, optional
        If *None* a default config is used.

    Returns
    -------
    EpochedSignals
        Ready-to-use epoched data.
    """
    if config is None:
        config = PreprocessingConfig()

    eeg_cands = list(config.eeg_candidates) or EEG_CANDIDATES
    eog_cands = list(config.eog_candidates) or EOG_CANDIDATES
    emg_cands = list(config.emg_candidates) or EMG_CANDIDATES

    eeg_ch = pick_channel(raw, eeg_cands, kind="EEG", required=True)
    eog_ch = pick_channel(raw, eog_cands, kind="EOG", required=False)
    emg_ch = pick_channel(raw, emg_cands, kind="EMG", required=False)

    channels = [eeg_ch]
    if eog_ch:
        channels.append(eog_ch)
    if emg_ch:
        channels.append(emg_ch)

    raw_work = raw.copy().pick(channels).load_data()
    sfreq = float(raw_work.info["sfreq"])

    if not np.isclose(sfreq, config.target_sfreq, atol=0.5):
        raw_work.resample(config.target_sfreq, npad="auto")
        sfreq = config.target_sfreq

    epoch_samples = int(round(sfreq * config.epoch_duration_s))
    total_samples = raw_work.n_times
    n_epochs = total_samples // epoch_samples
    if n_epochs == 0:
        raise RuntimeError(
            f"Recording too short for a single {config.epoch_duration_s}s epoch "
            f"({total_samples} samples at {sfreq} Hz)."
        )

    def _extract_epochs(
        ch_name: Optional[str],
        bandpass: Tuple[float, float],
        modality_key: str,
    ) -> Optional[np.ndarray]:
        if ch_name is None:
            return None
        data = raw_work.get_data(picks=[ch_name])[0] * 1e6  # V -> uV

        override = config.filter_overrides.get(modality_key)
        if override is not None:
            try:
                data = override.apply(data, sfreq)
            except Exception:
                data = _bandpass(data, sfreq, bandpass[0], bandpass[1], config.filter_order)
        else:
            data = _bandpass(data, sfreq, bandpass[0], bandpass[1], config.filter_order)

        usable = data[: n_epochs * epoch_samples]
        return usable.reshape(n_epochs, epoch_samples)

    eeg_epochs = _extract_epochs(eeg_ch, config.eeg_bandpass, "eeg")
    eog_epochs = _extract_epochs(eog_ch, config.eog_bandpass, "eog")
    emg_epochs = _extract_epochs(emg_ch, config.emg_bandpass, "emg")

    # Simple artifact rejection: mark epochs exceeding peak-to-peak threshold
    rejected = np.zeros(n_epochs, dtype=bool)
    if eeg_epochs is not None:
        ptp = np.ptp(eeg_epochs, axis=1)
        rejected = ptp > config.artifact_uv_threshold

    n_rejected = int(rejected.sum())
    if n_rejected:
        logger.info(
            "Artifact rejection: %d / %d epochs exceed %.0f uV p-t-p",
            n_rejected, n_epochs, config.artifact_uv_threshold,
        )

    return EpochedSignals(
        eeg_epochs=eeg_epochs,
        eog_epochs=eog_epochs,
        emg_epochs=emg_epochs,
        sfreq=sfreq,
        epoch_duration_s=config.epoch_duration_s,
        n_epochs=n_epochs,
        eeg_channel=eeg_ch or "",
        eog_channel=eog_ch,
        emg_channel=emg_ch,
        rejected_mask=rejected,
    )
