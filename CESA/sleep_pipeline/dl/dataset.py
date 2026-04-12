"""PyTorch Dataset for multi-channel sleep epochs.

Each sample is a (C, T) tensor where C is the number of input channels
(EEG + optional EOG/EMG) and T = sfreq * epoch_duration samples.

Labels are integer-encoded: W=0, N1=1, N2=2, N3=3, R=4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError("PyTorch is required for the DL module: pip install torch")

from ..contracts import StageLabel
from ..preprocessing import EpochedSignals

_STAGE_TO_INT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


class SleepEpochDataset(Dataset):
    """PyTorch Dataset wrapping pre-processed epoch arrays.

    Parameters
    ----------
    epoched : EpochedSignals
        Output of ``preprocessing.preprocess``.
    labels : sequence of str, optional
        Stage labels aligned with epochs.  If *None*, all labels default
        to -1 (for inference without ground truth).
    transform : callable, optional
        Applied to each (C, T) numpy array before conversion to tensor.
    """

    def __init__(
        self,
        epoched: EpochedSignals,
        labels: Optional[Sequence[str]] = None,
        transform=None,
    ) -> None:
        super().__init__()
        self.sfreq = epoched.sfreq
        self.epoch_samples = epoched.epoch_samples
        self.transform = transform

        # Stack channels: (n_epochs, C, T)
        channels = [epoched.eeg_epochs]
        if epoched.eog_epochs is not None:
            channels.append(epoched.eog_epochs)
        if epoched.emg_epochs is not None:
            channels.append(epoched.emg_epochs)
        self.data = np.stack(channels, axis=1).astype(np.float32)

        if labels is not None:
            self.labels = np.array(
                [_STAGE_TO_INT.get(StageLabel.from_string(s).value, -1) for s in labels],
                dtype=np.int64,
            )
        else:
            self.labels = np.full(self.data.shape[0], -1, dtype=np.int64)

        # Exclude artifact epochs from training
        self.valid_mask = np.ones(len(self.data), dtype=bool)
        if epoched.rejected_mask is not None:
            self.valid_mask &= ~epoched.rejected_mask
        if labels is not None:
            self.valid_mask &= self.labels >= 0

        self._valid_indices = np.where(self.valid_mask)[0]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self._valid_indices[idx]
        x = self.data[real_idx]  # (C, T)
        if self.transform is not None:
            x = self.transform(x)
        return torch.from_numpy(x), int(self.labels[real_idx])

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def n_classes(self) -> int:
        return 5

    @staticmethod
    def from_files(
        edf_path: str,
        scoring_path: Optional[str] = None,
        *,
        epoch_duration_s: float = 30.0,
        target_sfreq: float = 100.0,
    ) -> "SleepEpochDataset":
        """Convenience constructor: load EDF + optional scoring CSV."""
        import mne
        import pandas as pd
        from ..preprocessing import PreprocessingConfig, preprocess

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        config = PreprocessingConfig(target_sfreq=target_sfreq, epoch_duration_s=epoch_duration_s)
        epoched = preprocess(raw, config)

        labels = None
        if scoring_path is not None:
            df = pd.read_csv(scoring_path)
            # Align labels to epochs by time
            labels_list: List[str] = []
            for i in range(epoched.n_epochs):
                t = i * epoch_duration_s
                diffs = np.abs(df["time"].values - t)
                best = int(np.argmin(diffs))
                if diffs[best] < epoch_duration_s / 2:
                    labels_list.append(str(df["stage"].iloc[best]))
                else:
                    labels_list.append("U")
            labels = labels_list

        return SleepEpochDataset(epoched, labels=labels)
