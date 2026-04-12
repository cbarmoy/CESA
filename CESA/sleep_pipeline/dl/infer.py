"""Inference entry-point for the DL sleep-staging model.

Loads a trained checkpoint, runs prediction on a Raw object, and returns
a standard ``ScoringResult`` compatible with the rest of the CESA pipeline.

Usage
-----
::

    from CESA.sleep_pipeline.dl.infer import predict_dl
    result = predict_dl(raw, checkpoint_path="models/dl/best_model.pt")
    df = result.to_dataframe()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required: pip install torch")

from ..contracts import Epoch, ScoringResult, StageLabel
from ..preprocessing import EpochedSignals, PreprocessingConfig, preprocess
from ..rules_aasm import smooth_stages
from .model import SleepStagingModel, SleepAttentionModel, SleepTransformer

logger = logging.getLogger(__name__)

_INT_TO_STAGE = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}


def predict_dl(
    raw,
    *,
    checkpoint_path: str,
    epoch_duration_s: float = 30.0,
    target_sfreq: float = 100.0,
    n_channels: int = 1,
    base_filters: int = 64,
    model_type: str = "cnn",
    n_heads: int = 4,
    transformer_layers: int = 2,
    dim_feedforward: int = 256,
    device: str = "auto",
    apply_smoothing: bool = True,
) -> ScoringResult:
    """Run DL inference on a recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object.
    checkpoint_path : str
        Path to a ``best_model.pt`` file saved by ``train.py``.
    epoch_duration_s : float
        Epoch duration in seconds.
    target_sfreq : float
        Resampling frequency.
    n_channels : int
        Number of input channels the model was trained on.
    base_filters : int
        CNN filter base matching training config.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    apply_smoothing : bool
        Apply transition-rule smoothing.

    Returns
    -------
    ScoringResult
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    config = PreprocessingConfig(target_sfreq=target_sfreq, epoch_duration_s=epoch_duration_s)
    epoched = preprocess(raw, config)

    # Build tensor: (n_epochs, C, T)
    channels_list = [epoched.eeg_epochs]
    if epoched.eog_epochs is not None:
        channels_list.append(epoched.eog_epochs)
    if epoched.emg_epochs is not None:
        channels_list.append(epoched.emg_epochs)
    actual_channels = len(channels_list)

    # Handle channel mismatch: pad or truncate to match model's expected n_channels
    data = np.stack(channels_list, axis=1).astype(np.float32)
    if actual_channels < n_channels:
        pad = np.zeros((data.shape[0], n_channels - actual_channels, data.shape[2]), dtype=np.float32)
        data = np.concatenate([data, pad], axis=1)
    elif actual_channels > n_channels:
        data = data[:, :n_channels, :]

    # Load model
    if model_type == "attention":
        model = SleepAttentionModel(
            n_channels=n_channels, n_classes=5, base_filters=base_filters, n_heads=n_heads,
        )
    elif model_type == "transformer":
        model = SleepTransformer(
            n_channels=n_channels, n_classes=5, base_filters=base_filters,
            n_heads=n_heads, n_layers=transformer_layers, dim_feedforward=dim_feedforward,
        )
    else:
        model = SleepStagingModel(n_channels=n_channels, n_classes=5, base_filters=base_filters)
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"DL checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(str(ckpt), map_location=dev)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev)
    model.eval()

    # Predict in batches
    batch_size = 128
    is_sequence_model = model_type in ("attention", "transformer")
    all_logits = []
    with torch.no_grad():
        if is_sequence_model:
            # Sequence models expect (batch, seq_len, C, T) -- process as one sequence
            seq = torch.from_numpy(data).unsqueeze(0).to(dev)  # (1, T, C, samples)
            logits = model(seq)  # (1, T, n_classes)
            if isinstance(logits, tuple):
                logits = logits[0]
            all_logits.append(logits.squeeze(0).cpu().numpy())
        else:
            for start in range(0, len(data), batch_size):
                batch = torch.from_numpy(data[start: start + batch_size]).to(dev)
                logits = model(batch)
                all_logits.append(logits.cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)  # (n_epochs, 5)
    preds = logits_arr.argmax(axis=1)
    probs = _softmax(logits_arr)

    # Build ScoringResult
    epochs = []
    for i in range(len(preds)):
        stage_str = _INT_TO_STAGE.get(int(preds[i]), "U")
        stage = StageLabel.from_string(stage_str)
        conf = float(probs[i, preds[i]])
        is_artifact = bool(epoched.rejected_mask[i]) if epoched.rejected_mask is not None else False
        if is_artifact:
            stage = StageLabel.U
            conf = 0.0

        epochs.append(Epoch(
            index=i,
            start_s=i * epoch_duration_s,
            duration_s=epoch_duration_s,
            stage=stage,
            confidence=conf,
            decision_reason=f"dl_pred={stage_str}",
        ))

    if apply_smoothing and len(epochs) > 1:
        raw_stages = [ep.stage for ep in epochs]
        raw_confs = [ep.confidence for ep in epochs]
        smoothed = smooth_stages(raw_stages, raw_confs)
        for ep, new_stage in zip(epochs, smoothed):
            if new_stage != ep.stage:
                ep.decision_reason += f"|smoothed_from_{ep.stage.value}"
                ep.stage = new_stage

    return ScoringResult(
        epochs=epochs,
        epoch_duration_s=epoch_duration_s,
        backend="dl_cnn",
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
