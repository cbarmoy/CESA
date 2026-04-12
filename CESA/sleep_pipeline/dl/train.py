"""Training loop for the DL sleep-staging model.

Supports:
- Single-epoch CNN (``SleepStagingModel``)
- Sequence-level CNN+LSTM (``SleepStagingLSTM``)
- Class-weighted cross-entropy for imbalanced stages
- Early stopping on validation Cohen's kappa
- Checkpoint saving

Usage
-----
::

    python -m CESA.sleep_pipeline.dl.train --config config/dl_train.json

Or programmatically::

    from CESA.sleep_pipeline.dl.train import Trainer, TrainConfig
    trainer = Trainer(config)
    trainer.run()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
except ImportError:
    raise ImportError("PyTorch is required: pip install torch")

from .dataset import SleepEpochDataset
from .model import SleepStagingModel, SleepStagingLSTM, SleepAttentionModel, SleepTransformer

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """All training hyperparameters."""

    # Data
    edf_paths: List[str] = field(default_factory=list)
    scoring_paths: List[str] = field(default_factory=list)
    epoch_duration_s: float = 30.0
    target_sfreq: float = 100.0

    # Model
    model_type: str = "cnn"  # "cnn", "lstm", "attention", or "transformer"
    base_filters: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.5
    n_heads: int = 4
    transformer_layers: int = 2
    dim_feedforward: int = 256

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    val_split: float = 0.2

    # Output
    output_dir: str = "models/dl"
    device: str = "auto"

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Trainer:
    """Encapsulates the training loop."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _build_dataset(self) -> SleepEpochDataset:
        """Concatenate multiple recordings into one dataset."""
        import mne
        from ..preprocessing import PreprocessingConfig, preprocess

        all_eeg: List[np.ndarray] = []
        all_eog: List[Optional[np.ndarray]] = []
        all_emg: List[Optional[np.ndarray]] = []
        all_labels: List[str] = []
        all_rejected: List[np.ndarray] = []

        for edf_path, score_path in zip(self.config.edf_paths, self.config.scoring_paths):
            logger.info("Loading %s", edf_path)
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            cfg = PreprocessingConfig(
                target_sfreq=self.config.target_sfreq,
                epoch_duration_s=self.config.epoch_duration_s,
            )
            epoched = preprocess(raw, cfg)

            import pandas as pd
            df = pd.read_csv(score_path)
            labels: List[str] = []
            for i in range(epoched.n_epochs):
                t = i * self.config.epoch_duration_s
                diffs = np.abs(df["time"].values - t)
                best = int(np.argmin(diffs))
                if diffs[best] < self.config.epoch_duration_s / 2:
                    labels.append(str(df["stage"].iloc[best]).strip().upper())
                else:
                    labels.append("U")

            all_eeg.append(epoched.eeg_epochs)
            all_eog.append(epoched.eog_epochs)
            all_emg.append(epoched.emg_epochs)
            all_labels.extend(labels)
            if epoched.rejected_mask is not None:
                all_rejected.append(epoched.rejected_mask)
            else:
                all_rejected.append(np.zeros(epoched.n_epochs, dtype=bool))

        # Build a combined EpochedSignals-like wrapper
        from ..preprocessing import EpochedSignals
        combined = EpochedSignals(
            eeg_epochs=np.concatenate(all_eeg, axis=0),
            eog_epochs=np.concatenate([e for e in all_eog if e is not None], axis=0) if any(e is not None for e in all_eog) else None,
            emg_epochs=np.concatenate([e for e in all_emg if e is not None], axis=0) if any(e is not None for e in all_emg) else None,
            sfreq=self.config.target_sfreq,
            epoch_duration_s=self.config.epoch_duration_s,
            n_epochs=sum(len(e) for e in all_eeg),
            rejected_mask=np.concatenate(all_rejected, axis=0),
        )
        return SleepEpochDataset(combined, labels=all_labels)

    def run(self) -> Dict[str, Any]:
        """Execute training and return metrics."""
        dataset = self._build_dataset()
        n_val = max(1, int(len(dataset) * self.config.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=0)

        # Class weights from training set
        train_labels = np.array([dataset.labels[i] for i in train_ds.indices])
        valid_labels = train_labels[train_labels >= 0]
        if len(valid_labels) == 0:
            raise RuntimeError("No valid training labels found.")
        counts = np.bincount(valid_labels, minlength=5).astype(float)
        counts[counts == 0] = 1.0
        weights = (1.0 / counts) / (1.0 / counts).sum() * 5.0
        class_weights = torch.FloatTensor(weights).to(self.device)

        # Build model
        if self.config.model_type == "lstm":
            model = SleepStagingLSTM(
                n_channels=dataset.n_channels,
                n_classes=5,
                base_filters=self.config.base_filters,
                lstm_hidden=self.config.lstm_hidden,
                lstm_layers=self.config.lstm_layers,
                dropout=self.config.dropout,
            )
        elif self.config.model_type == "attention":
            model = SleepAttentionModel(
                n_channels=dataset.n_channels,
                n_classes=5,
                base_filters=self.config.base_filters,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
            )
        elif self.config.model_type == "transformer":
            model = SleepTransformer(
                n_channels=dataset.n_channels,
                n_classes=5,
                base_filters=self.config.base_filters,
                n_heads=self.config.n_heads,
                n_layers=self.config.transformer_layers,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
            )
        else:
            model = SleepStagingModel(
                n_channels=dataset.n_channels,
                n_classes=5,
                base_filters=self.config.base_filters,
                dropout=self.config.dropout,
            )
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

        best_kappa = -1.0
        patience_counter = 0
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_path = output_dir / "best_model.pt"

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_kappa": [], "val_acc": []}

        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.time()

            # -- Train --
            model.train()
            train_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)

            # -- Validate --
            model.eval()
            val_loss = 0.0
            all_true: List[int] = []
            all_pred: List[int] = []
            n_val_batches = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
                    n_val_batches += 1
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_true.extend(y_batch.cpu().numpy().tolist())
                    all_pred.extend(preds.tolist())

            val_loss /= max(n_val_batches, 1)
            y_t = np.array(all_true)
            y_p = np.array(all_pred)
            val_acc = float(np.mean(y_t == y_p)) if len(y_t) else 0.0

            # Cohen's kappa
            from ..evaluation import _cohen_kappa
            val_kappa = _cohen_kappa(y_t, y_p, list(range(5)))

            scheduler.step(val_kappa)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_kappa"].append(val_kappa)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f | val_kappa=%.3f | %.1fs",
                epoch, self.config.max_epochs, train_loss, val_loss, val_acc, val_kappa, elapsed,
            )

            if val_kappa > best_kappa:
                best_kappa = val_kappa
                patience_counter = 0
                torch.save(model.state_dict(), str(best_path))
                logger.info("  -> New best model saved (kappa=%.3f)", best_kappa)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.config.patience)
                    break

        # Save history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return {"best_kappa": best_kappa, "final_epoch": epoch, "model_path": str(best_path)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Train DL sleep-staging model")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)
    trainer = Trainer(config)
    result = trainer.run()
    logger.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
