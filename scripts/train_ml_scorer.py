#!/usr/bin/env python3
"""Train the interpretable ML sleep scorer.

Usage example
-------------
::

    python scripts/train_ml_scorer.py \\
        --edf data/subject01.edf \\
        --scoring data/subject01_scoring.csv \\
        --output models/sleep_ml.joblib

Multiple recordings can be concatenated with repeated ``--edf / --scoring``
pairs.  The script extracts features from each recording, aligns them with
the provided manual scoring, and trains a ``HistGradientBoostingClassifier``.

The resulting model can be used by CESA's ``SleepScorer(method="ml")`` or
by setting *Backend: ML* in the UI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CESA.sleep_pipeline.contracts import StageLabel
from CESA.sleep_pipeline.preprocessing import PreprocessingConfig, preprocess
from CESA.sleep_pipeline.features import extract_all_features
from CESA.sleep_pipeline.ml_scorer import train_model, save_model, feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_scoring(path: str, epoch_duration_s: float) -> pd.DataFrame:
    """Load scoring CSV/Excel and return DataFrame with (time, stage)."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif p.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported scoring file format: {p.suffix}")
    if "time" not in df.columns or "stage" not in df.columns:
        raise ValueError("Scoring file must contain 'time' and 'stage' columns.")
    df["stage"] = df["stage"].astype(str).str.strip().str.upper()
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the ML sleep scorer.")
    parser.add_argument("--edf", nargs="+", required=True, help="EDF recording file(s).")
    parser.add_argument("--scoring", nargs="+", required=True, help="Scoring file(s) (CSV/Excel).")
    parser.add_argument("--output", default="models/sleep_ml.joblib", help="Model output path.")
    parser.add_argument("--epoch-duration", type=float, default=30.0, help="Epoch duration (s).")
    parser.add_argument("--model-type", choices=["hgb", "rf"], default="hgb")
    parser.add_argument("--target-sfreq", type=float, default=100.0)
    args = parser.parse_args()

    if len(args.edf) != len(args.scoring):
        logger.error("Number of --edf and --scoring files must match.")
        return 1

    all_features: List[Dict[str, float]] = []
    all_labels: List[str] = []

    import mne

    for edf_path, score_path in zip(args.edf, args.scoring):
        logger.info("Processing %s + %s", edf_path, score_path)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        config = PreprocessingConfig(
            target_sfreq=args.target_sfreq,
            epoch_duration_s=args.epoch_duration,
        )
        epoched = preprocess(raw, config)
        features = extract_all_features(epoched)

        scoring = _load_scoring(score_path, args.epoch_duration)
        # Align: match each epoch index to the scoring by time
        for i, feats in enumerate(features):
            t = i * args.epoch_duration
            # Find closest scoring entry
            diffs = np.abs(scoring["time"].values - t)
            best = int(np.argmin(diffs))
            if diffs[best] < args.epoch_duration / 2:
                label = scoring["stage"].iloc[best]
                stage = StageLabel.from_string(label)
                if stage != StageLabel.U:
                    all_features.append(feats)
                    all_labels.append(stage.value)

    logger.info("Total training epochs: %d", len(all_features))
    if len(all_features) < 50:
        logger.error("Not enough labelled epochs to train (minimum 50).")
        return 1

    model = train_model(all_features, all_labels, model_type=args.model_type)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.output)

    # Feature importance on training set (for quick audit)
    imp = feature_importance(model, all_features, all_labels, n_repeats=5)
    logger.info("Top-10 features by importance:")
    for name, score in list(imp.items())[:10]:
        logger.info("  %s: %.4f", name, score)

    return 0


if __name__ == "__main__":
    sys.exit(main())
