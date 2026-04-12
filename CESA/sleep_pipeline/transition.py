"""Bridge between legacy CESA scoring and the new modular pipeline.

This module allows the existing ``SleepScorer`` facade and the UI in
``eeg_studio_fixed.py`` to call the new pipeline **without** breaking any
existing functionality.  The user can switch backends via a simple config
flag (``scoring_backend``).

Usage from ``eeg_studio_fixed``::

    from CESA.sleep_pipeline.transition import run_pipeline

    result = run_pipeline(raw, backend="aasm_rules")
    df = result.to_dataframe()  # legacy-compatible (time, stage, confidence)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import mne
import pandas as pd

from .contracts import ScoringResult, StageLabel
from .preprocessing import PreprocessingConfig, preprocess
from .features import extract_all_features
from .rules_aasm import score_rule_based

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    raw: mne.io.BaseRaw,
    *,
    backend: str = "aasm_rules",
    eeg_candidates: Sequence[str] = (),
    eog_candidates: Sequence[str] = (),
    emg_candidates: Sequence[str] = (),
    epoch_duration_s: float = 30.0,
    target_sfreq: float = 100.0,
    thresholds: Optional[Dict[str, float]] = None,
    ml_model_path: Optional[str] = None,
) -> ScoringResult:
    """Run the full sleep-scoring pipeline.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object.
    backend : str
        One of ``"aasm_rules"``, ``"ml"``, ``"ml_hmm"``, ``"rules_hmm"``,
        ``"legacy"``.
    eeg_candidates, eog_candidates, emg_candidates : sequences
        Override default channel candidate lists.
    epoch_duration_s : float
        Epoch length (should be 30 for AASM compliance).
    target_sfreq : float
        Resampling target.
    thresholds : dict, optional
        Override AASM rule thresholds.
    ml_model_path : str, optional
        Path to a trained ML model (joblib/pickle) for ``"ml"`` backend.

    Returns
    -------
    ScoringResult
    """
    if backend == "legacy":
        raise ValueError(
            "Use the original SleepScorer class directly for legacy backends "
            "(yasa, usleep, pftsleep). The transition module only wraps the "
            "new pipeline backends."
        )

    config = PreprocessingConfig(
        target_sfreq=target_sfreq,
        epoch_duration_s=epoch_duration_s,
        eeg_candidates=eeg_candidates,
        eog_candidates=eog_candidates,
        emg_candidates=emg_candidates,
    )
    epoched = preprocess(raw, config)
    feature_list = extract_all_features(epoched)

    if backend == "aasm_rules":
        return score_rule_based(
            feature_list,
            epoch_duration_s=epoch_duration_s,
            thresholds=thresholds,
        )

    if backend == "rules_hmm":
        result = score_rule_based(
            feature_list,
            epoch_duration_s=epoch_duration_s,
            thresholds=thresholds,
            apply_smoothing=False,
        )
        return _apply_hmm(result)

    if backend == "ml":
        return _score_ml(feature_list, epoch_duration_s, ml_model_path)

    if backend == "ml_hmm":
        result = _score_ml(feature_list, epoch_duration_s, ml_model_path, apply_smoothing=False)
        return _apply_hmm(result)

    raise ValueError(f"Unknown pipeline backend: {backend!r}")


def _score_ml(
    feature_list,
    epoch_duration_s: float,
    model_path: Optional[str],
    apply_smoothing: bool = True,
) -> ScoringResult:
    """Score using a trained ML model (RandomForest / HistGBM)."""
    from .ml_scorer import score_ml  # deferred to avoid import cost
    return score_ml(
        feature_list,
        epoch_duration_s=epoch_duration_s,
        model_path=model_path,
        apply_smoothing=apply_smoothing,
    )


def _apply_hmm(result: ScoringResult) -> ScoringResult:
    """Apply HMM Viterbi decoding to refine a scoring result."""
    from .sequence_model import hmm_decode_scoring  # deferred import
    return hmm_decode_scoring(result)


# ---------------------------------------------------------------------------
# Legacy DataFrame <-> ScoringResult adapters
# ---------------------------------------------------------------------------

def legacy_df_to_result(
    df: pd.DataFrame,
    *,
    epoch_duration_s: float = 30.0,
    backend: str = "legacy",
) -> ScoringResult:
    """Convert a legacy DataFrame(time, stage) to a ``ScoringResult``."""
    return ScoringResult.from_dataframe(df, epoch_duration_s=epoch_duration_s, backend=backend)


def result_to_legacy_df(result: ScoringResult) -> pd.DataFrame:
    """Convert a ``ScoringResult`` back to a legacy DataFrame."""
    return result.to_dataframe()
