"""Machine-learning sleep scorer (interpretable).

This module provides a **supervised classification pipeline** for sleep
staging using interpretable features computed by ``features.py``.

Default model: ``HistGradientBoostingClassifier`` (scikit-learn), which
is fast, handles missing values natively, and supports probability
calibration.  A ``RandomForestClassifier`` fallback is also available.

Post-processing: optional HMM-style transition smoothing to enforce
physiologically plausible stage sequences.

Explainability: permutation importance is computed after prediction so
that clinicians can audit which features drove the decision.

Training workflow
-----------------
1. Collect labelled epochs (manual scoring aligned with EEG).
2. Run ``features.extract_all_features`` on each recording.
3. Call ``train_model`` with the concatenated feature matrix + labels.
4. Save the model with ``save_model``.
5. Use ``score_ml`` at inference time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import CLINICAL_STAGE_STRINGS, Epoch, ScoringResult, StageLabel
from .rules_aasm import smooth_stages

logger = logging.getLogger(__name__)

# Numeric encoding consistent with contracts.ScoringResult.stage_array
_STAGE_TO_INT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
_INT_TO_STAGE = {v: k for k, v in _STAGE_TO_INT.items()}


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def _features_to_matrix(
    feature_list: List[Dict[str, float]],
) -> Tuple[np.ndarray, List[str]]:
    """Convert list-of-dicts to a (n_epochs, n_features) array.

    Returns (X, feature_names).
    """
    if not feature_list:
        return np.empty((0, 0)), []
    names = sorted(feature_list[0].keys())
    X = np.array([[f.get(n, 0.0) for n in names] for f in feature_list], dtype=np.float64)
    return X, names


# ---------------------------------------------------------------------------
# Temporal context augmentation
# ---------------------------------------------------------------------------

_CONTEXT_OFFSETS = (-2, -1, 1, 2)

# Subset of base features propagated from neighbours (keeps dimensionality manageable)
_CONTEXT_KEYS = [
    "relpow_delta", "relpow_theta", "relpow_alpha", "relpow_sigma", "relpow_beta",
    "spectral_entropy", "spindle_count", "eog_rem_activity", "emg_rms",
]


def add_temporal_context(
    feature_list: List[Dict[str, float]],
    *,
    context_offsets: Sequence[int] = _CONTEXT_OFFSETS,
    context_keys: Sequence[str] = _CONTEXT_KEYS,
) -> List[Dict[str, float]]:
    """Augment each epoch's feature dict with information from neighbours.

    For each offset in *context_offsets* (e.g. -2, -1, +1, +2) and each key
    in *context_keys*, a new feature ``{key}_t{offset:+d}`` is appended.
    At boundaries, zero-padding is used.

    Additional derived temporal features are added:
    - ``night_fraction``: position 0..1 in the recording.
    - ``delta_power_trend``: slope of delta power over the last 5 epochs.
    """
    T = len(feature_list)
    if T == 0:
        return feature_list

    augmented: List[Dict[str, float]] = []
    for t in range(T):
        f = dict(feature_list[t])

        # Neighbour features
        for offset in context_offsets:
            idx = t + offset
            for key in context_keys:
                name = f"{key}_t{offset:+d}"
                if 0 <= idx < T:
                    f[name] = feature_list[idx].get(key, 0.0)
                else:
                    f[name] = 0.0

        # Night fraction (0 = start, 1 = end)
        f["night_fraction"] = t / max(T - 1, 1)

        # Delta power trend: linear slope over last 5 epochs
        window = min(5, t + 1)
        deltas = [feature_list[t - j].get("relpow_delta", 0.0) for j in range(window)]
        deltas.reverse()
        if len(deltas) >= 2:
            x = np.arange(len(deltas), dtype=float)
            f["delta_power_trend"] = float(np.polyfit(x, deltas, 1)[0])
        else:
            f["delta_power_trend"] = 0.0

        augmented.append(f)

    return augmented


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    feature_list: List[Dict[str, float]],
    labels: Sequence[str],
    *,
    model_type: str = "hgb",
    random_state: int = 42,
    use_temporal_context: bool = True,
) -> Any:
    """Train a sleep-staging model.

    Parameters
    ----------
    feature_list : list of dicts
        Output of ``features.extract_all_features`` for *all* training epochs.
    labels : sequence of str
        Stage labels aligned with *feature_list* (W/N1/N2/N3/R).
    model_type : str
        ``"hgb"`` for HistGradientBoosting, ``"rf"`` for RandomForest.
    random_state : int
        Reproducibility seed.
    use_temporal_context : bool
        If True, augment features with neighbour-epoch context.

    Returns
    -------
    Trained sklearn estimator (with ``.predict`` and ``.predict_proba``).
    """
    from sklearn.ensemble import (  # type: ignore
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore

    if use_temporal_context:
        feature_list = add_temporal_context(feature_list)

    X, names = _features_to_matrix(feature_list)
    y = np.array([_STAGE_TO_INT.get(StageLabel.from_string(s).value, -1) for s in labels])

    valid = y >= 0
    X = X[valid]
    y = y[valid]
    if len(X) == 0:
        raise ValueError("No valid labelled epochs for training.")

    logger.info("Training %s on %d epochs, %d features", model_type, len(X), X.shape[1])

    if model_type == "rf":
        base = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        base = HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=random_state,
            class_weight="balanced",
        )

    # Probability calibration with cross-validation
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X, y)

    # Attach feature names for later explainability
    model._sleep_feature_names = names  # type: ignore[attr-defined]
    return model


def save_model(model: Any, path: str) -> None:
    """Persist a trained model to disk (joblib)."""
    import joblib  # type: ignore
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: str) -> Any:
    """Load a previously saved model."""
    import joblib  # type: ignore
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def score_ml(
    feature_list: List[Dict[str, float]],
    *,
    epoch_duration_s: float = 30.0,
    model_path: Optional[str] = None,
    model: Optional[Any] = None,
    apply_smoothing: bool = True,
    use_temporal_context: bool = True,
) -> ScoringResult:
    """Score epochs using a trained ML model.

    Parameters
    ----------
    feature_list : list of dicts
        Feature dicts for each epoch.
    epoch_duration_s : float
        Epoch length in seconds.
    model_path : str, optional
        Path to a saved model file.
    model : sklearn estimator, optional
        Pre-loaded model (takes precedence over *model_path*).
    apply_smoothing : bool
        Apply AASM transition smoothing after prediction.
    use_temporal_context : bool
        Augment features with neighbour-epoch context before prediction.

    Returns
    -------
    ScoringResult
    """
    if model is None:
        if model_path is None:
            raise RuntimeError(
                "ML scoring requires a trained model. "
                "Provide model_path or train a model first."
            )
        model = load_model(model_path)

    if use_temporal_context:
        feature_list = add_temporal_context(feature_list)

    X, _ = _features_to_matrix(feature_list)
    if X.shape[0] == 0:
        return ScoringResult(epoch_duration_s=epoch_duration_s, backend="ml")

    y_pred = model.predict(X)
    probas = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X)
        except Exception:
            pass

    epochs: List[Epoch] = []
    for i, (pred, feats) in enumerate(zip(y_pred, feature_list)):
        stage_str = _INT_TO_STAGE.get(int(pred), "U")
        stage = StageLabel.from_string(stage_str)
        conf = 0.0
        if probas is not None:
            conf = float(np.max(probas[i]))
        epochs.append(Epoch(
            index=i,
            start_s=i * epoch_duration_s,
            duration_s=epoch_duration_s,
            features=feats,
            stage=stage,
            confidence=conf,
            decision_reason=f"ml_pred={stage_str}",
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
        backend="ml",
    )


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def feature_importance(
    model: Any,
    feature_list: List[Dict[str, float]],
    labels: Sequence[str],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute permutation importance for each feature.

    Parameters
    ----------
    model : trained sklearn estimator.
    feature_list : feature dicts (validation set).
    labels : true stage labels.
    n_repeats : number of permutation rounds.

    Returns
    -------
    Dict mapping feature name to mean importance (accuracy drop).
    """
    from sklearn.inspection import permutation_importance  # type: ignore

    X, names = _features_to_matrix(feature_list)
    y = np.array([_STAGE_TO_INT.get(StageLabel.from_string(s).value, -1) for s in labels])
    valid = y >= 0
    X, y = X[valid], y[valid]
    if len(X) == 0:
        return {}

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )
    importance: Dict[str, float] = {}
    for i, name in enumerate(names):
        importance[name] = float(result.importances_mean[i])
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
