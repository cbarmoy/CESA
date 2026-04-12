"""Explainability module for sleep scoring decisions.

Provides per-epoch and global explanations combining ML feature importance
with AASM rule coverage, enabling clinicians to audit every scoring decision.

Key functions
-------------
* ``explain_epoch``          -- per-epoch SHAP values + rule contribution.
* ``global_feature_importance`` -- SHAP mean |values| across all epochs.
* ``rule_coverage_report``   -- how often each AASM rule was the primary driver.

Dependencies
------------
* ``shap`` (optional): used for TreeExplainer on gradient-boosted models.
  Falls back to permutation importance if unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .contracts import ScoringResult, StageLabel
from .ml_scorer import _features_to_matrix, _STAGE_TO_INT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-epoch explanation
# ---------------------------------------------------------------------------

@dataclass
class EpochExplanation:
    """Explanation for a single epoch's scoring decision."""

    epoch_index: int
    predicted_stage: str
    confidence: float
    ml_shap_values: Dict[str, float] = field(default_factory=dict)
    top_features: List[str] = field(default_factory=list)
    rule_fired: str = ""
    rule_margin: float = 0.0
    agreement_ml_rules: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_index": self.epoch_index,
            "predicted_stage": self.predicted_stage,
            "confidence": round(self.confidence, 4),
            "top_features": self.top_features,
            "rule_fired": self.rule_fired,
            "rule_margin": round(self.rule_margin, 4),
            "agreement_ml_rules": self.agreement_ml_rules,
        }


def explain_epoch(
    epoch_features: Dict[str, float],
    *,
    model: Optional[Any] = None,
    explainer: Optional[Any] = None,
    rules_result_epoch: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
) -> EpochExplanation:
    """Explain a single epoch's classification.

    Parameters
    ----------
    epoch_features : dict
        Feature dict for the epoch.
    model : sklearn estimator, optional
        Trained model for SHAP/permutation analysis.
    explainer : shap.TreeExplainer, optional
        Pre-built SHAP explainer (reuse across epochs for efficiency).
    rules_result_epoch : Epoch, optional
        Epoch from rule-based scoring for comparison.
    feature_names : list of str, optional
        Ordered feature names matching the model's training.

    Returns
    -------
    EpochExplanation
    """
    explanation = EpochExplanation(
        epoch_index=0,
        predicted_stage="U",
        confidence=0.0,
    )

    if rules_result_epoch is not None:
        explanation.rule_fired = rules_result_epoch.decision_reason
        explanation.confidence = rules_result_epoch.confidence
        explanation.predicted_stage = rules_result_epoch.stage.value

    # SHAP values
    if model is not None:
        shap_vals = _compute_shap_single(epoch_features, model, explainer, feature_names)
        explanation.ml_shap_values = shap_vals
        # Top 5 features by absolute SHAP
        sorted_feats = sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)
        explanation.top_features = [name for name, _ in sorted_feats[:5]]

        # Check agreement with rules
        if rules_result_epoch is not None:
            if hasattr(model, "predict"):
                if feature_names is None:
                    feature_names = sorted(epoch_features.keys())
                x = np.array([[epoch_features.get(n, 0.0) for n in feature_names]])
                ml_pred = model.predict(x)[0]
                ml_stage = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}.get(int(ml_pred), "U")
                explanation.predicted_stage = ml_stage
                explanation.agreement_ml_rules = (ml_stage == rules_result_epoch.stage.value)

    return explanation


def _compute_shap_single(
    features: Dict[str, float],
    model: Any,
    explainer: Optional[Any],
    feature_names: Optional[List[str]],
) -> Dict[str, float]:
    """Compute SHAP values for one epoch. Falls back to zeros if shap unavailable."""
    if feature_names is None:
        feature_names = sorted(features.keys())

    x = np.array([[features.get(n, 0.0) for n in feature_names]])

    # Try SHAP
    if explainer is not None:
        try:
            sv = explainer.shap_values(x)
            # sv may be list (per class) or array
            if isinstance(sv, list):
                # Use the predicted class's SHAP values
                pred = int(model.predict(x)[0])
                vals = sv[pred][0] if pred < len(sv) else sv[0][0]
            else:
                vals = sv[0]
            return {name: float(vals[i]) for i, name in enumerate(feature_names)}
        except Exception as e:
            logger.debug("SHAP failed for epoch: %s", e)

    # Try creating a TreeExplainer on-the-fly
    try:
        import shap  # type: ignore
        # CalibratedClassifierCV wraps base estimators
        base_model = model
        if hasattr(model, "estimator"):
            base_model = model.estimator
        elif hasattr(model, "calibrated_classifiers_"):
            base_model = model.calibrated_classifiers_[0].estimator
        exp = shap.TreeExplainer(base_model)
        sv = exp.shap_values(x)
        if isinstance(sv, list):
            pred = int(model.predict(x)[0])
            vals = sv[pred][0] if pred < len(sv) else sv[0][0]
        else:
            vals = sv[0]
        return {name: float(vals[i]) for i, name in enumerate(feature_names)}
    except Exception:
        pass

    # Fallback: return zeros
    return {name: 0.0 for name in feature_names}


# ---------------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------------

def global_feature_importance(
    model: Any,
    feature_list: List[Dict[str, float]],
    labels: Sequence[str],
    *,
    method: str = "auto",
    n_repeats: int = 10,
) -> Dict[str, float]:
    """Compute global feature importance.

    Parameters
    ----------
    model : trained estimator.
    feature_list : feature dicts.
    labels : true labels.
    method : ``"shap"`` for SHAP mean |values|, ``"permutation"`` for
        permutation importance, ``"auto"`` tries SHAP first.

    Returns
    -------
    Dict[feature_name, importance] sorted descending.
    """
    X, names = _features_to_matrix(feature_list)
    y = np.array([_STAGE_TO_INT.get(StageLabel.from_string(s).value, -1) for s in labels])
    valid = y >= 0
    X, y = X[valid], y[valid]
    if len(X) == 0:
        return {}

    if method in ("shap", "auto"):
        result = _try_shap_global(model, X, names)
        if result is not None:
            return result
        if method == "shap":
            logger.warning("SHAP failed, returning empty importance.")
            return {}

    # Permutation importance fallback
    return _permutation_global(model, X, y, names, n_repeats)


def _try_shap_global(model: Any, X: np.ndarray, names: List[str]) -> Optional[Dict[str, float]]:
    """Try SHAP TreeExplainer for global importance."""
    try:
        import shap  # type: ignore
        base_model = model
        if hasattr(model, "estimator"):
            base_model = model.estimator
        elif hasattr(model, "calibrated_classifiers_"):
            base_model = model.calibrated_classifiers_[0].estimator

        # Use a subsample for speed
        n_samples = min(500, len(X))
        idx = np.random.default_rng(42).choice(len(X), n_samples, replace=False)
        X_sub = X[idx]

        explainer = shap.TreeExplainer(base_model)
        sv = explainer.shap_values(X_sub)

        if isinstance(sv, list):
            # Average absolute SHAP across all classes
            abs_shap = np.mean([np.abs(s) for s in sv], axis=0)
        else:
            abs_shap = np.abs(sv)

        mean_importance = abs_shap.mean(axis=0)
        result = {names[i]: float(mean_importance[i]) for i in range(len(names))}
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))
    except Exception as e:
        logger.debug("SHAP global importance failed: %s", e)
        return None


def _permutation_global(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    n_repeats: int,
) -> Dict[str, float]:
    """Permutation importance fallback."""
    from sklearn.inspection import permutation_importance  # type: ignore
    result = permutation_importance(model, X, y, n_repeats=n_repeats, n_jobs=-1, random_state=42)
    importance = {names[i]: float(result.importances_mean[i]) for i in range(len(names))}
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Rule coverage report
# ---------------------------------------------------------------------------

def rule_coverage_report(result: ScoringResult) -> Dict[str, int]:
    """Count how often each AASM rule was the primary classification driver.

    Parses the ``decision_reason`` field of each epoch.

    Returns
    -------
    Dict mapping rule name to count, sorted descending.
    """
    counts: Dict[str, int] = {}
    for ep in result.epochs:
        reason = ep.decision_reason
        if not reason:
            continue
        # Take the primary rule (before any |smoothed_from or |hmm_from suffix)
        primary = reason.split("|")[0].strip()
        if primary:
            counts[primary] = counts.get(primary, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
