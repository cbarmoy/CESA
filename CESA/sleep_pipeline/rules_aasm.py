"""Explicit AASM rule-based sleep-stage scorer.

Each rule maps directly to a criterion in the AASM Manual v2.6 (2023).
Decisions are **traceable**: every epoch carries a ``decision_reason``
string that explains *why* it was classified into a given stage.

The scorer works on the feature dict produced by ``features.py`` and
optionally uses context from neighbouring epochs for transition smoothing.

References
----------
AASM Manual for the Scoring of Sleep and Associated Events, v2.6 (2023).
Berry, R. B. et al.  Rules for Scoring Respiratory Events in Sleep.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from .contracts import Epoch, ScoringResult, StageLabel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable thresholds (sensible defaults from literature)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, float] = {
    # Relative power thresholds
    "alpha_wake_min": 0.15,         # AASM: alpha rhythm > 50% of epoch for W
    "delta_n3_min": 0.20,           # AASM: slow-wave activity >= 20% of epoch
    "sigma_n2_min": 0.04,           # sigma / total power -- spindle proxy
    "theta_n1_min": 0.25,           # theta dominance for N1
    # Cross-channel
    "eog_rem_activity_min": 50.0,   # absolute fallback, uV^2 (see below)
    "eog_rem_relative_min": 2.0,    # times the night's own quiet-eye baseline
    "eog_rem_count_min": 3.0,       # discrete REMs in a 30 s epoch (AASM: present)
    "emg_atonia_max": 5.0,          # EMG RMS below this ≈ atonia (REM)
    # General
    "emg_wake_min": 10.0,           # EMG RMS high => likely Wake
    "delta_beta_n3_min": 4.0,       # delta/beta ratio for deep sleep
}


# ---------------------------------------------------------------------------
# Core rule engine
# ---------------------------------------------------------------------------

def _eyes_moving(
    feats: Dict[str, float],
    thresholds: Dict[str, float],
) -> tuple[bool, str]:
    """Whether this epoch shows rapid eye movements, relative to the night.

    Two complementary cues, because AASM scores REM on discrete REMs and a
    30 s variance of the EOG derivative barely moves when those REMs are
    sparse. Either cue is enough. When no night-wide baseline could be
    established, the absolute fallbacks apply.
    """
    relative = feats.get("eog_rem_relative")
    count_rel = feats.get("eog_rem_count_relative")
    count = float(feats.get("eog_rem_count", 0.0))
    eog_rem = float(feats.get("eog_rem_activity", 0.0))
    rel_min = thresholds["eog_rem_relative_min"]
    count_min = thresholds["eog_rem_count_min"]
    has_baseline = relative is not None or count_rel is not None

    if relative is not None and relative >= rel_min:
        return True, f"rem_eog_x{relative:.1f}_baseline"
    if count_rel is not None and count_rel >= rel_min and count >= count_min:
        return True, f"rem_count_x{count_rel:.1f}_baseline"
    if not has_baseline:
        if eog_rem >= thresholds["eog_rem_activity_min"]:
            return True, f"rem_eog={eog_rem:.1f}"
        if count >= count_min:
            return True, f"rem_count={count:.0f}"
    return False, "yeux_calmes"


def _classify_epoch(
    feats: Dict[str, float],
    thresholds: Dict[str, float],
    prev_stage: Optional[StageLabel] = None,
) -> tuple[StageLabel, float, str]:
    """Classify a single epoch based on AASM-inspired rules.

    Returns (stage, confidence, reason).
    """
    if feats.get("is_artifact", 0.0) > 0.5:
        return StageLabel.U, 0.0, "artifact_rejection"

    rp_alpha = feats.get("relpow_alpha", 0.0)
    rp_delta = feats.get("relpow_delta", 0.0)
    rp_theta = feats.get("relpow_theta", 0.0)
    rp_sigma = feats.get("relpow_sigma", 0.0)
    emg_rms = feats.get("emg_rms", 0.0)
    delta_beta = feats.get("ratio_delta_beta", 0.0)

    # An absent channel reports 0.0, which must not be read as a measurement:
    # requiring a high EMG for Wake makes Wake unreachable on a montage without
    # a chin EMG, and requiring atonia is trivially satisfied there.
    has_emg = feats.get("has_emg", 1.0) > 0.5
    has_eog = feats.get("has_eog", 1.0) > 0.5
    atonia = emg_rms <= thresholds["emg_atonia_max"] if has_emg else True
    eyes_moving, eog_detail = _eyes_moving(feats, thresholds)

    # --- Wake with EMG (AASM: alpha + elevated chin tone) -----------------
    # Checked before REM: blinks and saccades of Wake must not be scored as
    # REM when the chin EMG is high.
    if has_emg:
        if rp_alpha >= thresholds["alpha_wake_min"] and emg_rms >= thresholds["emg_wake_min"]:
            return StageLabel.W, 0.85, f"alpha={rp_alpha:.2f}>=thr,emg={emg_rms:.1f}"

        if emg_rms >= thresholds["emg_wake_min"] * 1.5 and rp_alpha >= thresholds["alpha_wake_min"] * 0.6:
            return StageLabel.W, 0.70, f"high_emg={emg_rms:.1f},alpha={rp_alpha:.2f}"

    # --- N3 / Slow-wave sleep (AASM: >= 20% delta) -----------------------
    if rp_delta >= thresholds["delta_n3_min"] and delta_beta >= thresholds["delta_beta_n3_min"]:
        return StageLabel.N3, 0.80, f"delta={rp_delta:.2f}>=thr,d/b={delta_beta:.1f}"

    # --- REM (AASM: mixed-frequency EEG + REMs + atonia) ------------------
    # Must run before the no-EMG Wake rule. Without a chin EMG, alpha in a
    # mixed-frequency epoch is ambiguous; the EOG is the discriminator.
    if has_eog and eyes_moving and atonia:
        detail = f"atonia_emg={emg_rms:.1f}" if has_emg else "sans_emg"
        return StageLabel.R, 0.75 if has_emg else 0.60, f"{eog_detail},{detail}"

    # --- Wake without EMG: alpha, quiet eyes, little slow activity --------
    if (
        not has_emg
        and rp_alpha >= thresholds["alpha_wake_min"]
        and rp_delta < thresholds["delta_n3_min"]
    ):
        return StageLabel.W, 0.60, f"alpha={rp_alpha:.2f}>=thr,sans_emg"

    # --- N2 (AASM: sleep spindles / K-complexes on background theta) ------
    if rp_sigma >= thresholds["sigma_n2_min"] and rp_delta < thresholds["delta_n3_min"]:
        return StageLabel.N2, 0.70, f"sigma={rp_sigma:.3f}>=thr,delta={rp_delta:.2f}<N3"

    # --- N1 (AASM: theta activity, slow eye movements, alpha dropout) -----
    if rp_theta >= thresholds["theta_n1_min"] and rp_alpha < thresholds["alpha_wake_min"]:
        return StageLabel.N1, 0.60, f"theta={rp_theta:.2f}>=thr,alpha={rp_alpha:.2f}<wake"

    # --- Fallback with reduced confidence ---------------------------------
    # Use dominant band as a soft indicator
    bands = {"delta": rp_delta, "theta": rp_theta, "alpha": rp_alpha, "sigma": rp_sigma}
    dominant = max(bands, key=bands.get)  # type: ignore[arg-type]
    fallback_map = {"delta": StageLabel.N3, "theta": StageLabel.N1, "alpha": StageLabel.W, "sigma": StageLabel.N2}
    stage = fallback_map.get(dominant, StageLabel.N1)
    return stage, 0.40, f"fallback_dominant_{dominant}={bands[dominant]:.2f}"


# ---------------------------------------------------------------------------
# Night-relative eye-movement activity
# ---------------------------------------------------------------------------

#: Below this many epochs the median is too noisy to serve as a baseline.
_MIN_EPOCHS_FOR_BASELINE = 20


def _night_median(
    feature_list: Sequence[Dict[str, float]],
    key: str,
    *,
    require_positive: bool = True,
) -> Optional[float]:
    values = [
        float(f.get(key, 0.0))
        for f in feature_list
        if f.get("has_eog", 1.0) > 0.5
    ]
    usable = [
        v for v in values
        if np.isfinite(v) and (v > 0.0 if require_positive else True)
    ]
    if len(usable) < _MIN_EPOCHS_FOR_BASELINE:
        return None
    baseline = float(np.median(usable))
    if require_positive and baseline <= 0.0:
        return None
    return baseline


def eog_activity_baseline(feature_list: Sequence[Dict[str, float]]) -> Optional[float]:
    """Quiet-eye reference of a recording, or ``None`` if it cannot be measured.

    The variance of the EOG derivative scales with the amplifier range and
    shrinks as the sampling rate rises, so no fixed value transfers from one
    recording to the next. Most of a night is quiet NREM, so the median over the
    night is a usable "eyes still" reference that REM and wake epochs stand out
    from, whatever the scale.
    """
    return _night_median(feature_list, "eog_rem_activity", require_positive=True)


def add_relative_eog_activity(
    feature_list: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Add night-relative EOG descriptors used by the REM rule.

    ``eog_rem_relative`` is the derivative-variance in units of the night
    median. ``eog_rem_count_relative`` is the same for the discrete REM count.
    Left absent when no baseline can be established, in which case the rules
    fall back to the absolute thresholds.
    """
    activity_base = eog_activity_baseline(feature_list)
    count_base = _night_median(
        feature_list, "eog_rem_count", require_positive=False,
    )
    if count_base is not None and count_base <= 0.0:
        count_base = None
    if activity_base is None and count_base is None:
        return feature_list
    for feats in feature_list:
        if feats.get("has_eog", 1.0) <= 0.5:
            continue
        if activity_base is not None:
            feats["eog_rem_relative"] = (
                float(feats.get("eog_rem_activity", 0.0)) / activity_base
            )
        if count_base is not None:
            feats["eog_rem_count_relative"] = (
                float(feats.get("eog_rem_count", 0.0)) / count_base
            )
    return feature_list


# ---------------------------------------------------------------------------
# Transition smoothing
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS = {
    StageLabel.W: {StageLabel.W, StageLabel.N1, StageLabel.R},
    StageLabel.N1: {StageLabel.W, StageLabel.N1, StageLabel.N2, StageLabel.R},
    StageLabel.N2: {StageLabel.W, StageLabel.N1, StageLabel.N2, StageLabel.N3, StageLabel.R},
    StageLabel.N3: {StageLabel.N2, StageLabel.N3, StageLabel.W},
    StageLabel.R: {StageLabel.W, StageLabel.N1, StageLabel.N2, StageLabel.R},
}


def smooth_stages(
    stages: List[StageLabel],
    confidences: List[float],
) -> List[StageLabel]:
    """Apply AASM-compatible transition rules to remove impossible jumps.

    For instance W -> N3 is not allowed; the epoch is relabelled to N1 or N2
    depending on context.
    """
    if len(stages) < 2:
        return list(stages)

    smoothed = list(stages)
    for i in range(1, len(smoothed)):
        prev = smoothed[i - 1]
        cur = smoothed[i]
        if cur in _ALLOWED_TRANSITIONS.get(prev, set()):
            continue
        # Disallowed transition -- pick closest allowed stage by confidence
        allowed = _ALLOWED_TRANSITIONS.get(prev, {StageLabel.W})
        # Prefer N2 as a safe intermediate (most common NREM stage)
        if StageLabel.N2 in allowed:
            smoothed[i] = StageLabel.N2
        elif StageLabel.N1 in allowed:
            smoothed[i] = StageLabel.N1
        else:
            smoothed[i] = StageLabel.W
        logger.debug(
            "Transition smoothing epoch %d: %s->%s corrected to %s",
            i, prev.value, cur.value, smoothed[i].value,
        )
    return smoothed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_rule_based(
    feature_list: List[Dict[str, float]],
    *,
    epoch_duration_s: float = 30.0,
    thresholds: Optional[Dict[str, float]] = None,
    apply_smoothing: bool = True,
) -> ScoringResult:
    """Score all epochs using explicit AASM rules.

    Parameters
    ----------
    feature_list : list of dicts
        Output of ``features.extract_all_features``.
    epoch_duration_s : float
        Duration of each epoch in seconds (should be 30).
    thresholds : dict, optional
        Override default thresholds.
    apply_smoothing : bool
        Whether to enforce transition-rule smoothing.

    Returns
    -------
    ScoringResult
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    add_relative_eog_activity(feature_list)

    epochs: List[Epoch] = []
    prev_stage: Optional[StageLabel] = None

    for idx, feats in enumerate(feature_list):
        stage, conf, reason = _classify_epoch(feats, thr, prev_stage)
        epochs.append(Epoch(
            index=idx,
            start_s=idx * epoch_duration_s,
            duration_s=epoch_duration_s,
            features=feats,
            stage=stage,
            confidence=conf,
            decision_reason=reason,
        ))
        prev_stage = stage

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
        backend="aasm_rules",
    )
