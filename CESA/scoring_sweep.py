"""Compare several automatic-scoring configurations on the same recording.

Picking a derivation, an EOG or an EMG by trial and error through the dialog is
slow and leaves no trace. This module runs a list of configurations one after
the other against the *same* manual reference and reports the agreement of
each, so the choice rests on measurements.

The controller state is snapshotted and restored: a sweep never silently
replaces the user's scoring or settings. Applying a configuration is a separate,
explicit step.

Qt-free, so the engine is testable without a display.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from CESA.scoring_channels import channel_profile, profile_candidates

logger = logging.getLogger(__name__)

#: Rough per-run cost, refined from the previous run's measured duration.
DEFAULT_RUN_SECONDS = 40.0

#: Backends that read the subject's age and sex. Sweeping the metadata is
#: pointless for the others.
METADATA_BACKENDS: frozenset[str] = frozenset({"yasa"})


@dataclass(frozen=True)
class SweepConfig:
    """One candidate configuration for the automatic scoring."""

    eeg: str
    eog: Optional[str] = None
    emg: Optional[str] = None
    age: Optional[int] = None
    male: Optional[bool] = None
    method: str = "yasa"

    @property
    def label(self) -> str:
        parts = [self.method, f"EEG {self.eeg}"]
        parts.append(f"EOG {self.eog}" if self.eog else "sans EOG")
        if channel_profile(self.method).uses("emg"):
            parts.append(f"EMG {self.emg}" if self.emg else "sans EMG")
        if self.age is not None or self.male is not None:
            sex = "" if self.male is None else ("H" if self.male else "F")
            parts.append(f"age {self.age if self.age is not None else '?'}{sex}")
        return ", ".join(parts)


@dataclass
class SweepResult:
    """Outcome of one configuration, or the reason it could not be scored."""

    config: SweepConfig
    accuracy: float = 0.0
    kappa: float = 0.0
    macro_f1: float = 0.0
    n_compared: int = 0
    deep_sleep_accuracy: float = 0.0
    stage_f1: Dict[str, float] = field(default_factory=dict)
    predicted_pct: Dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.n_compared > 0


def available_eeg(names: Sequence[str], method: str = "yasa") -> List[str]:
    return profile_candidates(method, "eeg", names)


def available_eog(names: Sequence[str], method: str = "yasa") -> List[str]:
    return profile_candidates(method, "eog", names)


def available_emg(names: Sequence[str], method: str = "yasa") -> List[str]:
    """Only channels that can legitimately serve as a staging EMG.

    Empty for a backend that ignores the EMG, so the sweep never spends runs on
    a parameter without effect.
    """
    return profile_candidates(method, "emg", names)


def build_default_configs(
    channel_names: Sequence[str],
    *,
    method: str = "yasa",
    age: Optional[int] = None,
    male: Optional[bool] = None,
    max_eeg: int = 4,
) -> List[SweepConfig]:
    """Candidate configurations for *channel_names*.

    Built as a coordinate search rather than a full product: every available
    EEG derivation with the best EOG, then the effect of dropping the EOG, then
    each valid EMG, then the effect of the metadata. That keeps the number of
    runs linear in the number of channels instead of exponential.
    """
    eegs = available_eeg(channel_names, method)[:max_eeg]
    if not eegs:
        return []
    eogs = available_eog(channel_names, method)
    emgs = available_emg(channel_names, method)
    best_eog = eogs[0] if eogs else None

    configs: List[SweepConfig] = []

    def add(config: SweepConfig) -> None:
        if config not in configs:
            configs.append(config)

    for eeg in eegs:
        add(SweepConfig(eeg=eeg, eog=best_eog, age=age, male=male, method=method))
    # Slow waves are maximal frontally; the first *max_eeg* slots are central
    # (and unreferenced fallbacks), so a frontal derivation would otherwise
    # never be measured even when it is the AASM derivation for N3.
    for extra in available_eeg(channel_names, method):
        if extra in eegs:
            continue
        if any(token in extra.upper() for token in ("F3", "F4", "FZ", "FP")):
            add(SweepConfig(
                eeg=extra, eog=best_eog, age=age, male=male, method=method,
            ))
            break
    reference = eegs[0]
    if best_eog is not None:
        add(SweepConfig(eeg=reference, eog=None, age=age, male=male, method=method))
    for eog in eogs[1:]:
        add(SweepConfig(eeg=reference, eog=eog, age=age, male=male, method=method))
    for emg in emgs:
        add(SweepConfig(
            eeg=reference, eog=best_eog, emg=emg, age=age, male=male, method=method,
        ))
    if (age is not None or male is not None) and method in METADATA_BACKENDS:
        # Quantify what the metadata actually buys on this recording.
        add(SweepConfig(eeg=reference, eog=best_eog, method=method))
    return configs


def build_backend_comparison(
    channel_names: Sequence[str],
    methods: Sequence[str],
    *,
    age: Optional[int] = None,
    male: Optional[bool] = None,
) -> List[SweepConfig]:
    """One configuration per backend, each with its own preferred channels.

    Answers "which algorithm suits this recording" rather than "which channel
    suits this algorithm".
    """
    configs: List[SweepConfig] = []
    for method in methods:
        eegs = available_eeg(channel_names, method)
        if not eegs:
            continue
        eogs = available_eog(channel_names, method)
        emgs = available_emg(channel_names, method)
        configs.append(SweepConfig(
            eeg=eegs[0],
            eog=eogs[0] if eogs else None,
            emg=emgs[0] if emgs else None,
            age=age if method in METADATA_BACKENDS else None,
            male=male if method in METADATA_BACKENDS else None,
            method=method,
        ))
    return configs


def estimate_duration_s(n_configs: int, per_run_s: float = DEFAULT_RUN_SECONDS) -> float:
    return max(0.0, float(n_configs)) * float(per_run_s)


def _snapshot(ctrl: Any) -> Dict[str, Any]:
    keys = (
        "yasa_eeg_candidates", "yasa_eog_candidates", "yasa_emg_candidates",
        "yasa_age", "yasa_male", "n3_aasm_correction", "sleep_scoring_method",
        "auto_scoring_data", "auto_scoring_method", "auto_scoring_epoch_seconds",
        "auto_scoring_params", "sleep_scoring_data", "active_scoring_source",
        "scoring_epoch_duration", "scoring_dirty", "last_auto_scoring_error",
    )
    return {key: getattr(ctrl, key, None) for key in keys}


def _restore(ctrl: Any, state: Dict[str, Any]) -> None:
    for key, value in state.items():
        try:
            setattr(ctrl, key, value)
        except Exception:
            logger.debug("Could not restore %s during sweep", key)


def _apply(ctrl: Any, config: SweepConfig) -> None:
    ctrl.yasa_eeg_candidates = [config.eeg]
    ctrl.yasa_eog_candidates = [config.eog] if config.eog else []
    ctrl.yasa_emg_candidates = [config.emg] if config.emg else []
    ctrl.yasa_age = config.age
    ctrl.yasa_male = config.male
    ctrl.sleep_scoring_method = config.method


def apply_config(ctrl: Any, config: SweepConfig) -> None:
    """Make *config* the controller's active automatic-scoring configuration."""
    _apply(ctrl, config)
    # Unlike a sweep run, this is meant to stick: remember it as this backend's
    # customisation so the configuration dialog reopens on it.
    setter = getattr(ctrl, "set_channel_candidates_for", None)
    if callable(setter):
        setter(
            config.method,
            eeg=[config.eeg],
            eog=[config.eog] if config.eog else [],
            emg=[config.emg] if config.emg else [],
        )


def _measure(ctrl: Any, config: SweepConfig, duration_s: float) -> SweepResult:
    from CESA.scoring_comparison import present_stages, stage_metrics
    from CESA.scoring_report import agreement_summary, merged_deep_sleep_accuracy

    merged = ctrl.compare_scoring()
    if merged is None or merged.empty:
        return SweepResult(
            config=config, duration_s=duration_s,
            error="Aucune epoque comparable avec le scoring manuel.",
        )
    summary = agreement_summary(merged)
    stages = present_stages(merged)
    metrics = stage_metrics(merged, stages)
    n_total = len(merged) or 1
    return SweepResult(
        config=config,
        accuracy=summary["accuracy"],
        kappa=summary["kappa"],
        macro_f1=summary["macro_f1"],
        n_compared=summary["n_total"],
        deep_sleep_accuracy=merged_deep_sleep_accuracy(merged),
        stage_f1={stage: metrics[stage]["f1"] for stage in stages},
        predicted_pct={
            stage: float((merged["auto"] == stage).sum()) / n_total * 100.0
            for stage in stages
        },
        duration_s=duration_s,
    )


def _blocked_backends(ctrl: Any, methods: Iterable[str]) -> Dict[str, str]:
    """Backends that cannot run here, mapped to the reason why."""
    preflight = getattr(ctrl, "auto_scoring_preflight", None)
    if not callable(preflight):
        return {}
    blocked: Dict[str, str] = {}
    for method in methods:
        try:
            problem = preflight(method)
        except Exception as exc:
            problem = f"Verification impossible : {exc}"
        if problem:
            blocked[method] = str(problem)
    return blocked


def usable_backends(ctrl: Any, methods: Iterable[str]) -> Dict[str, Optional[str]]:
    """For each method, ``None`` if it can run, otherwise the blocking reason."""
    blocked = _blocked_backends(ctrl, methods)
    return {method: blocked.get(method) for method in methods}


def run_sweep(
    ctrl: Any,
    configs: Sequence[SweepConfig],
    *,
    progress_cb: Optional[Callable[[int, int, SweepConfig], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> List[SweepResult]:
    """Score *configs* one by one and measure each against the manual scoring.

    The controller is left exactly as it was found, whatever happens. Returns
    one result per configuration attempted, in the order given.
    """
    if ctrl is None or getattr(ctrl, "raw", None) is None:
        raise ValueError("Aucun enregistrement charge.")
    manual = getattr(ctrl, "manual_scoring_data", None)
    if manual is None or manual.empty:
        raise ValueError(
            "Un scoring manuel de reference est necessaire pour comparer les "
            "configurations."
        )

    state = _snapshot(ctrl)
    results: List[SweepResult] = []
    blocked = _blocked_backends(ctrl, {c.method for c in configs})
    try:
        for index, config in enumerate(configs):
            if should_cancel is not None and should_cancel():
                break
            if progress_cb is not None:
                progress_cb(index, len(configs), config)
            # A backend whose prerequisites are missing fails identically for
            # every configuration: report it once instead of running it.
            if config.method in blocked:
                results.append(SweepResult(
                    config=config, error=blocked[config.method],
                ))
                continue
            started = time.time()
            _apply(ctrl, config)
            try:
                ok = ctrl.run_auto_scoring(method=config.method)
            except Exception as exc:
                logger.debug("Sweep run failed for %s: %s", config.label, exc)
                ok = False
            elapsed = time.time() - started
            if not ok:
                results.append(SweepResult(
                    config=config, duration_s=elapsed,
                    error=str(getattr(ctrl, "last_auto_scoring_error", None)
                              or "Le scoring a echoue."),
                ))
                continue
            try:
                results.append(_measure(ctrl, config, elapsed))
            except Exception as exc:
                logger.debug("Sweep measurement failed for %s: %s", config.label, exc)
                results.append(SweepResult(
                    config=config, duration_s=elapsed,
                    error=f"Comparaison impossible : {exc}",
                ))
    finally:
        _restore(ctrl, state)
    return results


def rank_results(
    results: Sequence[SweepResult], key: str = "kappa",
) -> List[SweepResult]:
    """Successful results, best first.

    Ranked on Cohen's kappa by default: raw agreement flatters a configuration
    that simply predicts the majority stage.
    """
    usable = [r for r in results if r.ok]
    return sorted(usable, key=lambda r: getattr(r, key, 0.0), reverse=True)


def best_result(
    results: Sequence[SweepResult], key: str = "kappa",
) -> Optional[SweepResult]:
    ranked = rank_results(results, key)
    return ranked[0] if ranked else None


def sweep_summary(results: Sequence[SweepResult], key: str = "kappa") -> List[str]:
    """Plain-language reading of a sweep, ordered by usefulness."""
    ranked = rank_results(results, key)
    failed = [r for r in results if not r.ok]
    notes: List[str] = []
    if not ranked:
        reasons = "; ".join(sorted({str(r.error) for r in failed})) if failed else ""
        notes.append(
            "Aucune configuration n'a pu etre evaluee."
            + (f" Cause : {reasons}" if reasons else "")
        )
        return notes

    best = ranked[0]
    notes.append(
        f"Meilleure configuration : {best.config.label} "
        f"(concordance {best.accuracy:.1f}%, kappa {best.kappa:.3f})."
    )
    if len(ranked) > 1:
        worst = ranked[-1]
        spread = best.kappa - worst.kappa
        notes.append(
            f"Ecart entre la meilleure et la moins bonne des {len(ranked)} "
            f"configurations testees : {spread:.3f} de kappa et "
            f"{best.accuracy - worst.accuracy:.1f} points de concordance."
        )
        if spread < 0.02:
            notes.append(
                "Les configurations sont pratiquement equivalentes : le choix "
                "des canaux n'est pas le facteur limitant ici."
            )

    n3_f1 = best.stage_f1.get("N3")
    if n3_f1 is not None and n3_f1 < 0.3:
        best_n3 = max(ranked, key=lambda r: r.stage_f1.get("N3", 0.0))
        if best_n3 is not best and best_n3.stage_f1.get("N3", 0.0) > (n3_f1 + 0.1):
            notes.append(
                f"Le N3 reste mal detecte par la meilleure configuration "
                f"(F1 {n3_f1 * 100:.0f}%). La configuration "
                f"'{best_n3.config.label}' fait mieux sur ce seul stade "
                f"(F1 {best_n3.stage_f1['N3'] * 100:.0f}%) : a preferer si le "
                "sommeil lent profond est l'objet de l'etude."
            )
    if failed:
        notes.append(
            f"{len(failed)} configuration(s) n'ont pas abouti : "
            + "; ".join(sorted({str(r.error) for r in failed}))
        )
    return notes
