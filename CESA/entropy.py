"""
CESA (Complex EEG Studio Analysis) v0.0beta1.0 - Entropy Analysis Module
=========================================================

Module d'analyse d'entropie renormée pour CESA v0.0beta1.0.
Développé pour l'Unité Neuropsychologie du Stress (IRBA).

Ce module implémente l'analyse d'entropie renormée basée sur les travaux
de Jean-Pierre Issartel (DOI: 10.1098/rspa.2007.1877 et 10.1007/s00024-011-0381-4).

Fonctionnalités principales:
- Calcul d'entropie renormée multi-canal
- Fenêtrage glissant configurable
- Moments généralisés d'ordre configurable
- Kernels de renormée (powerlaw, log, adaptive, identity)
- Support multi-échelle
- Interface compatible avec MNE-Python

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 0.0beta1.0
Date: 2025-01-27
Licence: MIT
"""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# Configuration dataclasses
# =============================================================================

PsiCallable = Callable[[np.ndarray], np.ndarray]


@dataclass
class RenormalizedEntropyConfig:
    """Configuration container for the renormalised entropy workflow.

    Parameters
    ----------
    window_length : float
        Sliding window length expressed in seconds.
    overlap : float
        Fractional overlap between successive windows (0.0 -> no overlap,
        0.5 -> 50 % overlap, etc.). Must satisfy ``0.0 <= overlap < 1.0``.
    moment_order : float
        Order of the generalised moment computed on each channel window. By
        default we use the second-order moment (root-mean-square amplitude),
        which is a robust energy proxy commonly used in Issartel's scaling
        analyses.
    central_moment : bool
        If True, subtract the local mean before computing the moment (central
        moments). If False, absolute values are used directly.
    normalize_moment : bool
        When True, the computed moment is normalised by taking the
        ``1 / moment_order`` power (generalised mean). Disable this if the raw
        power of the moment is preferred.
    detrend : bool
        If True, subtract a linear trend inside each window prior to moment
        computation.
    psi_name : str
        Name of the spectral weighting function to apply to the covariance
        eigenvalues. Available options are: ``identity``, ``powerlaw``,
        ``log`` and ``adaptive``.
    psi_params : Dict[str, float]
        Additional parameters for the psi transform (see `_get_psi_callable`).
    regularization : float
        Ridge regularisation added to the covariance matrix prior to
        diagonalisation to stabilise the estimation.
    min_eigenvalue : float
        Floor applied to eigenvalues before psi weighting to avoid numerical
        issues.
    max_windows : Optional[int]
        Optional cap on the number of windows processed (useful for large
        recordings).
    return_intermediate : bool
        Set to False to discard intermediate arrays from the output
        `RenormalizationResult` (helps saving memory if unneeded).
    entropy_unit : str
        Either ``"nat"`` (natural logarithm), ``"bit"`` (base-2 logarithm) or
        ``"both"`` to compute both units.
    scales : Optional[Sequence[float]]
        Optional sequence of additional scale factors (in seconds) to apply
        for multi-scale renormalisation. For each scale, an auxiliary moment is
        computed and concatenated with the base window length.
    """

    window_length: float = 4.0
    overlap: float = 0.5
    moment_order: float = 2.0
    central_moment: bool = True
    normalize_moment: bool = True
    detrend: bool = False
    psi_name: str = "powerlaw"
    psi_params: Dict[str, float] = field(default_factory=lambda: {"gamma": 0.5, "epsilon": 1e-12})
    regularization: float = 1e-9
    min_eigenvalue: float = 1e-12
    max_windows: Optional[int] = None
    return_intermediate: bool = True
    entropy_unit: str = "both"
    scales: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("`overlap` must satisfy 0.0 <= overlap < 1.0")
        if self.window_length <= 0:
            raise ValueError("`window_length` must be strictly positive")
        if self.moment_order == 0:
            raise ValueError("`moment_order` cannot be zero")
        if self.entropy_unit not in {"nat", "bit", "both"}:
            raise ValueError("`entropy_unit` must be 'nat', 'bit' or 'both'")


@dataclass
class RenormalizationResult:
    """Container storing the output of renormalised entropy computations."""

    entropy_nats: float
    entropy_bits: float
    weighted_covariance: np.ndarray
    psi_eigenvalues: np.ndarray
    raw_covariance: np.ndarray
    mu_matrix: Optional[np.ndarray]
    mean_mu: Optional[np.ndarray]
    config: RenormalizedEntropyConfig
    channel_names: Optional[Tuple[str, ...]] = None
    window_samples: int = 0
    step_samples: int = 0
    scales: Optional[Tuple[float, ...]] = None

    def as_dict(self) -> Dict[str, object]:
        """Return a serialisable view of the main outputs."""

        return {
            "entropy_nats": self.entropy_nats,
            "entropy_bits": self.entropy_bits,
            "weighted_covariance": self.weighted_covariance.tolist(),
            "psi_eigenvalues": self.psi_eigenvalues.tolist(),
            "raw_covariance": self.raw_covariance.tolist(),
            "channel_names": self.channel_names,
            "window_samples": self.window_samples,
            "step_samples": self.step_samples,
            "scales": self.scales,
        }


@dataclass
class MultiscaleEntropyConfig:
    """Configuration container for multiscale entropy (MSE).

    Parameters
    ----------
    m : int
        Embedding dimension (pattern length). Classical MSE uses ``m = 2``.
    r : float
        Relative tolerance applied to the channel standard deviation.
        NOTE: When using EntropyHub with RadNew=1, r is applied relative to
        std(coarse_grained_signal) for each scale, not std(original_signal).
        This is the standard approach but differs from some implementations.
    scales : Sequence[int]
        Positive integers describing each coarse-graining scale.
        Each scale τ corresponds to coarse-graining with window size τ.
    max_samples : Optional[int]
        Optional cap on the number of samples to speed-up experiments.
        Truncates the signal before processing (affects variance).
    max_pattern_length : int
        Maximum number of samples used for SampEn calculation per scale.
        If the signal is longer, it is downsampled using linear interpolation.
        WARNING: Downsampling affects variance and thus MSE values.
    return_intermediate : bool
        Whether to keep the coarse-grained signals in the result payload.
        NOTE: Currently ignored when using EntropyHub (not supported).
    """

    m: int = 2
    r: float = 0.2
    scales: Sequence[int] = field(default_factory=lambda: tuple(range(1, 21)))
    max_samples: Optional[int] = 200_000
    max_pattern_length: int = 5000
    return_intermediate: bool = False

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError("`m` must be >= 1 for multiscale entropy")
        if self.r <= 0:
            raise ValueError("`r` must be strictly positive")
        if not self.scales:
            raise ValueError("`scales` must contain at least one element")

        processed_scales = sorted({int(scale) for scale in self.scales if int(scale) > 0})
        if not processed_scales:
            raise ValueError("All `scales` must be strictly positive integers")
        self.scales = tuple(processed_scales)

        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("`max_samples`, when provided, must be > 0")
        if self.max_pattern_length <= self.m + 1:
            raise ValueError("`max_pattern_length` must be greater than `m + 1`")


@dataclass
class MultiscaleEntropyResult:
    """Detailed container storing the outcome of multiscale entropy."""

    entropy_by_scale: Dict[int, float]
    channel_entropy: Dict[str, Dict[int, float]]
    sample_entropy_by_scale: Dict[int, Dict[str, object]]
    coarse_grained_signals: Optional[Dict[int, np.ndarray]]
    config: MultiscaleEntropyConfig
    channel_names: Optional[Tuple[str, ...]]
    sampling_frequency: float
    processed_samples: int

    def summary(self) -> Dict[str, object]:
        """Concise, serialisable snapshot that can be logged or exported."""

        return {
            "entropy_by_scale": self.entropy_by_scale,
            "channel_entropy": self.channel_entropy,
            "scales": list(self.config.scales),
            "sampling_frequency": self.sampling_frequency,
            "processed_samples": self.processed_samples,
        }


# =============================================================================
# Psi transforms (renormalisation kernels)
# =============================================================================

def _psi_identity(eigenvalues: np.ndarray, **_: float) -> np.ndarray:
    return eigenvalues


def _psi_powerlaw(eigenvalues: np.ndarray, gamma: float = 0.5, epsilon: float = 1e-12) -> np.ndarray:
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return np.power(eigenvalues, gamma)


def _psi_log(eigenvalues: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return np.log1p(eigenvalues / epsilon)


def _psi_adaptive(eigenvalues: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Adaptive psi that preserves ordering while damping extremes.

    Borrowed from renormalisation heuristics: low eigenvalues receive a boost
    while large ones are compressed using a sigmoidal scaling. Works well for
    heterogeneous covariance spectra that are typical in EEG channel moments.
    """

    eigenvalues = np.maximum(eigenvalues, epsilon)
    median = np.median(eigenvalues)
    if median <= 0:
        median = np.mean(eigenvalues)
    scale = np.log1p(eigenvalues / median)
    return np.exp(scale) * median


_PSI_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "identity": _psi_identity,
    "powerlaw": _psi_powerlaw,
    "log": _psi_log,
    "adaptive": _psi_adaptive,
}


def _get_psi_callable(name: str, params: Optional[Dict[str, float]] = None) -> PsiCallable:
    if name not in _PSI_REGISTRY:
        available = ", ".join(sorted(_PSI_REGISTRY))
        raise KeyError(f"Unknown psi transform '{name}'. Available: {available}")

    params = params or {}

    def _wrapped(eigs: np.ndarray) -> np.ndarray:
        return _PSI_REGISTRY[name](eigs, **params)

    return _wrapped


# =============================================================================
# Core API
# =============================================================================

def compute_renormalized_entropy(
    data: np.ndarray,
    sfreq: float,
    channel_names: Optional[Sequence[str]] = None,
    config: Optional[RenormalizedEntropyConfig] = None,
) -> RenormalizationResult:
    """Compute the renormalised entropy from multichannel time series.

    The procedure mirrors the analytical steps described in Issartel's
    publications: generalised moments are extracted, their covariance matrix
    is renormalised with a spectral weighting :math:`\\psi(\\Sigma)`, and the
    differential entropy of the associated Gaussian measure is returned.

    Parameters
    ----------
    data : np.ndarray
        Array of shape ``(n_channels, n_samples)`` containing EEG signals.
    sfreq : float
        Sampling frequency in Hz.
    channel_names : Optional sequence of str
        Names of the channels corresponding to the first dimension of ``data``.
    config : Optional[RenormalizedEntropyConfig]
        Configuration instance. If omitted the defaults are used.

    Returns
    -------
    RenormalizationResult
        Container with the entropy (in nats and bits) and intermediate values.
    """

    if sfreq <= 0:
        raise ValueError("`sfreq` must be strictly positive")

    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim != 2:
        raise ValueError("`data` must be 1D (single channel) or 2D (channels x samples)")

    n_channels, n_samples = data.shape
    if n_channels < 1:
        raise ValueError("No channels provided")

    if channel_names is not None and len(channel_names) != n_channels:
        raise ValueError("Length of `channel_names` must match the number of channels")

    cfg = config or RenormalizedEntropyConfig()

    window_samples, step_samples = _compute_window_and_step(cfg, sfreq)
    windows = _extract_windows(data, window_samples, step_samples, cfg.max_windows)

    mu_matrix = _compute_mu_matrix(windows, cfg)
    raw_covariance, mean_mu = _estimate_covariance(mu_matrix, cfg.regularization)
    weighted_covariance, psi_eigs = _apply_psi(raw_covariance, cfg)

    entropy_nats, entropy_bits = _entropy_from_eigenvalues(psi_eigs, cfg.entropy_unit)

    if not cfg.return_intermediate:
        mu_matrix = None
        mean_mu = None

    result = RenormalizationResult(
        entropy_nats=entropy_nats,
        entropy_bits=entropy_bits,
        weighted_covariance=weighted_covariance,
        psi_eigenvalues=psi_eigs,
        raw_covariance=raw_covariance,
        mu_matrix=mu_matrix,
        mean_mu=mean_mu,
        config=cfg,
        channel_names=tuple(channel_names) if channel_names is not None else None,
        window_samples=window_samples,
        step_samples=step_samples,
        scales=tuple(cfg.scales) if cfg.scales is not None else None,
    )

    return result


def compute_entropy_from_raw(
    raw,  # type: ignore[valid-type]
    channel_names: Iterable[str],
    config: Optional[RenormalizedEntropyConfig] = None,
) -> RenormalizationResult:
    """Convenience wrapper for MNE ``Raw`` objects.

    The method pulls the requested EEG channels from the ``Raw`` instance and
    forwards the underlying NumPy arrays to :func:`compute_renormalized_entropy`.
    This keeps the scientific code strictly separated from the GUI layer while
    preserving traceability of the pre-processing steps applied to the data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw instance already loaded in CESA.
    channel_names : Iterable[str]
        Channels to include in the analysis.
    config : Optional[RenormalizedEntropyConfig]
        Optional configuration (defaults otherwise).
    """

    try:
        import mne  # type: ignore
    except Exception as exc:
        raise ImportError("mne must be installed to use `compute_entropy_from_raw`") from exc

    picks = mne.pick_channels(raw.info["ch_names"], include=list(channel_names))
    if len(picks) == 0:
        raise ValueError("None of the requested channels are present in the Raw object")

    data, _ = raw.get_data(picks=picks, return_times=True)
    selected_names = [raw.info["ch_names"][p] for p in picks]
    return compute_renormalized_entropy(data, raw.info["sfreq"], selected_names, config=config)


# =============================================================================
# Multiscale entropy API
# =============================================================================

def compute_multiscale_entropy(
    data: np.ndarray,
    sfreq: float,
    channel_names: Optional[Sequence[str]] = None,
    config: Optional[MultiscaleEntropyConfig] = None,
    *,
    progress_label: Optional[str] = None,
) -> MultiscaleEntropyResult:
    """Compute multiscale entropy (MSE) via EntropyHub's ``MSEn`` implementation."""

    try:
        import EntropyHub as EH  # type: ignore
        # Vérification de la version pour reproductibilité
        try:
            eh_version = getattr(EH, "__version__", "unknown")
        except Exception:
            eh_version = "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError("EntropyHub must be installed to compute multiscale entropy") from exc

    prefix = f"[{progress_label}] " if progress_label else ""

    def _log(msg: str) -> None:
        print(f"{prefix}{msg}", flush=True)

    if sfreq <= 0:
        raise ValueError("`sfreq` must be strictly positive")

    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim != 2:
        raise ValueError("`data` must be 1D or 2D (channels x samples)")

    cfg = config or MultiscaleEntropyConfig()
    n_channels, n_samples = data.shape
    if n_channels == 0 or n_samples == 0:
        raise ValueError("`data` must contain at least one channel and one sample")

    if channel_names is not None and len(channel_names) != n_channels:
        raise ValueError("Length of `channel_names` must match the number of channels")

    label_sequence = (
        tuple(channel_names) if channel_names is not None else tuple(f"ch{idx}" for idx in range(n_channels))
    )

    trimmed_samples = n_samples
    if cfg.max_samples is not None and n_samples > cfg.max_samples:
        trimmed_samples = cfg.max_samples
        _log(
            f"⚠️ CHECKPOINT MSE: truncating signal from {n_samples} to {trimmed_samples} samples "
            f"to satisfy max_samples={cfg.max_samples}"
        )
        data = data[:, : trimmed_samples]

    selected_scales = tuple(cfg.scales)
    
    # Reduced verbosity: removed per-epoch input validation log
    max_scale = int(selected_scales[-1])
    per_scale_values: Dict[int, list[float]] = {scale: [] for scale in selected_scales}
    entropy_by_scale: Dict[int, float] = {}
    channel_entropy: Dict[str, Dict[int, float]] = {name: {} for name in label_sequence}
    per_scale_details: Dict[int, Dict[str, object]] = {}
    channel_metadata: Dict[str, Dict[str, object]] = {}
    stored_signals = None

    if cfg.return_intermediate:
        _log("ℹ️ CHECKPOINT MSE: `return_intermediate` is ignored when using EntropyHub.MSEn.")

    # Downsampling avec anti-aliasing (moyenne par blocs) au lieu de linspace
    # pour éviter l'aliasing qui peut affecter différemment les différents stades
    if cfg.max_pattern_length is not None and cfg.max_pattern_length > 0 and data.shape[1] > cfg.max_pattern_length:
        # Utiliser un downsampling par moyenne (anti-aliasing) au lieu de linspace
        # pour préserver les caractéristiques fréquentielles du signal
        bin_size = max(1, int(np.ceil(data.shape[1] / cfg.max_pattern_length)))
        if bin_size > 1:
            # Downsampler les données avant le traitement
            n_bins = data.shape[1] // bin_size
            trimmed_data = data[:, :n_bins * bin_size]
            downsampled_data = trimmed_data.reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
            data = downsampled_data
            # Ajuster sfreq pour refléter le downsampling
            sfreq = sfreq / bin_size
            _log(
                f"⚠️ CHECKPOINT MSE: Downsampling signal from {trimmed_data.shape[1]} to {n_bins} samples "
                f"(bin_size={bin_size}, new_sfreq={sfreq:.2f} Hz) using anti-aliasing"
            )

    # Reduced verbosity: removed per-epoch initialization log
    # NOTE: EntropyHub's MSEn applies r * std(coarse_grained_signal) for each scale
    # when RadNew=1. This is the standard approach, but differs from some implementations
    # that use r * std(original_signal) for all scales.
    mobj = EH.MSobject("SampEn", m=int(cfg.m), r=float(cfg.r))

    for ch_idx, ch_name in enumerate(label_sequence):
        series = np.asarray(data[ch_idx], dtype=np.float64)
        original_series_size = series.size  # Note: après downsampling par moyenne si appliqué
        
        # Centrer le signal (soustraire la moyenne) pour éviter que les offsets DC affectent l'entropie
        # Cela est important car l'entropie est calculée sur les patterns, pas sur les valeurs absolues
        series_mean = np.mean(series)
        if abs(series_mean) > 1e-10:  # Seulement si la moyenne est significative
            series = series - series_mean

        ch_num = ch_idx + 1
        total_ch = len(label_sequence)
        series_duration = series.size / sfreq
        total_duration = series.size / sfreq

        # Validation: signal doit être assez long pour toutes les échelles demandées
        # Pour chaque scale τ, le signal coarse-grained a environ N/τ points
        # Il faut N/τ > m + 1 pour que SampEn soit calculable
        min_required_length = cfg.m + 1
        max_scale = int(selected_scales[-1]) if selected_scales else 1
        # Pour la plus grande échelle, le signal coarse-grained a environ series.size/max_scale points
        min_required_for_max_scale = (cfg.m + 1) * max_scale
        
        if series.size <= min_required_length:
            _log(
                f"⚠️  CHECKPOINT MSE: Channel '{ch_name}' ({ch_num}/{total_ch}) too short "
                f"({series.size} samples, {series_duration:.1f}s, "
                f"minimum required: {min_required_length}), skipping"
            )
            for scale in selected_scales:
                channel_entropy[ch_name][scale] = math.nan
            channel_metadata[ch_name] = {
                "status": "insufficient_length",
                "samples": int(series.size),
                "min_required": min_required_length,
            }
            continue
        
        # Avertissement si le signal est trop court pour les grandes échelles
        if series.size < min_required_for_max_scale:
            _log(
                f"⚠️  CHECKPOINT MSE: Channel '{ch_name}' may be too short for scale {max_scale} "
                f"({series.size} samples, recommended: >= {min_required_for_max_scale} for scale {max_scale})"
            )

        # Reduced verbosity: removed per-epoch channel processing log

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                # NOTE: RadNew=1 means r is applied relative to std(coarse_grained_signal) for each scale
                # This is the standard EntropyHub behavior and matches most literature
                ms_values, complexity_index = EH.MSEn(
                    series,
                    mobj,
                    Scales=max_scale,
                    Methodx="coarse",
                    RadNew=1,  # r * std(coarse_grained_signal) for each scale
                    Plotx=False,
                )
        except Exception as exc:  # pragma: no cover
            _log(
                f"❌ CHECKPOINT MSE: EntropyHub.MSEn failed on channel '{ch_name}' "
                f"({ch_num}/{total_ch}, {series.size} samples, {series_duration:.1f}s): {exc}"
            )
            for scale in selected_scales:
                channel_entropy[ch_name][scale] = math.nan
            channel_metadata[ch_name] = {
                "status": "error",
                "message": str(exc),
                "samples": int(series.size),
                "max_scale_requested": max_scale,
            }
            continue

        ms_values = np.asarray(ms_values, dtype=float).reshape(-1)
        
        # Validation: vérifier que toutes les échelles demandées sont disponibles
        if ms_values.size < max_scale:
            _log(
                f"⚠️  CHECKPOINT MSE: EntropyHub returned only {ms_values.size} scales "
                f"but {max_scale} were requested for channel '{ch_name}'"
            )
        
        channel_metadata[ch_name] = {
            "status": "ok",
            "complexity_index": float(complexity_index),
            "samples": int(series.size),
            "scales_returned": int(ms_values.size),
            "scales_requested": max_scale,
            "sfreq": float(sfreq),  # Inclure la fréquence d'échantillonnage (peut être modifiée par downsampling)
        }

        # Reduced verbosity: removed per-epoch channel completion log

        # Extraction des valeurs par échelle avec validation de l'indexation
        for scale in selected_scales:
            idx = scale - 1  # EntropyHub retourne les échelles 1-indexed, on convertit en 0-indexed
            if idx < 0:
                _log(f"⚠️  CHECKPOINT MSE: Invalid scale {scale} (must be >= 1)")
                channel_entropy[ch_name][scale] = math.nan
            elif idx >= ms_values.size:
                _log(
                    f"⚠️  CHECKPOINT MSE: Scale {scale} (index {idx}) not available "
                    f"(only {ms_values.size} scales returned) for channel '{ch_name}'"
                )
                channel_entropy[ch_name][scale] = math.nan
            else:
                value = float(ms_values[idx])
                # Validation: NaN ou inf indiquent un problème pour cette échelle
                if math.isnan(value) or math.isinf(value):
                    _log(
                        f"⚠️  CHECKPOINT MSE: Scale {scale} returned {value} for channel '{ch_name}' "
                        f"(signal may be too short or too regular for this scale)"
                    )
                channel_entropy[ch_name][scale] = value
                per_scale_values[scale].append(value)
    else:
        # executed if loop didn't encounter a `continue` due to exception; ensures metadata exists
        pass

    # Aggregate results across channels (reduced verbosity)
    for scale in selected_scales:
        values = per_scale_values[scale]
        mean_value = float(np.nanmean(values)) if values else math.nan
        entropy_by_scale[scale] = mean_value
        per_scale_details[scale] = {
            "status": "ok" if values else "no_channels",
            "mean_entropy": mean_value,
            "channels": len(values),
            "source": "EntropyHub.MSEn",
            "radial_rescaling": "r * std(coarse_grained_signal) per scale (RadNew=1)",
            "entropyhub_version": eh_version,
            "aggregation": "mean across channels",
            "note": "MSE values are averaged across channels. Individual channel values available in channel_entropy.",
            "channel_details": dict(channel_metadata),
        }

    # Reduced verbosity: only log summary if multiple channels or at completion milestone
    if n_channels > 1:
        ci_values_all = [
            ch_meta.get("complexity_index", math.nan)
            for ch_meta in channel_metadata.values()
            if isinstance(ch_meta, dict) and "complexity_index" in ch_meta
        ]
        if ci_values_all:
            valid_ci = [ci for ci in ci_values_all if not math.isnan(ci)]
            if valid_ci:
                mean_ci = float(np.nanmean(valid_ci))
                _log(
                    f"📈 CHECKPOINT MSE: Complexity Index (CI) - "
                    f"mean={mean_ci:.4f} (from {len(valid_ci)} channel(s))"
                )

    result = MultiscaleEntropyResult(
        entropy_by_scale=entropy_by_scale,
        channel_entropy=channel_entropy,
        sample_entropy_by_scale=per_scale_details,
        coarse_grained_signals=stored_signals,
        config=cfg,
        channel_names=tuple(label_sequence),
        sampling_frequency=sfreq,
        processed_samples=data.shape[1],
    )

    return result


def compute_multiscale_entropy_from_raw(
    raw,  # type: ignore[valid-type]
    channel_names: Iterable[str],
    config: Optional[MultiscaleEntropyConfig] = None,
    *,
    progress_label: Optional[str] = None,
) -> MultiscaleEntropyResult:
    """Convenience wrapper mirroring :func:`compute_entropy_from_raw`."""

    try:
        import mne  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError("mne must be installed to use `compute_multiscale_entropy_from_raw`") from exc

    picks = mne.pick_channels(raw.info["ch_names"], include=list(channel_names))
    if len(picks) == 0:
        raise ValueError("None of the requested channels are present in the Raw object")

    data, _ = raw.get_data(picks=picks, return_times=True)
    selected_names = [raw.info["ch_names"][p] for p in picks]

    prefix = f"[{progress_label}] " if progress_label else ""
    print(
        f"{prefix}📡 CHECKPOINT MSE: extracted {len(selected_names)} channels from Raw for analysis",
        flush=True,
    )

    return compute_multiscale_entropy(
        data,
        raw.info["sfreq"],
        selected_names,
        config=config,
        progress_label=progress_label,
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _compute_window_and_step(config: RenormalizedEntropyConfig, sfreq: float) -> Tuple[int, int]:
    """Convert temporal parameters into discrete samples.

    The sliding window and its overlap are expressed in seconds in the public
    configuration. Translating these values into integer samples early ensures
    that downstream window extraction remains numerically stable and easy to
    reason about during reviews.
    """

    window_samples = max(int(round(config.window_length * sfreq)), 1)
    step_fraction = max(1e-6, 1.0 - config.overlap)
    step_samples = max(int(round(window_samples * step_fraction)), 1)
    return window_samples, step_samples


def _extract_windows(
    data: np.ndarray,
    window_samples: int,
    step_samples: int,
    max_windows: Optional[int],
) -> np.ndarray:
    """Slice the continuous recording into overlapping windows.

    Each window preserves the original channel ordering to keep the moment
    computation reproducible. Limiting the total number of windows helps when
    reviewers want to replicate the analysis quickly on long recordings.
    """

    n_channels, n_samples = data.shape
    if n_samples < window_samples:
        raise ValueError("Signal shorter than the requested window size")

    indices = range(0, n_samples - window_samples + 1, step_samples)
    windows = []
    for start in indices:
        if max_windows is not None and len(windows) >= max_windows:
            break
        windows.append(data[:, start : start + window_samples])

    if not windows:
        raise RuntimeError("No windows extracted; check window length and overlap")

    return np.stack(windows, axis=0)


def _detrend(window: np.ndarray) -> np.ndarray:
    n_samples = window.shape[1]
    x = np.linspace(0.0, 1.0, n_samples, dtype=window.dtype)
    x_mean = np.mean(x)
    y = window
    slope = (np.sum((x - x_mean) * (y - y.mean(axis=1, keepdims=True)), axis=1) /
             np.sum((x - x_mean) ** 2))
    intercept = y.mean(axis=1) - slope * x_mean
    trend = slope[:, None] * x + intercept[:, None]
    return window - trend


def _compute_mu_matrix(windows: np.ndarray, config: RenormalizedEntropyConfig) -> np.ndarray:
    """Compute the matrix of generalised moments :math:`\\mu`.

    Each row corresponds to one temporal window and each column to a channel
    (or an augmented feature when multi-scale analysis is enabled). This
    explicit representation makes it straightforward to inspect the raw
    features contributing to the entropy estimate.
    """

    n_windows, n_channels, _ = windows.shape
    mu = np.empty((n_windows, n_channels), dtype=float)

    for idx in range(n_windows):
        window = windows[idx]
        if config.detrend:
            window = _detrend(window)

        mu[idx] = _generalized_moment(window, config)

    if config.scales:
        scaled_features = []
        for scale in config.scales:
            scaled_window_samples = max(int(round(scale / config.window_length)), 1)
            if scaled_window_samples == 1:
                scaled_features.append(mu)
            else:
                resampled = _resample_mu(mu, scaled_window_samples)
                scaled_features.append(resampled)
        mu = np.concatenate([mu] + scaled_features, axis=1)

    return mu


def _generalized_moment(window: np.ndarray, config: RenormalizedEntropyConfig) -> np.ndarray:
    """Compute the :math:`p`-order generalised moment for each channel.

    The expression is:

    .. math::

        \\mu_p = \\Big(\\frac{1}{N} \\sum_{i=1}^{N} |x_i - \\bar{x}|^p \\Big)^{1/p}

    where the centering term :math:`\bar{x}` is optional and controlled via
    ``config.central_moment``. Raising the result to ``1/p`` (when requested)
    yields the generalised mean, which aligns with Issartel's interpretation of
    energy across scales.
    """

    data = window
    if config.central_moment:
        data = data - data.mean(axis=1, keepdims=True)
    magnitude = np.abs(data) if not config.central_moment else data
    moment = np.mean(np.abs(magnitude) ** config.moment_order, axis=1)

    if config.normalize_moment:
        moment = np.power(np.maximum(moment, config.min_eigenvalue), 1.0 / config.moment_order)

    return moment


def _resample_mu(mu: np.ndarray, factor: int) -> np.ndarray:
    """Down-sample the moment matrix to emulate larger temporal scales."""

    if factor <= 1:
        return mu
    n_windows, n_features = mu.shape
    reduced = []
    for start in range(0, n_windows, factor):
        chunk = mu[start : start + factor]
        if chunk.size == 0:
            continue
        reduced.append(np.mean(chunk, axis=0))
    if not reduced:
        return mu
    return np.stack(reduced, axis=0)


def _estimate_covariance(mu_matrix: np.ndarray, regularization: float) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the covariance matrix of the moment ensemble.

    A light ridge regularisation is applied to stabilise the eigenvalue
    spectrum. The function returns both the covariance and the mean vector so
    reviewers can inspect the central tendency of the moments.
    """

    if mu_matrix.ndim != 2:
        raise ValueError("`mu_matrix` must be 2D")

    mean_mu = np.mean(mu_matrix, axis=0)
    centered = mu_matrix - mean_mu
    cov = np.cov(centered, rowvar=False, bias=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    n_features = cov.shape[0]
    cov += np.eye(n_features) * regularization
    return cov, mean_mu


def _apply_psi(covariance: np.ndarray, config: RenormalizedEntropyConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the renormalisation kernel on the covariance spectrum."""

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("`covariance` must be a square matrix")

    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, config.min_eigenvalue)

    psi = _get_psi_callable(config.psi_name, config.psi_params)
    psi_eigvals = psi(eigvals)
    psi_eigvals = np.maximum(psi_eigvals, config.min_eigenvalue)

    weighted_cov = eigvecs @ np.diag(psi_eigvals) @ eigvecs.T
    weighted_cov = _symmetrise(weighted_cov)
    return weighted_cov, psi_eigvals


def _entropy_from_eigenvalues(
    eigenvalues: np.ndarray,
    unit: str,
) -> Tuple[float, float]:
    """Convert eigenvalues of :math:`\\psi(\\Sigma)` into differential entropy."""

    k = eigenvalues.size
    ln_det = np.sum(np.log(eigenvalues))
    entropy_nats = 0.5 * (k * np.log(2.0 * np.pi * np.e) + ln_det)
    entropy_bits = entropy_nats / np.log(2.0)

    if unit == "nat":
        return entropy_nats, entropy_bits
    if unit == "bit":
        return entropy_nats, entropy_bits
    return entropy_nats, entropy_bits


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


__all__ = [
    "RenormalizedEntropyConfig",
    "RenormalizationResult",
    "compute_renormalized_entropy",
    "compute_entropy_from_raw",
    "MultiscaleEntropyConfig",
    "MultiscaleEntropyResult",
    "compute_multiscale_entropy",
    "compute_multiscale_entropy_from_raw",
]


