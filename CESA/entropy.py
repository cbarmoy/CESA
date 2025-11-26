"""
CCESA (Complex EEG Studio Analysis) v1.0 - Entropy Analysis Module
=========================================================

Module d'analyse d'entropie renormée pour CESA v3.0.
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
Version: 3.0.0
Date: 2025-01-27
Licence: MIT
"""

from __future__ import annotations

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
    scales : Sequence[int]
        Positive integers describing each coarse-graining scale.
    max_samples : Optional[int]
        Optional cap on the number of samples to speed-up experiments.
    return_intermediate : bool
        Whether to keep the coarse-grained signals in the result payload.
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
        Raw instance already loaded in CCESA.
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
    """Compute multiscale entropy (MSE) on multichannel signals.

    The implementation strictly follows Costa et al. (2002): each scale
    performs coarse-graining followed by a Sample Entropy (SampEn) estimation.
    Numerous runtime checkpoints are emitted via ``print`` to help trace the
    computation path when experimenting with real EEG recordings.
    """

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

    _log(
        f"🔍 CHECKPOINT MSE: validated input "
        f"(channels={n_channels}, samples={data.shape[1]}, fs={sfreq} Hz, scales={cfg.scales})"
    )

    entropy_by_scale: Dict[int, float] = {}
    channel_entropy: Dict[str, Dict[int, float]] = {name: {} for name in label_sequence}
    per_scale_details: Dict[int, Dict[str, object]] = {}
    stored_signals: Dict[int, np.ndarray] = {}

    for idx, scale in enumerate(cfg.scales, start=1):
        _log(f"⏩ CHECKPOINT MSE: starting scale τ={scale} ({idx}/{len(cfg.scales)})")
        coarse = _coarse_grain_signal(data, scale)
        _log(
            f"🧱 CHECKPOINT MSE: coarse-graining done for τ={scale} "
            f"(coarse_samples={coarse.shape[1]})"
        )

        if coarse.shape[1] <= cfg.m + 1:
            _log(
                f"⚠️ CHECKPOINT MSE: insufficient samples ({coarse.shape[1]}) for τ={scale}, "
                f"requires > {cfg.m + 1}"
            )
            entropy_by_scale[scale] = math.nan
            per_scale_details[scale] = {
                "status": "insufficient_samples",
                "coarse_samples": coarse.shape[1],
            }
            continue

        per_channel_results = []
        channel_details: Dict[str, Dict[str, object]] = {}

        for ch_idx, ch_name in enumerate(label_sequence):
            entropy_value, detail = _compute_sample_entropy(
                coarse[ch_idx],
                cfg.m,
                cfg.r,
                max_points=cfg.max_pattern_length,
                channel_name=ch_name,
            )
            channel_entropy[ch_name][scale] = entropy_value
            per_channel_results.append(entropy_value)
            channel_details[ch_name] = detail

        aggregated_entropy = float(np.nanmean(per_channel_results)) if per_channel_results else math.nan
        entropy_by_scale[scale] = aggregated_entropy
        per_scale_details[scale] = {
            "status": "ok",
            "channel_details": channel_details,
            "mean_entropy": aggregated_entropy,
            "coarse_samples": coarse.shape[1],
        }

        _log(f"✅ CHECKPOINT MSE: finished τ={scale} | mean SampEn={aggregated_entropy:.4f}")

        if cfg.return_intermediate:
            stored_signals[scale] = coarse.copy()

    _log("🏁 CHECKPOINT MSE: multiscale entropy pipeline completed")

    result = MultiscaleEntropyResult(
        entropy_by_scale=entropy_by_scale,
        channel_entropy=channel_entropy,
        sample_entropy_by_scale=per_scale_details,
        coarse_grained_signals=stored_signals if cfg.return_intermediate else None,
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


def _coarse_grain_signal(data: np.ndarray, scale: int) -> np.ndarray:
    """Average the signal over non-overlapping blocks of length ``scale``."""

    if scale <= 1:
        return np.asarray(data, dtype=np.float64)

    n_channels, n_samples = data.shape
    usable = (n_samples // scale) * scale
    if usable == 0:
        return np.asarray(data[:, :0], dtype=np.float64)

    trimmed = data[:, :usable]
    reshaped = trimmed.reshape(n_channels, -1, scale)
    coarse = reshaped.mean(axis=2, dtype=np.float64)
    return coarse


def _compute_sample_entropy(
    signal: np.ndarray,
    m: int,
    r: float,
    *,
    max_points: Optional[int] = None,
    channel_name: Optional[str] = None,
) -> Tuple[float, Dict[str, object]]:
    """Compute Sample Entropy for a 1D signal with verbose checkpoints."""

    series = np.asarray(signal, dtype=np.float64)
    n = series.size
    checkpoint_prefix = f"[{channel_name}]" if channel_name is not None else ""
    detail: Dict[str, object] = {
        "channel": channel_name,
        "signal_length": n,
        "m": m,
        "r": r,
    }

    if max_points is not None and max_points > 0 and n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        series = series[idx]
        n = series.size
        detail["downsampled_to"] = n
        print(
            f"⚖️ CHECKPOINT SampEn {checkpoint_prefix}: downsampling series to {n} samples "
            f"for memory safety",
            flush=True,
        )

    if n <= m + 1:
        print(f"⚠️ CHECKPOINT SampEn {checkpoint_prefix}: insufficient length ({n})", flush=True)
        detail["status"] = "insufficient_length"
        return math.nan, detail

    std = float(np.std(series))
    tolerance = r * std if std > 0 else r * 1e-12
    detail["tolerance"] = tolerance

    patterns_m = _extract_pattern_matrix(series, m)
    patterns_m1 = _extract_pattern_matrix(series, m + 1)
    detail["patterns_m"] = patterns_m.shape[0]
    detail["patterns_m1"] = patterns_m1.shape[0]

    print(
        f"🧮 CHECKPOINT SampEn {checkpoint_prefix}: "
        f"patterns(m)={patterns_m.shape[0]}, patterns(m+1)={patterns_m1.shape[0]}, tol={tolerance:.3e}",
        flush=True,
    )

    matches_m = _count_pattern_matches(patterns_m, tolerance)
    matches_m1 = _count_pattern_matches(patterns_m1, tolerance)
    detail["matches_m"] = matches_m
    detail["matches_m1"] = matches_m1

    print(
        f"📊 CHECKPOINT SampEn {checkpoint_prefix}: matches(m)={matches_m}, matches(m+1)={matches_m1}",
        flush=True,
    )

    if matches_m == 0 or matches_m1 == 0:
        detail["status"] = "no_matches"
        print(f"⚠️ CHECKPOINT SampEn {checkpoint_prefix}: zero matches, entropy=inf", flush=True)
        return math.inf, detail

    ratio = matches_m1 / matches_m
    entropy_value = float(-math.log(ratio))
    detail["status"] = "ok"

    print(
        f"✅ CHECKPOINT SampEn {checkpoint_prefix}: SampEn={entropy_value:.4f}",
        flush=True,
    )

    return entropy_value, detail


def _extract_pattern_matrix(signal: np.ndarray, length: int) -> np.ndarray:
    """Return a matrix of all consecutive patterns of size ``length``."""

    if signal.size < length:
        return np.empty((0, length), dtype=np.float64)

    windows = np.lib.stride_tricks.sliding_window_view(signal, length)
    return np.asarray(windows, dtype=np.float64)


def _count_pattern_matches(patterns: np.ndarray, tolerance: float) -> int:
    """Count pattern pairs whose Chebyshev distance is below ``tolerance``."""

    n_patterns = patterns.shape[0]
    if n_patterns <= 1:
        return 0

    diffs = np.abs(patterns[:, None, :] - patterns[None, :, :])
    chebyshev = np.max(diffs, axis=2)
    mask = chebyshev <= tolerance

    # Remove self-comparisons; symmetric pairs are counted twice which is OK
    total_matches = int(np.sum(mask) - n_patterns)
    return max(total_matches, 0)


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


