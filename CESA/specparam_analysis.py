"""SpecParam (FOOOF) fitting on Welch PSDs for CESA.

Separates aperiodic 1/f background from periodic peaks. Uses the ``specparam``
package (v2 API) with a graceful fallback to the legacy ``fooof`` package.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from CESA.spectral_analysis import (
    compute_psd_welch,
    compute_stage_psd_welch_for_array,
)

logger = logging.getLogger(__name__)


class SpecParamUnavailableError(RuntimeError):
    """Raised when neither specparam nor fooof can be imported."""


def _import_fitter():
    try:
        from specparam import SpectralModel  # type: ignore

        return ("specparam", SpectralModel)
    except ImportError:
        pass
    try:
        from fooof import FOOOF  # type: ignore

        return ("fooof", FOOOF)
    except ImportError as exc:
        raise SpecParamUnavailableError(
            "Installez specparam>=2.0.0rc0 (ou fooof) pour l'analyse SpecParam."
        ) from exc


def fit_specparam(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: Tuple[float, float] = (0.5, 40.0),
    *,
    peak_width_limits: Tuple[float, float] = (0.5, 12.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.0,
    aperiodic_mode: str = "fixed",
) -> Dict[str, Any]:
    """Fit SpecParam / FOOOF on a 1D PSD.

    Returns a dict with ``aperiodic_offset``, ``aperiodic_exponent``,
    ``peaks`` (list of ``{cf, pw, bw}``), ``r_squared``, ``error``,
    ``freqs``, ``psd``, ``modeled_spectrum``, ``warning``.
    """
    freqs = np.asarray(freqs, dtype=float).ravel()
    psd = np.asarray(psd, dtype=float).ravel()
    out: Dict[str, Any] = {
        "aperiodic_offset": float("nan"),
        "aperiodic_exponent": float("nan"),
        "peaks": [],
        "r_squared": float("nan"),
        "error": float("nan"),
        "freqs": freqs,
        "psd": psd,
        "modeled_spectrum": np.array([]),
        "warning": "",
        "ok": False,
    }
    if freqs.size < 8 or psd.size != freqs.size:
        out["warning"] = "PSD trop court ou incompatible."
        return out
    if not np.all(np.isfinite(psd)) or np.any(psd <= 0):
        psd = np.clip(np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0), 1e-20, None)

    fmin, fmax = float(freq_range[0]), float(freq_range[1])
    mask = (freqs >= fmin) & (freqs <= fmax)
    if int(mask.sum()) < 8:
        out["warning"] = "Pas assez de points dans la bande SpecParam."
        return out
    freqs_f = freqs[mask]
    psd_f = psd[mask]
    out["freqs"] = freqs_f
    out["psd"] = psd_f

    try:
        backend, Fitter = _import_fitter()
    except SpecParamUnavailableError as exc:
        out["warning"] = str(exc)
        return out

    try:
        if backend == "specparam":
            fm = Fitter(aperiodic_mode=aperiodic_mode, verbose=False)
            # peak settings via algorithm kwargs when supported
            try:
                fm.fit(freqs_f, psd_f, [fmin, fmax])
            except TypeError:
                fm.fit(freqs_f, psd_f)
            fr = fm.results.get_results()
            ap = np.asarray(getattr(fr, "aperiodic_fit", []), dtype=float).ravel()
            if ap.size < 2:
                # Fallback: params.asdict()
                try:
                    ad = fm.results.params.asdict()
                    ap = np.asarray(ad.get("aperiodic_fit", []), dtype=float).ravel()
                except Exception:
                    ap = np.array([])
            if ap.size >= 2:
                out["aperiodic_offset"] = float(ap[0])
                out["aperiodic_exponent"] = float(ap[1])
            peaks = np.asarray(getattr(fr, "peak_fit", []), dtype=float)
            if peaks.size == 0:
                try:
                    ad = fm.results.params.asdict()
                    peaks = np.asarray(ad.get("peak_fit", []), dtype=float)
                except Exception:
                    peaks = np.array([])
            if peaks.ndim == 1 and peaks.size >= 3:
                peaks = peaks.reshape(1, -1)
            if peaks.ndim == 2 and peaks.size:
                for row in peaks:
                    if len(row) >= 3 and np.isfinite(row[0]):
                        out["peaks"].append({
                            "cf": float(row[0]),
                            "pw": float(row[1]),
                            "bw": float(row[2]),
                        })
            metrics = getattr(fr, "metrics", None)
            if not isinstance(metrics, dict):
                metrics = {}
            # Alternate metric keys across specparam versions
            r2 = metrics.get("gof_rsquared", metrics.get("r_squared", float("nan")))
            err = metrics.get("error_mae", metrics.get("error", float("nan")))
            try:
                out["r_squared"] = float(r2)
            except (TypeError, ValueError):
                out["r_squared"] = float("nan")
            try:
                out["error"] = float(err)
            except (TypeError, ValueError):
                out["error"] = float("nan")
            try:
                modeled = fm.results.model.modeled_spectrum()
                out["modeled_spectrum"] = np.asarray(modeled, dtype=float).ravel()
            except Exception:
                out["modeled_spectrum"] = np.array([])
        else:
            # Legacy FOOOF
            fm = Fitter(
                peak_width_limits=peak_width_limits,
                max_n_peaks=max_n_peaks,
                min_peak_height=min_peak_height,
                aperiodic_mode=aperiodic_mode,
                verbose=False,
            )
            fm.fit(freqs_f, psd_f, [fmin, fmax])
            ap = np.asarray(fm.aperiodic_params_, dtype=float).ravel()
            if ap.size >= 2:
                out["aperiodic_offset"] = float(ap[0])
                out["aperiodic_exponent"] = float(ap[1])
            peaks = np.asarray(fm.peak_params_, dtype=float)
            if peaks.ndim == 2 and peaks.size:
                for row in peaks:
                    if len(row) >= 3 and np.isfinite(row[0]):
                        out["peaks"].append({
                            "cf": float(row[0]),
                            "pw": float(row[1]),
                            "bw": float(row[2]),
                        })
            out["r_squared"] = float(getattr(fm, "r_squared_", float("nan")))
            out["error"] = float(getattr(fm, "error_", float("nan")))
            out["modeled_spectrum"] = np.asarray(
                getattr(fm, "fooofed_spectrum_", []), dtype=float
            ).ravel()
        out["ok"] = bool(np.isfinite(out["aperiodic_exponent"]))
    except Exception as exc:
        logger.warning("SpecParam fit failed: %s", exc, exc_info=True)
        out["warning"] = f"Fit echoue: {exc}"
    return out


def fit_specparam_on_signal(
    signal: np.ndarray,
    sfreq: float,
    freq_range: Tuple[float, float] = (0.5, 40.0),
    **kwargs: Any,
) -> Dict[str, Any]:
    """Welch PSD then SpecParam fit on a continuous signal (µV)."""
    freqs, psd = compute_psd_welch(np.asarray(signal, dtype=float).ravel(), float(sfreq))
    if freqs.size == 0:
        return fit_specparam(np.array([]), np.array([]), freq_range, **kwargs)
    return fit_specparam(freqs, psd, freq_range, **kwargs)


def fit_specparam_by_stage(
    signal: np.ndarray,
    sfreq: float,
    scoring_df: Optional[pd.DataFrame],
    epoch_s: float = 30.0,
    freq_range: Tuple[float, float] = (0.5, 40.0),
    stages: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Fit SpecParam per sleep stage using Welch stage-aggregated PSDs."""
    if stages is None:
        stages = ["W", "N1", "N2", "N3", "R"]
    fmin, fmax = float(freq_range[0]), float(freq_range[1])
    stage_psd = compute_stage_psd_welch_for_array(
        signal=np.asarray(signal, dtype=float).ravel(),
        fs=float(sfreq),
        scoring_df=scoring_df,
        epoch_len=float(epoch_s),
        stages=list(stages),
        fmin=fmin,
        fmax=fmax,
        equalize_epochs=True,
        robust_stats=True,
        band_filter=None,
        notch_hz=None,
    )
    results: Dict[str, Dict[str, Any]] = {}
    for stage in stages:
        payload = stage_psd.get(stage)
        if not payload or len(payload) < 2:
            results[stage] = {
                "ok": False,
                "warning": "Pas de PSD pour ce stade.",
                "aperiodic_offset": float("nan"),
                "aperiodic_exponent": float("nan"),
                "peaks": [],
                "r_squared": float("nan"),
                "error": float("nan"),
                "freqs": np.array([]),
                "psd": np.array([]),
                "modeled_spectrum": np.array([]),
                "n_epochs": 0,
            }
            continue
        freqs, mean_psd, _sem, n_epochs = payload[0], payload[1], payload[2], payload[3]
        fit = fit_specparam(freqs, mean_psd, freq_range, **kwargs)
        fit["n_epochs"] = int(n_epochs)
        results[stage] = fit
    return results


def results_to_rows(
    results: Dict[str, Dict[str, Any]],
    *,
    channel: str = "",
) -> List[Dict[str, Any]]:
    """Flatten SpecParam results (global or by-stage) into CSV-friendly rows."""
    rows: List[Dict[str, Any]] = []
    for key, fit in results.items():
        stage = key if key else "all"
        base = {
            "channel": channel,
            "stage": stage,
            "aperiodic_offset": fit.get("aperiodic_offset"),
            "aperiodic_exponent": fit.get("aperiodic_exponent"),
            "r_squared": fit.get("r_squared"),
            "error": fit.get("error"),
            "n_peaks": len(fit.get("peaks") or []),
            "n_epochs": fit.get("n_epochs", ""),
            "warning": fit.get("warning", ""),
            "ok": bool(fit.get("ok")),
        }
        peaks = fit.get("peaks") or []
        if not peaks:
            rows.append({**base, "peak_cf": "", "peak_pw": "", "peak_bw": ""})
        else:
            for peak in peaks:
                rows.append({
                    **base,
                    "peak_cf": peak.get("cf"),
                    "peak_pw": peak.get("pw"),
                    "peak_bw": peak.get("bw"),
                })
    return rows
