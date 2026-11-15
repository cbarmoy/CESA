"""Event-locked figures for SO-spindle coupling: average, PETH, TFR.

These are the temporal companions of the circular statistics in
:mod:`CESA.coupling_plots`. They need the raw channel (uV) plus the trough
and spindle-peak sample indices; they are therefore computed only when the
caller asks for them.

Phase convention is unused here: time 0 is the SO trough (TFR, PETH) or the
spindle envelope peak (locked average), matching Hahn et al. 2020 / Helfrich.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from CESA.sleep_pipeline.so_spindle_coupling import (
    MIN_COUPLED_EVENTS,
    SO_BAND_HZ,
    bandpass,
)

logger = logging.getLogger(__name__)

TFR_TMIN_S = -2.0
TFR_TMAX_S = 2.0
TFR_FMIN_HZ = 5.0
TFR_FMAX_HZ = 30.0
TFR_WINDOW_S = 0.5
TFR_BASELINE = (-2.0, -1.5)

AVG_HALF_S = 2.5
PETH_HALF_S = 2.0
PETH_BINS = 40

_LIGHT = {
    "figure": "#FFFFFF",
    "text": "#1A1A1A",
    "trace": "#4C78A8",
    "so": "#E45756",
    "grid": "#C8C8C8",
    "bar": "#4C78A8",
    "warn": "#B45309",
}


def _new_figure(figsize: Tuple[float, float]):
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, facecolor=_LIGHT["figure"])
    return fig


def _indices_in_bounds(idx: Sequence[int], n: int, half: int) -> np.ndarray:
    arr = np.asarray(list(idx), dtype=int)
    if arr.size == 0:
        return arr
    return arr[(arr - half >= 0) & (arr + half < n)]


def spindle_locked_average(
    signal: np.ndarray,
    sfreq: float,
    peak_idx: Sequence[int],
    *,
    half_s: float = AVG_HALF_S,
    so_band: Tuple[float, float] = SO_BAND_HZ,
    min_events: int = MIN_COUPLED_EVENTS,
    fig=None,
):
    """Z-normalised EEG average locked to spindle peaks, with SO overlay.

    Each epoch is z-scored in the time domain before averaging so amplitude
    differences between recordings cannot masquerade as coupling (Hahn 2020).
    """
    sig = np.asarray(signal, dtype=float).ravel()
    half = int(round(float(half_s) * float(sfreq)))
    peaks = _indices_in_bounds(peak_idx, sig.size, half)
    times = np.arange(-half, half) / float(sfreq)
    so = bandpass(sig, sfreq, *so_band)

    fig = fig if fig is not None else _new_figure((7.2, 4.2))
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor(_LIGHT["figure"])
    if peaks.size < int(min_events):
        ax.text(
            0.5, 0.5,
            f"{peaks.size} fuseau(x) < {int(min_events)} : moyenne non tracee",
            transform=ax.transAxes, ha="center", va="center",
            color=_LIGHT["warn"],
        )
        ax.set_axis_off()
        return fig

    epochs = np.empty((peaks.size, 2 * half), dtype=float)
    so_epochs = np.empty_like(epochs)
    for i, peak in enumerate(peaks):
        chunk = sig[peak - half: peak + half]
        sd = float(np.std(chunk))
        epochs[i] = (chunk - float(np.mean(chunk))) / (sd if sd > 1e-12 else 1.0)
        so_epochs[i] = so[peak - half: peak + half]

    mean = epochs.mean(axis=0)
    sem = epochs.std(axis=0, ddof=1) / np.sqrt(peaks.size)
    so_mean = so_epochs.mean(axis=0)
    so_scale = (np.nanmax(np.abs(mean)) / (np.nanmax(np.abs(so_mean)) + 1e-12)) * 0.6

    ax.fill_between(
        times, mean - sem, mean + sem,
        color=_LIGHT["trace"], alpha=0.25, linewidth=0,
    )
    ax.plot(times, mean, color=_LIGHT["trace"], lw=1.6, label="EEG (z)")
    ax.plot(
        times, so_mean * so_scale, color=_LIGHT["so"], lw=1.4,
        label="SO (< 1.25 Hz), echelle",
    )
    ax.axvline(0.0, color=_LIGHT["grid"], lw=0.8, ls="--")
    ax.set_xlabel("Temps depuis le pic du fuseau (s)")
    ax.set_ylabel("Amplitude (z)")
    ax.set_title(f"Moyenne fuseau-verrouillee  n = {peaks.size}")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def peri_event_histogram(
    trough_idx: Sequence[int],
    peak_idx: Sequence[int],
    sfreq: float,
    *,
    half_s: float = PETH_HALF_S,
    bins: int = PETH_BINS,
    fig=None,
):
    """Histogram of spindle peaks relative to the nearest SO trough."""
    troughs = np.asarray(list(trough_idx), dtype=float)
    peaks = np.asarray(list(peak_idx), dtype=float)
    fig = fig if fig is not None else _new_figure((7.2, 3.6))
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor(_LIGHT["figure"])
    if troughs.size == 0 or peaks.size == 0 or sfreq <= 0:
        ax.text(
            0.5, 0.5, "aucun evenement pour l'histogramme",
            transform=ax.transAxes, ha="center", va="center",
            color=_LIGHT["warn"],
        )
        ax.set_axis_off()
        return fig

    half = float(half_s) * float(sfreq)
    lags = []
    for peak in peaks:
        nearest = troughs[np.argmin(np.abs(troughs - peak))]
        delta = peak - nearest
        if abs(delta) <= half:
            lags.append(delta / float(sfreq))
    lags = np.asarray(lags, dtype=float)

    ax.hist(
        lags, bins=int(bins), range=(-float(half_s), float(half_s)),
        color=_LIGHT["bar"], edgecolor=_LIGHT["figure"],
    )
    ax.axvline(0.0, color=_LIGHT["so"], lw=1.2, label="creux SO")
    ax.set_xlabel("Temps du pic de fuseau relativement au creux SO (s)")
    ax.set_ylabel("Nombre de fuseaux")
    ax.set_title(f"Histogramme peri-evenement  n = {lags.size}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def so_trough_tfr(
    signal: np.ndarray,
    sfreq: float,
    trough_idx: Sequence[int],
    *,
    tmin: float = TFR_TMIN_S,
    tmax: float = TFR_TMAX_S,
    fmin: float = TFR_FMIN_HZ,
    fmax: float = TFR_FMAX_HZ,
    window_s: float = TFR_WINDOW_S,
    baseline: Tuple[float, float] = TFR_BASELINE,
    so_band: Tuple[float, float] = SO_BAND_HZ,
    min_events: int = MIN_COUPLED_EVENTS,
    fig=None,
):
    """SO-trough-locked time-frequency representation, baseline z-scored.

    Frequencies 5–30 Hz, window ~0.5 s, baseline −2.0 to −1.5 s relative to
    the trough, as in Hahn et al. 2020 Fig. 3G. The mean SO waveform is
    superimposed.
    """
    from scipy.signal import stft

    sig = np.asarray(signal, dtype=float).ravel()
    fig = fig if fig is not None else _new_figure((7.6, 4.6))
    fig.clear()
    ax = fig.add_subplot(111)
    troughs = np.asarray(list(trough_idx), dtype=int)
    if troughs.size < int(min_events) or sfreq <= 0 or sig.size < 2:
        ax.text(
            0.5, 0.5,
            f"{troughs.size} onde(s) lente(s) < {int(min_events)} : TFR non tracee",
            transform=ax.transAxes, ha="center", va="center",
            color=_LIGHT["warn"],
        )
        ax.set_axis_off()
        return fig

    nperseg = max(8, int(round(float(window_s) * float(sfreq))))
    noverlap = int(round(0.8 * nperseg))
    freqs, times_abs, zxx = stft(
        sig, fs=float(sfreq), nperseg=nperseg, noverlap=noverlap,
        window="hann", boundary=None,
    )
    band = (freqs >= float(fmin)) & (freqs <= float(fmax))
    freqs = freqs[band]
    power = np.abs(zxx[band]) ** 2

    rel = np.arange(tmin, tmax + 1e-9, times_abs[1] - times_abs[0] if times_abs.size > 1 else 0.05)
    if rel.size < 4:
        rel = np.linspace(tmin, tmax, 80)
    trials: List[np.ndarray] = []
    so = bandpass(sig, sfreq, *so_band)
    so_half = int(round(max(abs(tmin), abs(tmax)) * float(sfreq)))
    so_epochs: List[np.ndarray] = []
    so_times = np.arange(-so_half, so_half) / float(sfreq)

    for trough in troughs:
        t0 = trough / float(sfreq)
        cols = np.interp(t0 + rel, times_abs, np.arange(times_abs.size), left=-1, right=-1)
        if np.any(cols < 0):
            continue
        idx = np.clip(np.round(cols).astype(int), 0, power.shape[1] - 1)
        trials.append(power[:, idx])
        if trough - so_half >= 0 and trough + so_half < sig.size:
            so_epochs.append(so[trough - so_half: trough + so_half])

    if len(trials) < int(min_events):
        ax.text(
            0.5, 0.5,
            f"{len(trials)} epoques TFR < {int(min_events)} : TFR non tracee",
            transform=ax.transAxes, ha="center", va="center",
            color=_LIGHT["warn"],
        )
        ax.set_axis_off()
        return fig

    stack = np.stack(trials, axis=0)
    base = (rel >= float(baseline[0])) & (rel <= float(baseline[1]))
    if not np.any(base):
        base = rel <= rel.min() + 0.4
    mu = stack[:, :, base].mean(axis=2, keepdims=True)
    sd = stack[:, :, base].std(axis=2, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    zmean = ((stack - mu) / sd).mean(axis=0)

    mesh = ax.pcolormesh(
        rel, freqs, zmean, shading="auto", cmap="RdBu_r",
        vmin=-np.nanpercentile(np.abs(zmean), 98),
        vmax=np.nanpercentile(np.abs(zmean), 98),
    )
    fig.colorbar(mesh, ax=ax, label="Puissance (z vs baseline)")
    ax.axvline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Temps depuis le creux SO (s)")
    ax.set_ylabel("Frequence (Hz)")
    ax.set_title(f"TFR alignee sur le creux SO  n = {len(trials)}")
    ax.set_ylim(float(fmin), float(fmax))

    if so_epochs:
        so_mean = np.mean(np.stack(so_epochs, axis=0), axis=0)
        ax_so = ax.twinx()
        ax_so.plot(so_times, so_mean, color="k", lw=1.3, alpha=0.85)
        ax_so.set_ylabel("SO (uV)")
        ax_so.spines["top"].set_visible(False)
    return fig


def event_indices_from_dicts(
    events: Sequence,
    sfreq: float,
    channel: str,
) -> Tuple[List[int], List[int]]:
    """Trough and spindle-peak sample indices of *channel* from UI/event dicts.

    Other channels are ignored. YASA events (no ``trough_s`` / ``peak_s``, or
    ``detector=yasa``) are ignored so a neighbour channel cannot change C3's
    coupling figures.
    """
    troughs: List[int] = []
    peaks: List[int] = []
    for raw in events:
        event = raw.to_dict() if hasattr(raw, "to_dict") else raw
        meta = dict(event.get("metadata") or {})
        ch = str(event.get("channel") or meta.get("channel") or "")
        if ch != channel:
            continue
        detector = str(meta.get("detector") or event.get("detector") or "")
        if detector == "yasa":
            continue
        kind = str(event.get("type") or event.get("event_type") or "")
        if kind in ("slow_wave", "SLOW_WAVE") and meta.get("trough_s") is not None:
            troughs.append(int(round(float(meta["trough_s"]) * float(sfreq))))
        if kind in ("spindle", "SPINDLE") and meta.get("peak_s") is not None:
            peaks.append(int(round(float(meta["peak_s"]) * float(sfreq))))
    return troughs, peaks


class CohortEventPlots:
    """Running-mean accumulator for batch TFR / averages, one panel per channel."""

    def __init__(self) -> None:
        self._tfr: Dict[str, Dict[str, object]] = {}
        self._avg: Dict[str, Dict[str, object]] = {}
        self._peth: Dict[str, List[float]] = defaultdict(list)
        self._sfreq: Optional[float] = None

    def add_recording(
        self,
        channel: str,
        signal: np.ndarray,
        sfreq: float,
        trough_idx: Sequence[int],
        peak_idx: Sequence[int],
        *,
        want_tfr: bool = False,
        want_averages: bool = False,
    ) -> None:
        if not want_tfr and not want_averages:
            return
        self._sfreq = float(sfreq)
        sig = np.asarray(signal, dtype=float).ravel()
        if want_averages:
            self._accumulate_average(channel, sig, sfreq, peak_idx)
            self._accumulate_peth(channel, trough_idx, peak_idx, sfreq)
        if want_tfr:
            self._accumulate_tfr(channel, sig, sfreq, trough_idx)

    def save(
        self,
        output_dir: Union[str, os.PathLike],
        *,
        prefix: str = "coupling",
        tfr: bool = False,
        averages: bool = False,
        dpi: int = 200,
    ) -> List[str]:
        import matplotlib

        matplotlib.use("Agg", force=False)
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        os.makedirs(str(output_dir), exist_ok=True)
        written: List[str] = []
        jobs: List[Tuple[str, object]] = []
        if averages:
            jobs.append(("moyenne", self._figure_averages()))
            jobs.append(("peth", self._figure_peth()))
        if tfr:
            jobs.append(("tfr", self._figure_tfr()))
        for name, fig in jobs:
            if fig is None:
                continue
            try:
                path = os.path.join(str(output_dir), f"{prefix}_{name}.png")
                FigureCanvasAgg(fig)
                fig.savefig(
                    path, dpi=int(dpi), bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                )
                written.append(path)
            except Exception as exc:
                logger.warning("Figure '%s' non generee: %s", name, exc)
        return written

    # ------------------------------------------------------------------

    def _accumulate_average(self, channel, sig, sfreq, peak_idx) -> None:
        half = int(round(AVG_HALF_S * float(sfreq)))
        peaks = _indices_in_bounds(peak_idx, sig.size, half)
        if peaks.size == 0:
            return
        so = bandpass(sig, sfreq, *SO_BAND_HZ)
        acc = self._avg.setdefault(channel, {"n": 0, "sum": None, "so": None})
        for peak in peaks:
            chunk = sig[peak - half: peak + half]
            sd = float(np.std(chunk))
            z = (chunk - float(np.mean(chunk))) / (sd if sd > 1e-12 else 1.0)
            so_chunk = so[peak - half: peak + half]
            if acc["sum"] is None:
                acc["sum"] = z.astype(float)
                acc["so"] = so_chunk.astype(float)
            else:
                acc["sum"] += z
                acc["so"] += so_chunk
            acc["n"] += 1

    def _accumulate_peth(self, channel, trough_idx, peak_idx, sfreq) -> None:
        troughs = np.asarray(list(trough_idx), dtype=float)
        peaks = np.asarray(list(peak_idx), dtype=float)
        if troughs.size == 0 or peaks.size == 0:
            return
        half = PETH_HALF_S * float(sfreq)
        for peak in peaks:
            nearest = troughs[np.argmin(np.abs(troughs - peak))]
            delta = peak - nearest
            if abs(delta) <= half:
                self._peth[channel].append(delta / float(sfreq))

    def _accumulate_tfr(self, channel, sig, sfreq, trough_idx) -> None:
        from scipy.signal import stft

        troughs = np.asarray(list(trough_idx), dtype=int)
        if troughs.size == 0:
            return
        nperseg = max(8, int(round(TFR_WINDOW_S * float(sfreq))))
        noverlap = int(round(0.8 * nperseg))
        freqs, times_abs, zxx = stft(
            sig, fs=float(sfreq), nperseg=nperseg, noverlap=noverlap,
            window="hann", boundary=None,
        )
        band = (freqs >= TFR_FMIN_HZ) & (freqs <= TFR_FMAX_HZ)
        freqs = freqs[band]
        power = np.abs(zxx[band]) ** 2
        step = times_abs[1] - times_abs[0] if times_abs.size > 1 else 0.05
        rel = np.arange(TFR_TMIN_S, TFR_TMAX_S + 1e-9, step)
        acc = self._tfr.get(channel)
        for trough in troughs:
            t0 = trough / float(sfreq)
            cols = np.interp(
                t0 + rel, times_abs, np.arange(times_abs.size), left=-1, right=-1,
            )
            if np.any(cols < 0):
                continue
            idx = np.clip(np.round(cols).astype(int), 0, power.shape[1] - 1)
            trial = power[:, idx]
            base = (rel >= TFR_BASELINE[0]) & (rel <= TFR_BASELINE[1])
            if not np.any(base):
                continue
            mu = trial[:, base].mean(axis=1, keepdims=True)
            sd = trial[:, base].std(axis=1, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            z = (trial - mu) / sd
            if acc is None:
                acc = {"n": 0, "sum": z.astype(float), "freqs": freqs, "times": rel}
                self._tfr[channel] = acc
            else:
                acc["sum"] += z
            acc["n"] += 1

    def _figure_averages(self):
        channels = sorted(self._avg)
        if not channels or self._sfreq is None:
            return None
        n_cols = min(3, len(channels))
        n_rows = int(np.ceil(len(channels) / n_cols))
        fig = _new_figure((5.0 * n_cols, 3.4 * n_rows))
        half = int(round(AVG_HALF_S * self._sfreq))
        times = np.arange(-half, half) / self._sfreq
        for index, channel in enumerate(channels, start=1):
            ax = fig.add_subplot(n_rows, n_cols, index)
            acc = self._avg[channel]
            n = int(acc["n"])
            ax.set_title(f"{channel}  n = {n}")
            if n < MIN_COUPLED_EVENTS or acc["sum"] is None:
                ax.text(
                    0.5, 0.5, "moyenne non tracee",
                    transform=ax.transAxes, ha="center", color=_LIGHT["warn"],
                )
                ax.set_axis_off()
                continue
            mean = acc["sum"] / n
            so = acc["so"] / n
            scale = (np.nanmax(np.abs(mean)) / (np.nanmax(np.abs(so)) + 1e-12)) * 0.6
            ax.plot(times, mean, color=_LIGHT["trace"], lw=1.4)
            ax.plot(times, so * scale, color=_LIGHT["so"], lw=1.2)
            ax.axvline(0.0, color=_LIGHT["grid"], lw=0.7, ls="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle("Moyenne fuseau-verrouillee (EEG z, overlay SO)")
        fig.subplots_adjust(top=0.88, hspace=0.45)
        return fig

    def _figure_peth(self):
        channels = sorted(self._peth)
        if not channels:
            return None
        n_cols = min(3, len(channels))
        n_rows = int(np.ceil(len(channels) / n_cols))
        fig = _new_figure((5.0 * n_cols, 3.2 * n_rows))
        for index, channel in enumerate(channels, start=1):
            ax = fig.add_subplot(n_rows, n_cols, index)
            lags = np.asarray(self._peth[channel], dtype=float)
            ax.set_title(f"{channel}  n = {lags.size}")
            if lags.size == 0:
                ax.set_axis_off()
                continue
            ax.hist(
                lags, bins=PETH_BINS, range=(-PETH_HALF_S, PETH_HALF_S),
                color=_LIGHT["bar"], edgecolor=_LIGHT["figure"],
            )
            ax.axvline(0.0, color=_LIGHT["so"], lw=1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle("Pics de fuseaux relativement au creux SO")
        fig.subplots_adjust(top=0.88, hspace=0.45)
        return fig

    def _figure_tfr(self):
        channels = sorted(self._tfr)
        if not channels:
            return None
        n_cols = min(3, len(channels))
        n_rows = int(np.ceil(len(channels) / n_cols))
        fig = _new_figure((5.2 * n_cols, 3.6 * n_rows))
        for index, channel in enumerate(channels, start=1):
            ax = fig.add_subplot(n_rows, n_cols, index)
            acc = self._tfr[channel]
            n = int(acc["n"])
            ax.set_title(f"{channel}  n = {n}")
            if n < MIN_COUPLED_EVENTS:
                ax.text(
                    0.5, 0.5, "TFR non tracee",
                    transform=ax.transAxes, ha="center", color=_LIGHT["warn"],
                )
                ax.set_axis_off()
                continue
            zmean = acc["sum"] / n
            vmax = float(np.nanpercentile(np.abs(zmean), 98) or 1.0)
            ax.pcolormesh(
                acc["times"], acc["freqs"], zmean, shading="auto",
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            )
            ax.axvline(0.0, color="k", lw=0.7, ls="--")
            ax.set_ylabel("Hz")
        fig.suptitle("TFR alignee sur le creux SO (z vs baseline -2.0..-1.5 s)")
        fig.subplots_adjust(top=0.88, hspace=0.45)
        return fig
