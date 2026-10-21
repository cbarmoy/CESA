"""Headband montage support for the classical SO-spindle coupling pipeline.

A headband only carries frontal electrodes and usually no EOG channel, so the
two assumptions of the PSG path do not hold: the montage must be reduced to
frontal derivations, and ocular artifacts have to be handled without a
reference eye channel.

Blinks are *masked*, not subtracted. A blink is a 50-500 ms deflection whose
spectrum overlaps the slow-oscillation band, so the EOG regression of
:mod:`CESA.ocular_cleaning` would also remove genuine slow oscillations. The
samples carrying a blink are excluded from detection instead, which is enough
because :func:`~CESA.sleep_pipeline.so_spindle_coupling.detect_slow_oscillations`
drops any cycle that is not fully inside the mask.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .so_spindle_coupling import bandpass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blink detection
# ---------------------------------------------------------------------------

#: Above the SO band (0.5-1.25 Hz) so a slow oscillation cannot trigger it.
BLINK_BAND_HZ: Tuple[float, float] = (1.5, 10.0)
BLINK_THRESHOLD_SD = 3.5
BLINK_MIN_AMPLITUDE_UV = 100.0
BLINK_MIN_DURATION_S = 0.05
BLINK_MAX_DURATION_S = 0.5
#: Rejected on both sides of the deflection, for the filter ringing it causes.
BLINK_PADDING_S = 0.2

#: Columns added to ``STATS_COLUMNS`` when blink rejection is active.
BLINK_STATS_COLUMNS: List[str] = ["n_blinks", "blink_rejected_s"]


@dataclass(frozen=True)
class Blink:
    """One ocular deflection on a frontal channel."""

    start_idx: int
    end_idx: int
    peak_idx: int
    peak_amplitude_uv: float


def detect_blinks(
    signal: np.ndarray,
    sfreq: float,
    *,
    band: Tuple[float, float] = BLINK_BAND_HZ,
    threshold_sd: float = BLINK_THRESHOLD_SD,
    min_amplitude_uv: float = BLINK_MIN_AMPLITUDE_UV,
    min_duration_s: float = BLINK_MIN_DURATION_S,
    max_duration_s: float = BLINK_MAX_DURATION_S,
) -> List[Blink]:
    """Detect blinks on one frontal channel (uV), without an EOG reference.

    A blink is a run where the signal deviates from its mean by more than
    ``threshold_sd`` standard deviations, lasting *min_duration_s* to
    *max_duration_s*, and peaking above *min_amplitude_uv*. Both criteria are
    required: the SD alone fires on any quiet stretch, the amplitude alone
    fires on slow drifts.
    """
    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size < 2 or sfreq <= 0:
        return []

    filtered = bandpass(sig, sfreq, *band)
    deviation = np.abs(filtered - float(np.mean(filtered)))
    sd = float(np.std(filtered))
    if sd <= 0.0:
        return []
    threshold = threshold_sd * sd

    min_samples = max(1, int(round(min_duration_s * sfreq)))
    max_samples = int(round(max_duration_s * sfreq))

    blinks: List[Blink] = []
    for start, end in _runs(deviation > threshold):
        length = end - start
        if not (min_samples <= length <= max_samples):
            continue
        segment = deviation[start:end]
        peak_offset = int(np.argmax(segment))
        peak = float(segment[peak_offset])
        if peak < float(min_amplitude_uv):
            continue
        blinks.append(Blink(
            start_idx=start,
            end_idx=end,
            peak_idx=start + peak_offset,
            peak_amplitude_uv=peak,
        ))
    return blinks


def blink_keep_mask(
    n_samples: int,
    sfreq: float,
    blinks: Sequence[Blink],
    *,
    padding_s: float = BLINK_PADDING_S,
) -> np.ndarray:
    """Boolean mask of the samples usable for detection (``True`` = keep).

    Each blink is widened by *padding_s* on both sides.
    """
    keep = np.ones(int(n_samples), dtype=bool)
    if not blinks or sfreq <= 0:
        return keep
    pad = int(round(float(padding_s) * float(sfreq)))
    for blink in blinks:
        start = max(0, int(blink.start_idx) - pad)
        end = min(int(n_samples), int(blink.end_idx) + pad)
        if end > start:
            keep[start:end] = False
    return keep


def _runs(flags: np.ndarray) -> List[Tuple[int, int]]:
    """Start/stop index pairs of contiguous ``True`` runs in *flags*."""
    if flags.size == 0:
        return []
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------

#: 10-20 sites a headband can reach. A derivation counts when one of its
#: members is frontal (``F3-M2``, ``F7-O1``).
FRONTAL_SITES: Tuple[str, ...] = (
    "fp1", "fp2", "fpz", "af3", "af4", "af7", "af8",
    "f3", "f4", "f7", "f8", "fz",
)

#: Never analysed, whatever the montage.
_NON_EEG_TOKENS: Tuple[str, ...] = (
    "e1", "e2", "eog", "loc", "roc",
    "ecg", "ekg", "emg", "chin", "menton",
    "spo2", "sao2", "pleth", "resp", "flow", "rip", "ptt", "pwa",
)

_SPLIT = re.compile(r"[^a-z0-9]+")


def _parts(name: str) -> List[str]:
    """Lower-case alphanumeric parts of a channel name (``F3-M2`` -> f3, m2)."""
    return [p for p in _SPLIT.split(str(name).lower().strip()) if p]


def _is_impedance(name: str) -> bool:
    """True for the impedance trace that doubles every electrode of an EDF."""
    lowered = str(name).lower()
    return "imp" in lowered or "ohm" in lowered


def is_frontal_channel(name: str) -> bool:
    """True when *name* is a frontal EEG channel or derivation."""
    if _is_impedance(name):
        return False
    parts = _parts(name)
    if not parts or any(part in _NON_EEG_TOKENS for part in parts):
        return False
    return any(part in FRONTAL_SITES for part in parts)


def select_headband_channels(
    names: Sequence[str],
    requested: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Frontal channels of *names*, and the ones left out.

    An empty *requested* means "every frontal channel of the montage": on a
    clinical PSG that is ``F3-M2`` and ``F4-M1``, on a headband the raw frontal
    electrodes. When a site is available both as a derivation and alone, only
    the derivation is kept: the monopolar trace is the same electrode without
    a reference.
    """
    available = [str(n) for n in names]
    candidates = (
        [c for c in (requested or []) if c in available] if requested else available
    )
    if requested and not candidates:
        raise ValueError(
            "aucun canal demande n'est present (" + ", ".join(requested) + ")"
        )

    picked = [name for name in candidates if is_frontal_channel(name)]
    referenced = {
        part
        for name in picked
        if len(_parts(name)) > 1
        for part in _parts(name)
        if part in FRONTAL_SITES
    }
    picked = [
        name for name in picked
        if len(_parts(name)) > 1 or _parts(name)[0] not in referenced
    ]
    skipped = [name for name in candidates if name not in picked]
    if not picked:
        raise ValueError(
            "aucun canal frontal parmi " + ", ".join(candidates or list(requested or []))
        )
    return picked, skipped
