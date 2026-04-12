"""Adaptive min-max envelope downsampler for real-time signal display.

The algorithm keeps the visual envelope intact (no aliasing of peaks or
troughs) while reducing the number of points sent to the GPU/painter to
roughly ``2 * target_points``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def downsample_minmax(
    data: np.ndarray,
    target_points: int = 4000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, values)`` using min-max envelope decimation.

    For every bin of ``bin_size`` samples the minimum and maximum are
    kept, interleaved so the resulting waveform faithfully reproduces
    the visual envelope of the original signal.

    If ``len(data) <= target_points`` the original arrays are returned
    unchanged (zero-copy).

    Parameters
    ----------
    data : 1-D array
        Signal samples (any dtype convertible to float64).
    target_points : int
        Approximate number of output points.  The actual output length
        is ``2 * n_bins`` where ``n_bins = len(data) // bin_size``.

    Returns
    -------
    indices : 1-D int array
        Sample indices into the original *data*.
    values : 1-D float64 array
        Corresponding sample values.
    """
    n = len(data)
    if n <= target_points or target_points < 4:
        return np.arange(n, dtype=np.intp), np.asarray(data, dtype=np.float64)

    half_target = max(2, target_points // 2)
    bin_size = max(1, n // half_target)
    n_bins = n // bin_size

    if n_bins < 1:
        return np.arange(n, dtype=np.intp), np.asarray(data, dtype=np.float64)

    trimmed = np.asarray(data[: n_bins * bin_size], dtype=np.float64)
    reshaped = trimmed.reshape(n_bins, bin_size)

    min_vals = reshaped.min(axis=1)
    max_vals = reshaped.max(axis=1)
    min_idx = reshaped.argmin(axis=1) + np.arange(n_bins) * bin_size
    max_idx = reshaped.argmax(axis=1) + np.arange(n_bins) * bin_size

    out_len = 2 * n_bins
    out_idx = np.empty(out_len, dtype=np.intp)
    out_val = np.empty(out_len, dtype=np.float64)

    # Within each bin, put the earlier index first to preserve time order
    for i in range(n_bins):
        if min_idx[i] <= max_idx[i]:
            out_idx[2 * i] = min_idx[i]
            out_val[2 * i] = min_vals[i]
            out_idx[2 * i + 1] = max_idx[i]
            out_val[2 * i + 1] = max_vals[i]
        else:
            out_idx[2 * i] = max_idx[i]
            out_val[2 * i] = max_vals[i]
            out_idx[2 * i + 1] = min_idx[i]
            out_val[2 * i + 1] = min_vals[i]

    return out_idx, out_val


def compute_target_points(widget_width_px: int) -> int:
    """Heuristic: aim for ~2x the pixel width of the plot area."""
    return max(500, min(8000, 2 * widget_width_px))
