"""Simple 1D median filter (ports medfilt1.m)."""

import numpy as np


def medfilt1(x: np.ndarray, n: int = 3) -> np.ndarray:
    """Apply 1D median filter with truncated edge windows.

    This matches MATLAB's medfilt1.m with edge windows that shrink
    rather than pad. For each point i, the window is
    [max(0, i-half), min(m, i+half+1)] where half = n//2.

    Parameters
    ----------
    x : np.ndarray
        Input signal, 1D array.
    n : int, default=3
        Window size.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)

    # Handle both row and column vectors - always work on flat
    was_row = x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1)
    x_flat = x.ravel()
    m = len(x_flat)

    y = np.zeros(m, dtype=np.float64)
    half_win = n // 2

    for i in range(m):
        idx_start = max(0, i - half_win)
        idx_end = min(m, i + half_win + 1)
        y[i] = np.median(x_flat[idx_start:idx_end])

    if was_row:
        return y
    return y.reshape(x.shape)
