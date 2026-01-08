"""Jacobi polynomial evaluation (ports j_polynomial.m)."""

import numpy as np


def j_polynomial(
    m: int, n: int, alpha: float, beta: float, x: np.ndarray
) -> np.ndarray:
    """Evaluate Jacobi polynomials J(0..N, alpha, beta, X).

    Uses the Burkardt three-term recurrence. Matches MATLAB j_polynomial.m.

    Parameters
    ----------
    m : int
        Number of evaluation points (len(x)).
    n : int
        Highest polynomial order to compute (returns orders 0..n).
    alpha : float
        First Jacobi parameter. Must be > -1.
    beta : float
        Second Jacobi parameter. Must be > -1.
    x : np.ndarray
        Evaluation points, shape (m,) or (m, 1).

    Returns
    -------
    np.ndarray
        Shape (m, n+1). Column j contains P_j(alpha, beta, x).

    Raises
    ------
    ValueError
        If alpha <= -1 or beta <= -1.
    """
    if alpha <= -1.0:
        raise ValueError(f"alpha must be > -1, got {alpha}")
    if beta <= -1.0:
        raise ValueError(f"beta must be > -1, got {beta}")

    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) != m:
        raise ValueError(f"x has {len(x)} elements but m={m}")

    if n < 0:
        return np.empty((m, 0), dtype=np.float64)

    v = np.zeros((m, n + 1), dtype=np.float64)
    v[:, 0] = 1.0

    if n == 0:
        return v

    # P_1(x) = (1 + (alpha+beta)/2)*x + (alpha-beta)/2
    v[:, 1] = (1.0 + 0.5 * (alpha + beta)) * x + 0.5 * (alpha - beta)

    for i in range(2, n + 1):
        c1 = 2 * i * (i + alpha + beta) * (2 * i - 2 + alpha + beta)
        c2 = (
            (2 * i - 1 + alpha + beta)
            * (2 * i + alpha + beta)
            * (2 * i - 2 + alpha + beta)
        )
        c3 = (2 * i - 1 + alpha + beta) * (alpha + beta) * (alpha - beta)
        c4 = -2 * (i - 1 + alpha) * (i - 1 + beta) * (2 * i + alpha + beta)

        v[:, i] = ((c3 + c2 * x) * v[:, i - 1] + c4 * v[:, i - 2]) / c1

    return v
