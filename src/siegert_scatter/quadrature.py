"""Gauss-Jacobi quadrature (ports get_gaussian_quadrature.m).

This is a direct port of Burkardt/IQPACK Golub-Welsch implementation.
"""

import numpy as np
from scipy.special import gammaln


def get_gaussian_quadrature(
    order: int,
    alpha: float,
    beta: float,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gauss-Jacobi quadrature nodes and weights.

    Parameters
    ----------
    order : int
        Number of quadrature points.
    alpha : float
        Jacobi parameter alpha (weight (b-x)^alpha). Must be > -1.
    beta : float
        Jacobi parameter beta (weight (x-a)^beta). Must be > -1.
    a, b : float
        Integration interval [a, b].

    Returns
    -------
    omega : np.ndarray
        Weights, shape (order,).
    nodes : np.ndarray
        Nodes, shape (order,).
    """
    # Compute on default interval [-1, 1]
    nodes, weights = _cdgqf(order, alpha, beta)

    # Scale to [a, b]
    nodes, weights = _scqf(order, nodes, weights, alpha, beta, a, b)

    return weights, nodes


def _cdgqf(
    nt: int,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gauss quadrature on default interval [-1, 1]."""
    _parchk(alpha, beta)

    # Build Jacobi matrix and zero-th moment
    aj, bj, zemu, log2zemu = _class_matrix_jacobi(nt, alpha, beta)

    # Compute nodes and weights via Golub-Welsch
    nodes, weights = _sgqf(nt, aj, bj, zemu, log2zemu)

    return nodes, weights


def _class_matrix_jacobi(
    m: int,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build Jacobi matrix for Jacobi weight function (kind=4 in MATLAB)."""
    aj = np.zeros(m, dtype=np.float64)
    bj = np.zeros(m, dtype=np.float64)

    ab = alpha + beta
    abi = 2.0 + ab

    # log2(zemu) = (ab+1) + gammaln(alpha+1)/log(2) + gammaln(beta+1)/log(2) - gammaln(abi)/log(2)
    log2zemu = (
        (ab + 1.0)
        + gammaln(alpha + 1.0) / np.log(2)
        + gammaln(beta + 1.0) / np.log(2)
        - gammaln(abi) / np.log(2)
    )
    zemu = 2.0**log2zemu

    aj[0] = (beta - alpha) / abi
    bj[0] = 4.0 * (1.0 + alpha) * (1.0 + beta) / ((abi + 1.0) * abi * abi)
    a2b2 = beta * beta - alpha * alpha

    for i in range(1, m):
        abi_i = 2.0 * (i + 1) + ab  # i+1 because 0-indexed
        aj[i] = a2b2 / ((abi_i - 2.0) * abi_i)
        abi_sq = abi_i * abi_i
        bj[i] = (
            4.0
            * (i + 1)
            * (i + 1 + alpha)
            * (i + 1 + beta)
            * (i + 1 + ab)
            / ((abi_sq - 1.0) * abi_sq)
        )

    bj = np.sqrt(bj)

    return aj, bj, zemu, log2zemu


def _sgqf(
    nt: int,
    aj: np.ndarray,
    bj: np.ndarray,
    zemu: float,
    log2zemu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute nodes and weights from Jacobi matrix via Golub-Welsch."""
    if zemu <= 0.0:
        raise ValueError(f"ZEMU must be positive, got {zemu}")

    # Set up vector for IMTQLX
    wts = np.zeros(nt, dtype=np.float64)
    wts[0] = 2.0 ** (log2zemu / 2.0)

    # Diagonalize the Jacobi matrix
    nodes, wts = _imtqlx(nt, aj.copy(), bj.copy(), wts)

    wts = wts**2

    return nodes, wts


def _imtqlx(
    n: int,
    d: np.ndarray,
    e: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize symmetric tridiagonal matrix (implicit QL algorithm)."""
    itn = 30
    prec = np.finfo(np.float64).eps

    if n == 1:
        return d, z

    e[n - 1] = 0.0

    for ell in range(n):
        j = 0

        while True:
            # Find small subdiagonal element
            m = ell
            for m in range(ell, n):
                if m == n - 1:
                    break
                if abs(e[m]) <= prec * (abs(d[m]) + abs(d[m + 1])):
                    break

            p = d[ell]

            if m == ell:
                break

            if j >= itn:
                raise RuntimeError("IMTQLX: iteration limit exceeded")

            j += 1
            g = (d[ell + 1] - p) / (2.0 * e[ell])
            r = np.sqrt(g * g + 1.0)
            g = d[m] - p + e[ell] / (g + np.sign(g) * abs(r) if g != 0 else r)
            s = 1.0
            c = 1.0
            p = 0.0

            for ii in range(1, m - ell + 1):
                i = m - ii
                f = s * e[i]
                bb = c * e[i]

                if abs(f) >= abs(g):
                    c = g / f
                    r = np.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c = c * s
                else:
                    s = f / g
                    r = np.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s = s * c

                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * bb
                p = s * r
                d[i + 1] = g + p
                g = c * r - bb
                f = z[i + 1]
                z[i + 1] = s * z[i] + c * f
                z[i] = c * z[i] - s * f

            d[ell] = d[ell] - p
            e[ell] = g
            e[m] = 0.0

    # Sort eigenvalues ascending
    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]

        for j in range(ii, n):
            if d[j] < p:
                k = j
                p = d[j]

        if k != i:
            d[k] = d[i]
            d[i] = p
            p = z[i]
            z[i] = z[k]
            z[k] = p

    return d, z


def _scqf(
    nt: int,
    t: np.ndarray,
    wts: np.ndarray,
    alpha: float,
    beta: float,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale quadrature from [-1, 1] to [a, b]."""
    temp = np.finfo(np.float64).eps

    if abs(b - a) <= temp:
        raise ValueError(f"|b - a| too small: a={a}, b={b}")

    al = alpha
    be = beta
    shft = (a + b) / 2.0
    slp = (b - a) / 2.0

    # Scale
    t_out = shft + slp * t
    p = slp ** (al + be + 1.0)
    wts_out = wts * p

    return t_out, wts_out


def _parchk(alpha: float, beta: float) -> None:
    """Check Jacobi parameters."""
    if alpha <= -1.0:
        raise ValueError(f"alpha must be > -1, got {alpha}")
    if beta <= -1.0:
        raise ValueError(f"beta must be > -1, got {beta}")
