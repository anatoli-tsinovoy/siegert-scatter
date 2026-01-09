"""Gauss-Jacobi quadrature via scipy."""

import numpy as np
from scipy.special import roots_jacobi


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
    weights : np.ndarray
        Weights, shape (order,).
    nodes : np.ndarray
        Nodes, shape (order,).
    """
    # Get nodes and weights on [-1, 1]
    nodes, weights = roots_jacobi(order, alpha, beta)

    # Scale to [a, b]
    slp = (b - a) / 2.0
    shft = (a + b) / 2.0
    nodes = shft + slp * nodes
    weights = weights * slp ** (alpha + beta + 1.0)

    return weights, nodes
