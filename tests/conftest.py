"""Shared fixtures and utilities for tests."""

import numpy as np
import pytest


@pytest.fixture
def poschl_teller_potential():
    """Pöschl-Teller potential with lambda=4: V(r) = -10/cosh²(r)."""

    def V(r: np.ndarray) -> np.ndarray:
        return -10 / np.cosh(r) ** 2

    return V


@pytest.fixture
def poschl_teller_shifted(poschl_teller_potential):
    """Pöschl-Teller shifted to center at a/2."""
    a = 20

    def V(r: np.ndarray) -> np.ndarray:
        return -10 / np.cosh(r - a / 2) ** 2

    return V, a


# Tolerances for numerical comparisons
RTOL_LOOSE = 1e-4
RTOL_TIGHT = 1e-8
ATOL_ENERGY = 1e-4
ATOL_SMALL = 1e-10
