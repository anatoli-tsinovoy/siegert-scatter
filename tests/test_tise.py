"""Tests for TISE solver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from siegert_scatter import tise_by_sps

# Tolerances for numerical comparisons
RTOL_LOOSE = 1e-4
ATOL_ENERGY = 1e-4


class TestTISE:
    """Tests matching SPS_test_suite.m Test 5."""

    @pytest.fixture
    def poschl_teller(self):
        """Pöschl-Teller potential with lambda=4."""

        def V(r):
            return -10 / np.cosh(r) ** 2

        return V

    def test_eigenvalue_count(self, poschl_teller):
        """Number of eigenvalues should be 2N + ell."""
        N, a, ell = 10, 10, 0
        result = tise_by_sps(poschl_teller, N, a, ell)
        assert len(result.k_n) == 2 * N + ell

        N, a, ell = 10, 10, 1
        result = tise_by_sps(poschl_teller, N, a, ell)
        assert len(result.k_n) == 2 * N + ell

    def test_bound_states_l0(self, poschl_teller):
        """Test 5.1/5.3: Bound states for radial l=0.

        For radial problem with psi(0)=0, only odd-parity states survive.
        Expected: E = -0.5 and E = -4.5.
        """
        result = tise_by_sps(poschl_teller, 30, 10, 0)

        # Find bound states: purely imaginary k with positive imag part
        k_n = result.k_n
        bound_mask = (np.abs(np.real(k_n)) < 1e-8) & (np.imag(k_n) > 0)
        k_bound = k_n[bound_mask]
        E_bound = -(np.abs(k_bound) ** 2) / 2
        E_bound = np.sort(E_bound)[::-1]

        assert len(E_bound) == 2
        assert_allclose(E_bound, [-0.5, -4.5], atol=ATOL_ENERGY)

    def test_bound_states_l1(self, poschl_teller):
        """Bound states for l=1."""
        result = tise_by_sps(poschl_teller, 30, 10, 1)

        k_n = result.k_n
        bound_mask = (np.abs(np.real(k_n)) < 1e-8) & (np.imag(k_n) > 0)
        k_bound = k_n[bound_mask]
        E_bound = -(np.abs(k_bound) ** 2) / 2

        # l=1 should have one bound state near E = -1.73
        assert len(E_bound) >= 1

    def test_quadrature_points(self, poschl_teller):
        """Quadrature points r_i should be in (0, a)."""
        N, a, ell = 30, 10, 0
        result = tise_by_sps(poschl_teller, N, a, ell)

        assert len(result.r_i) == N
        assert np.all(result.r_i > 0)
        assert np.all(result.r_i < a)

    def test_eigenvalues_sorted_example(self, poschl_teller):
        """Test eigenvalue structure matches MATLAB."""
        N, a, ell = 10, 10, 0
        result = tise_by_sps(poschl_teller, N, a, ell)

        # Sort by real part
        idx = np.argsort(np.real(result.k_n))
        k_sorted = result.k_n[idx]

        # First few should have negative real parts (resonances)
        # The bound states should have Re(k) ≈ 0 and Im(k) > 0
        bound_states = [k for k in k_sorted if np.abs(k.real) < 1e-6 and k.imag > 0]
        assert len(bound_states) == 2

    def test_convergence_with_N(self, poschl_teller):
        """Bound state energies should converge as N increases."""
        energies = []
        for N in [20, 40, 60]:
            result = tise_by_sps(poschl_teller, N, 10, 0)
            k_n = result.k_n
            bound_mask = (np.abs(np.real(k_n)) < 1e-8) & (np.imag(k_n) > 0)
            k_bound = k_n[bound_mask]
            E = -(np.abs(k_bound) ** 2) / 2
            energies.append(np.sort(E)[::-1])

        # Energy should get closer to exact values with larger N
        E_exact = np.array([-0.5, -4.5])
        assert np.max(np.abs(energies[-1] - E_exact)) < np.max(
            np.abs(energies[0] - E_exact)
        )


class TestTISEFullPotential:
    """Test with potential centered at a/2 (captures all bound states)."""

    def test_all_bound_states(self):
        """Full Pöschl-Teller should have 4 bound states."""
        a = 20

        def V(r):
            return -10 / np.cosh(r - a / 2) ** 2

        result = tise_by_sps(V, 100, a, 0)

        k_n = result.k_n
        bound_mask = (np.abs(np.real(k_n)) < 1e-6) & (np.imag(k_n) > 0.5)
        k_bound = k_n[bound_mask]
        E_bound = -(np.abs(k_bound) ** 2) / 2
        E_bound = np.sort(E_bound)[::-1]

        # Should find all 4: -0.5, -2, -4.5, -8
        assert len(E_bound) == 4
        assert_allclose(E_bound, [-0.5, -2, -4.5, -8], atol=0.01)
