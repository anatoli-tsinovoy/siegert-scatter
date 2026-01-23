"""Tests for cross section calculations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from siegert_scatter import calc_cross_section

RTOL_LOOSE = 1e-4


def poschl_teller_potential(r):
    """Pöschl-Teller potential with lambda=4."""
    return -10 / np.cosh(r) ** 2


def poschl_teller_base(x):
    LAMBDA_PT = 4
    return -LAMBDA_PT * (LAMBDA_PT + 1) / 2 / np.cosh(x) ** 2


def poschl_teller_v1(x):
    return poschl_teller_base(x) - 1e-6 * (1 / 4) / np.cosh(x)


def poschl_teller_v2(x):
    return poschl_teller_base(x) + 1e-6 * (3 / 4) / np.cosh(x)


class TestCrossSection:
    """Tests matching SPS_test_suite.m Test 6-7."""

    def test_scattering_length(self):
        """Test scattering length calculation."""
        result = calc_cross_section(
            poschl_teller_potential,
            N=50,
            a=10,
            l_max=0,
            E_vec=np.array([0.01]),
            adaptive_grid=False,
        )
        assert_allclose(result.alpha, 2.0833, rtol=0.01)

    def test_s_matrix_unitarity(self):
        """S-matrix should be unitary: |S|² = 1."""
        E_vec = np.linspace(0.1, 5, 50)
        result = calc_cross_section(
            poschl_teller_potential,
            N=50,
            a=10,
            l_max=3,
            E_vec=E_vec,
            adaptive_grid=False,
        )
        for ell in range(4):
            S_mag_sq = np.abs(result.S_l[:, ell]) ** 2
            assert_allclose(S_mag_sq, 1.0, rtol=1e-8)

    def test_cross_section_positive(self):
        """Cross sections should be non-negative."""
        E_vec = np.linspace(0.1, 5, 50)
        result = calc_cross_section(
            poschl_teller_potential,
            N=20,
            a=10,
            l_max=3,
            E_vec=E_vec,
            adaptive_grid=False,
        )
        assert np.all(result.sigma_l >= 0)

    def test_k_n_l_structure(self):
        """k_n_l should have one array per l."""
        result = calc_cross_section(
            poschl_teller_potential,
            N=20,
            a=10,
            l_max=3,
            E_vec=np.array([1.0]),
            adaptive_grid=False,
        )
        assert len(result.k_n_l) == 4

    def test_s_matrix_at_low_energy(self):
        """At very low energy, S_0 should be close to 1."""
        E_vec = np.array([1e-4])
        result = calc_cross_section(
            poschl_teller_potential,
            N=50,
            a=10,
            l_max=0,
            E_vec=E_vec,
            adaptive_grid=False,
        )
        assert_allclose(np.abs(result.S_l[0, 0]), 1.0, rtol=1e-8)

    def test_partial_wave_ordering(self):
        """Higher l waves should contribute less at low energy."""
        E_vec = np.array([0.01])
        result = calc_cross_section(
            poschl_teller_potential,
            N=30,
            a=10,
            l_max=4,
            E_vec=E_vec,
            adaptive_grid=False,
        )
        sigmas = result.sigma_l[0, :]
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1]


class TestCrossSectionSuite:
    """Tests matching SPS_test_suite.m Test 7 (full calculation)."""

    def test_two_potential_comparison(self):
        """Compare slightly different potentials."""
        E_vec = np.linspace(1e-6, 10, 100)

        result_1 = calc_cross_section(
            poschl_teller_v1, N=50, a=10, l_max=5, E_vec=E_vec, adaptive_grid=False
        )
        result_2 = calc_cross_section(
            poschl_teller_v2, N=50, a=10, l_max=5, E_vec=E_vec, adaptive_grid=False
        )

        assert_allclose(result_1.alpha, result_2.alpha, rtol=0.001)
        assert_allclose(np.abs(result_1.S_l[:, 0]) ** 2, 1.0, rtol=1e-8)
        assert_allclose(np.abs(result_2.S_l[:, 0]) ** 2, 1.0, rtol=1e-8)

    def test_energy_grid_shape(self):
        """Output arrays should match energy grid size."""
        E_vec = np.linspace(0.1, 5, 73)
        result = calc_cross_section(
            poschl_teller_potential,
            N=20,
            a=10,
            l_max=3,
            E_vec=E_vec,
            adaptive_grid=False,
        )

        assert result.S_l.shape == (73, 4)
        assert result.sigma_l.shape == (73, 4)
        assert result.tau_l.shape == (73, 4)
        assert len(result.E_vec) == 73
        assert len(result.E_vec_input) == 73


class TestAdaptiveGrid:
    """Tests for resonance-adaptive energy grid feature."""

    def test_adaptive_grid_default_is_true(self):
        """Adaptive grid is on by default."""
        E_vec = np.linspace(0.1, 5, 10)
        result = calc_cross_section(
            poschl_teller_potential, N=30, a=10, l_max=2, E_vec=E_vec
        )

        assert len(result.E_vec) >= len(E_vec)
        assert len(result.E_vec_input) == 10
        assert_allclose(result.E_vec_input, E_vec)
        for E in E_vec:
            assert np.any(np.isclose(result.E_vec, E, rtol=1e-10))

    def test_adaptive_grid_respects_bounds(self):
        """Adaptive grid stays within original energy bounds."""
        E_vec = np.linspace(0.5, 3.0, 10)
        result = calc_cross_section(
            poschl_teller_potential, N=30, a=10, l_max=2, E_vec=E_vec
        )

        k_min = np.sqrt(2 * E_vec.min())
        k_max = np.sqrt(2 * E_vec.max())
        k_vec_out = np.sqrt(2 * result.E_vec)

        assert k_vec_out.min() >= k_min - 1e-10
        assert k_vec_out.max() <= k_max + 1e-10

    def test_adaptive_grid_false_preserves_input(self):
        """adaptive_grid=False should use exact input grid."""
        E_vec = np.linspace(0.1, 5, 73)
        result = calc_cross_section(
            poschl_teller_potential,
            N=20,
            a=10,
            l_max=3,
            E_vec=E_vec,
            adaptive_grid=False,
        )

        assert len(result.E_vec) == 73
        assert_allclose(result.E_vec, E_vec)
        assert_allclose(result.E_vec_input, E_vec)

    def test_adaptive_grid_array_shapes_consistent(self):
        """Output arrays should match the (adaptive) E_vec length."""
        E_vec = np.linspace(0.1, 5, 10)
        result = calc_cross_section(
            poschl_teller_potential, N=30, a=10, l_max=2, E_vec=E_vec
        )

        n_E = len(result.E_vec)
        assert result.S_l.shape == (n_E, 3)
        assert result.sigma_l.shape == (n_E, 3)
        assert result.tau_l.shape == (n_E, 3)

    def test_adaptive_grid_s_matrix_still_unitary(self):
        """S-matrix should be unitary even with adaptive grid."""
        E_vec = np.linspace(0.1, 5, 10)
        result = calc_cross_section(
            poschl_teller_potential, N=50, a=10, l_max=2, E_vec=E_vec
        )

        for ell in range(3):
            S_mag_sq = np.abs(result.S_l[:, ell]) ** 2
            assert_allclose(S_mag_sq, 1.0, rtol=1e-8)
