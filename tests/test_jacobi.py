"""Tests for Jacobi polynomial evaluation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from siegert_scatter import j_polynomial


class TestJPolynomial:
    """Tests matching SPS_test_suite.m Test 1."""

    def test_shape(self):
        """Output has correct shape (m, n+1)."""
        x = np.array([-1, -0.5, 0, 0.5, 1])
        JP = j_polynomial(5, 3, 0, 4, x)
        assert JP.shape == (5, 4)

    def test_P0_is_ones(self):
        """P_0 should always be 1."""
        x = np.array([-1, -0.5, 0, 0.5, 1])
        JP = j_polynomial(5, 5, 0, 4, x)
        assert_allclose(JP[:, 0], 1.0)

    def test_alpha0_beta4_values(self):
        """Test 1.1: j_polynomial(5, 5, 0, 4, [-1,-0.5,0,0.5,1])."""
        x = np.array([-1, -0.5, 0, 0.5, 1])
        JP = j_polynomial(5, 5, 0, 4, x)

        # From MATLAB reference output
        expected_P1 = [-5.0, -3.5, -2.0, -0.5, 1.0]
        expected_P2 = [15.0, 6.25, 1.0, -0.75, 1.0]

        assert_allclose(JP[:, 1], expected_P1, rtol=1e-10)
        assert_allclose(JP[:, 2], expected_P2, rtol=1e-10)

    def test_alpha0_beta0_legendre(self):
        """Test 1.2: Legendre polynomials (alpha=beta=0)."""
        x = np.array([-0.9, -0.425, 0.05, 0.525, 1.0])
        JP = j_polynomial(5, 3, 0, 0, x)

        # P_1(x) = x for Legendre
        assert_allclose(JP[:, 1], x, rtol=1e-10)

        # P_2(x) = (3x^2 - 1)/2
        expected_P2 = (3 * x**2 - 1) / 2
        assert_allclose(JP[:, 2], expected_P2, rtol=1e-10)

    def test_at_x_equals_1(self):
        """Special value: P_n^(0,0)(1) = 1 for all n."""
        x = np.array([1.0])
        JP = j_polynomial(1, 10, 0, 0, x)
        assert_allclose(JP[0, :], 1.0, rtol=1e-10)

    def test_invalid_alpha(self):
        """Should raise for alpha <= -1."""
        with pytest.raises(ValueError, match="alpha"):
            j_polynomial(3, 2, -1.5, 0, np.array([0]))

    def test_invalid_beta(self):
        """Should raise for beta <= -1."""
        with pytest.raises(ValueError, match="beta"):
            j_polynomial(3, 2, 0, -1.5, np.array([0]))

    def test_n_negative_returns_empty(self):
        """n < 0 returns empty array."""
        result = j_polynomial(3, -1, 0, 0, np.array([0, 0.5, 1]))
        assert result.shape == (3, 0)
