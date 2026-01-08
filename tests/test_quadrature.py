"""Tests for Gauss-Jacobi quadrature."""

import numpy as np
from numpy.testing import assert_allclose

from siegert_scatter import get_gaussian_quadrature


class TestGaussianQuadrature:
    """Tests matching SPS_test_suite.m Test 2."""

    def test_gauss_legendre_n5(self):
        """Test 2.1: Gauss-Legendre N=5, alpha=0, beta=0."""
        omega, nodes = get_gaussian_quadrature(5, 0, 0, -1, 1)

        # From MATLAB reference
        expected_nodes = np.array(
            [
                -9.061798459386641e-01,
                -5.384693101056830e-01,
                0.0,  # ~1e-16 in MATLAB
                5.384693101056831e-01,
                9.061798459386639e-01,
            ]
        )
        expected_weights = np.array(
            [
                2.369268850561892e-01,
                4.786286704993669e-01,
                5.688888888888890e-01,
                4.786286704993672e-01,
                2.369268850561891e-01,
            ]
        )

        assert_allclose(nodes, expected_nodes, rtol=1e-10, atol=1e-15)
        assert_allclose(omega, expected_weights, rtol=1e-10)

    def test_gauss_jacobi_n5_beta4(self):
        """Test 2.2: Gauss-Jacobi N=5, alpha=0, beta=4."""
        omega, nodes = get_gaussian_quadrature(5, 0, 4, -1, 1)

        # From MATLAB reference
        expected_nodes = np.array(
            [
                -5.204159103950379e-01,
                -7.813265093696906e-02,
                3.601184654820191e-01,
                7.217726873528852e-01,
                9.452288370685309e-01,
            ]
        )
        expected_weights = np.array(
            [
                2.230326649360297e-02,
                3.267373352151109e-01,
                1.408783024172255e00,
                2.646806819262074e00,
                1.995369554856965e00,
            ]
        )

        assert_allclose(nodes, expected_nodes, rtol=1e-10)
        assert_allclose(omega, expected_weights, rtol=1e-10)

    def test_gauss_legendre_n20(self):
        """Test 2.3: Larger quadrature N=20."""
        omega, nodes = get_gaussian_quadrature(20, 0, 0, -1, 1)

        # From MATLAB: first and last nodes
        expected_first_nodes = np.array(
            [
                -9.931285991850950e-01,
                -9.639719272779136e-01,
                -9.122344282513257e-01,
                -8.391169718222189e-01,
                -7.463319064601508e-01,
            ]
        )
        expected_first_weights = np.array(
            [
                1.761400713915209e-02,
                4.060142980038740e-02,
                6.267204833410854e-02,
                8.327674157670478e-02,
                1.019301198172402e-01,
            ]
        )

        assert_allclose(nodes[:5], expected_first_nodes, rtol=1e-10)
        assert_allclose(omega[:5], expected_first_weights, rtol=1e-10)

    def test_weights_sum(self):
        """Gauss-Legendre weights should sum to 2 (length of [-1,1])."""
        omega, _ = get_gaussian_quadrature(10, 0, 0, -1, 1)
        assert_allclose(np.sum(omega), 2.0, rtol=1e-10)

    def test_nodes_symmetric(self):
        """Gauss-Legendre nodes are symmetric about 0."""
        _, nodes = get_gaussian_quadrature(7, 0, 0, -1, 1)
        assert_allclose(nodes + nodes[::-1], 0.0, atol=1e-14)

    def test_interval_scaling(self):
        """Nodes should scale correctly to [a, b]."""
        _, nodes_default = get_gaussian_quadrature(5, 0, 0, -1, 1)
        _, nodes_scaled = get_gaussian_quadrature(5, 0, 0, 0, 2)

        # [0, 2] = ([-1, 1] + 1) -> nodes should shift by 1
        assert_allclose(nodes_scaled, nodes_default + 1, rtol=1e-10)
