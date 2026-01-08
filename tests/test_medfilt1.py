"""Tests for median filter."""

import numpy as np
from numpy.testing import assert_allclose

from siegert_scatter import medfilt1


class TestMedfilt1:
    """Tests matching SPS_test_suite.m Test 4."""

    def test_window_3(self):
        """Test 4.1: medfilt1([1,5,2,8,3,7,4,6,9,0], 3)."""
        x = np.array([1, 5, 2, 8, 3, 7, 4, 6, 9, 0], dtype=float)
        y = medfilt1(x, 3)

        # From MATLAB reference
        expected = np.array([3, 2, 5, 3, 7, 4, 6, 6, 6, 4.5])
        assert_allclose(y, expected)

    def test_window_5(self):
        """Test 4.2: medfilt1(..., 5)."""
        x = np.array([1, 5, 2, 8, 3, 7, 4, 6, 9, 0], dtype=float)
        y = medfilt1(x, 5)

        # From MATLAB reference
        expected = np.array([2, 3.5, 3, 5, 4, 6, 6, 6, 5, 6])
        assert_allclose(y, expected)

    def test_identity_window_1(self):
        """Window size 1 should return input unchanged."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = medfilt1(x, 1)
        assert_allclose(y, x)

    def test_constant_signal(self):
        """Constant signal should be unchanged."""
        x = np.array([5, 5, 5, 5, 5], dtype=float)
        y = medfilt1(x, 3)
        assert_allclose(y, x)

    def test_edge_handling(self):
        """Edges use truncated windows."""
        # At edges, window shrinks: first element uses [x[0], x[1]] for window 3
        x = np.array([1, 10, 2], dtype=float)
        y = medfilt1(x, 3)

        # First: median([1, 10]) = 5.5
        # Middle: median([1, 10, 2]) = 2
        # Last: median([10, 2]) = 6
        expected = np.array([5.5, 2, 6])
        assert_allclose(y, expected)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        x = np.array([1, 2, 3, 4, 5])
        y = medfilt1(x, 3)
        assert y.shape == x.shape
