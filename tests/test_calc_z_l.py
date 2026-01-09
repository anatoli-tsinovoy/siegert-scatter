"""Tests for spherical Bessel zeros."""

import numpy as np
from numpy.testing import assert_allclose

from siegert_scatter import calc_z_l


def sort_roots(roots: list[complex]) -> list[complex]:
    """Sort roots canonically: by real part, then imaginary part."""
    return sorted(roots, key=lambda z: (z.real, z.imag))


class TestCalcZL:
    """Tests matching SPS_test_suite.m Test 3."""

    def test_ell_0_empty(self):
        """l=0 should return empty array."""
        z = calc_z_l(0, False)
        assert len(z) == 0

    def test_ell_1(self):
        """l=1: single real zero at -1."""
        z = calc_z_l(1, False)
        assert len(z) == 1
        assert_allclose(z[0], -1.0 + 0j, rtol=1e-14)

    def test_ell_2(self):
        """l=2: conjugate pair."""
        z = calc_z_l(2, False)
        assert len(z) == 2

        expected = [
            -1.5 - 0.8660254037844386j,
            -1.5 + 0.8660254037844386j,
        ]
        assert_allclose(sort_roots(z), sort_roots(expected), rtol=1e-14)

    def test_ell_3(self):
        """l=3: one real + conjugate pair."""
        z = calc_z_l(3, False)
        assert len(z) == 3

        expected = [
            -2.322185354626086 + 0j,
            -1.838907322686957 - 1.754380959783722j,
            -1.838907322686957 + 1.754380959783722j,
        ]
        assert_allclose(sort_roots(z), sort_roots(expected), rtol=1e-12)

    def test_ell_4(self):
        """l=4: two conjugate pairs."""
        z = calc_z_l(4, False)
        assert len(z) == 4

        expected = [
            -2.103789397179628 - 2.657418041856753j,
            -2.103789397179628 + 2.657418041856753j,
            -2.896210602820372 - 0.867234128934504j,
            -2.896210602820372 + 0.867234128934504j,
        ]
        assert_allclose(sort_roots(z), sort_roots(expected), rtol=1e-12)

    def test_ell_5(self):
        """l=5: one real + two conjugate pairs."""
        z = calc_z_l(5, False)
        assert len(z) == 5

        expected = [
            -3.646738595329643 + 0j,
            -3.351956399153533 - 1.742661416183198j,
            -3.351956399153533 + 1.742661416183198j,
            -2.324674303181645 - 3.571022920337976j,
            -2.324674303181645 + 3.571022920337976j,
        ]
        assert_allclose(sort_roots(z), sort_roots(expected), rtol=1e-12)

    def test_ell_60(self):
        """l=60: verify count and conjugate pair structure."""
        z = calc_z_l(60, False)
        assert len(z) == 60

        sorted_z = sort_roots(z)
        # All roots should have negative real parts
        assert all(root.real < 0 for root in sorted_z)
        # Even l: all roots come in conjugate pairs (no purely real roots)
        # Check conjugate symmetry
        for i in range(0, 60, 2):
            assert_allclose(sorted_z[i].real, sorted_z[i + 1].real, rtol=1e-10)
            assert_allclose(sorted_z[i].imag, -sorted_z[i + 1].imag, rtol=1e-10)

    def test_ell_150(self):
        """l=150: verify count and conjugate pair structure."""
        z = calc_z_l(150, False)
        assert len(z) == 150

        sorted_z = sort_roots(z)
        # All roots should have negative real parts
        assert all(root.real < 0 for root in sorted_z)
        # Even l: all roots come in conjugate pairs
        for i in range(0, 150, 2):
            assert_allclose(sorted_z[i].real, sorted_z[i + 1].real, rtol=1e-10)
            assert_allclose(sorted_z[i].imag, -sorted_z[i + 1].imag, rtol=1e-10)

    def test_count_matches_ell(self):
        """Number of zeros should equal ell."""
        for ell in range(0, 20):
            z = calc_z_l(ell, False)
            assert len(z) == ell

    def test_real_only_mode(self):
        """real_only returns unique real and |imag| parts."""
        # For l=2, zeros are -1.5 ± 0.866i
        z_real = calc_z_l(2, True)

        # Should contain -1.5 and 0.866 (unique real parts and |imag| parts)
        # But not 0 (excluded)
        assert -1.5 in z_real
        assert_allclose(
            0.8660254037844386 in z_real
            or np.any(np.isclose(z_real, 0.8660254037844386)),
            True,
        )

    def test_invalid_ell(self):
        """Should raise for ell out of range."""
        import pytest

        with pytest.raises(ValueError):
            calc_z_l(-1, False)
