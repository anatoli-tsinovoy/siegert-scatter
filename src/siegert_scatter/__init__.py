"""Siegert pseudostate scattering calculations.

This package provides tools for computing quantum scattering properties
using the Siegert pseudostate (SPS) method:

- S-matrices and scattering phases
- Cross sections (partial and total)
- Wigner time delays
- Scattering lengths
- Bound state energies

Example
-------
>>> import numpy as np
>>> from siegert_scatter import tise_by_sps, calc_cross_section_by_sps
>>>
>>> # Define a potential (Pöschl-Teller)
>>> def V(r):
...     return -10 / np.cosh(r)**2
>>>
>>> # Compute scattering
>>> E = np.linspace(0.01, 5, 100)
>>> result = calc_cross_section_by_sps(V, N=50, a=10, l_max=3, E_vec=E)
>>> print(f"Scattering length: {result.alpha:.4f}")
"""

from .bessel_zeros import calc_z_l
from .cli import main
from .cross_section import CrossSectionResult, calc_cross_section_by_sps
from .polynomials import j_polynomial
from .quadrature import get_gaussian_quadrature
from .tise import TISEResult, tise_by_sps

__all__ = [
    "calc_cross_section_by_sps",
    "calc_z_l",
    "CrossSectionResult",
    "get_gaussian_quadrature",
    "j_polynomial",
    "main",
    "tise_by_sps",
    "TISEResult",
]

__version__ = "0.1.0"
