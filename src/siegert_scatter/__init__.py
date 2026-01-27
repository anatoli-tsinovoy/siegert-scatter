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
>>> from siegert_scatter import tise_by_sps, calc_cross_section
>>>
>>> # Define a potential (Pöschl-Teller)
>>> def V(r):
...     return -10 / np.cosh(r)**2
>>>
>>> # Compute scattering
>>> E = np.linspace(0.01, 5, 100)
>>> result = calc_cross_section(V, N=50, a=10, l_max=3, E_vec=E)
>>> print(f"Scattering length: {result.alpha:.4f}")
"""

from .bessel_zeros import calc_z_l
from .cli import main
from .cross_section import (
    CrossSectionResult,
    augment_poles,
    calc_cross_section,
    s_matrix_from_poles,
    s_matrix_from_poles_all_l,
)
from .perturbation import PerturbationResult, calc_perturbation
from .polynomials import j_polynomial
from .quadrature import get_gaussian_quadrature
from .tise import TISEResult, calc_eigenmodes, tise_by_sps
from .units import reduced_mass, tau_to_dGamma

__all__ = [
    "augment_poles",
    "calc_cross_section",
    "calc_eigenmodes",
    "calc_perturbation",
    "calc_z_l",
    "CrossSectionResult",
    "get_gaussian_quadrature",
    "j_polynomial",
    "main",
    "PerturbationResult",
    "reduced_mass",
    "s_matrix_from_poles",
    "s_matrix_from_poles_all_l",
    "tau_to_dGamma",
    "tise_by_sps",
    "TISEResult",
]

__version__ = "0.1.0"
