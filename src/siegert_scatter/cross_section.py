"""Cross section calculation via SPS (ports calc_cross_section_by_SPS.m)."""

from typing import Callable

import numpy as np

from .tise import TISEResult, tise_by_sps


class CrossSectionResult:
    """Results from cross section calculation.

    Attributes
    ----------
    S_l : np.ndarray
        S-matrix elements, shape (n_E, l_max+1).
    sigma_l : np.ndarray
        Partial cross sections, shape (n_E, l_max+1).
    tau_l : np.ndarray
        Time delays, shape (n_E, l_max+1).
    alpha : float
        Scattering length (l=0).
    k_n_l : list[np.ndarray]
        Eigenvalues k_n for each l.
    E_vec : np.ndarray
        Energy grid used for calculations.
    """

    def __init__(
        self,
        S_l: np.ndarray,
        sigma_l: np.ndarray,
        tau_l: np.ndarray,
        alpha: float,
        k_n_l: list[np.ndarray],
        E_vec: np.ndarray,
    ):
        self.S_l = S_l
        self.sigma_l = sigma_l
        self.tau_l = tau_l
        self.alpha = alpha
        self.k_n_l = k_n_l
        self.E_vec = E_vec


def calc_cross_section_by_sps(
    f_x: Callable[[np.ndarray], np.ndarray],
    N: int,
    a: float,
    l_max: int,
    E_vec: np.ndarray,
    dtau: float = np.inf,
    V_2: Callable[[np.ndarray], np.ndarray] | None = None,
    verbose: bool = False,
) -> CrossSectionResult:
    """Compute scattering cross sections using Siegert pseudostates.

    Parameters
    ----------
    f_x : callable
        Radial potential V(r) in atomic units.
    N : int
        Number of basis functions.
    a : float
        Potential cutoff radius (a.u.).
    l_max : int
        Maximum angular momentum.
    E_vec : np.ndarray
        Energies to compute scattering at (a.u.).
    dtau : float, default=inf
        Time delay parameter. If finite, poles are augmented with width.
    V_2 : callable, optional
        Perturbation potential (for transition calculations).
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    CrossSectionResult
        Container with S-matrix, cross sections, time delays, etc.
    """
    E_vec = np.atleast_1d(np.asarray(E_vec, dtype=np.float64))

    # Solve TISE for each l
    k_n_l: list[np.ndarray] = []
    tise_results: list[TISEResult] = []

    for ell in range(l_max + 1):
        if verbose:
            print(f"l = {ell}")
        result = tise_by_sps(f_x, N, a, ell, V_2)
        k_n_l.append(result.k_n)
        tise_results.append(result)

    # Compute scattering length from l=0 poles
    # alpha = real(a + sum(1i / k_n_l{1}))
    k_n_0 = k_n_l[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.real(a + np.sum(1j / k_n_0))

    # Compute k vector
    k_vec = np.sqrt(2 * E_vec)

    # Width parameter for pole augmentation
    dGamma = 1.2279e-13 / dtau if np.isfinite(dtau) else 0.0

    n_E = len(k_vec)
    S_l = np.zeros((n_E, l_max + 1), dtype=np.complex128)
    sigma_l = np.zeros((n_E, l_max + 1), dtype=np.float64)
    tau_l = np.zeros((n_E, l_max + 1), dtype=np.float64)

    for ell in range(l_max + 1):
        if verbose:
            print(f"Computing S-matrix for l = {ell}")

        k_n = k_n_l[ell]

        # Augment poles with decay width if dtau is finite
        if np.isfinite(dtau) and dGamma > 0:
            # k_n_aug = real(k_n) + 1i*(imag(k_n) - (real(k_n)!=0)*dGamma/(|k_n| + (real(k_n)==0)))
            real_mask = np.real(k_n) != 0
            denom = np.abs(k_n) + (~real_mask).astype(float)
            k_n_aug = np.real(k_n) + 1j * (np.imag(k_n) - real_mask * dGamma / denom)
        else:
            k_n_aug = k_n

        # S-matrix: S_l(k) = exp(-2i*k*a) * prod((k_n + k) / (k_n - k))
        S_l_k = np.exp(-2j * k_vec * a)

        # Time delay derivative: d_delta/dk
        d_delta_dk = -a * np.ones_like(k_vec)

        for k_pole in k_n_aug:
            # S-matrix product
            S_l_k = S_l_k * (k_pole + k_vec) / (k_pole - k_vec)

            # Time delay sum term
            im_k = np.imag(k_pole)
            re_k = np.real(k_pole)
            numerator = -im_k * (im_k**2 + k_vec**2 + re_k**2)
            denominator = (
                k_vec**4
                + 2 * k_vec**2 * (im_k - re_k) * (im_k + re_k)
                + (im_k**2 + re_k**2) ** 2
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                term = numerator / denominator
                term = np.where(np.isfinite(term), term, 0)
            d_delta_dk = d_delta_dk + term

        # Cross section: sigma_l = (2l+1) * (pi/k^2) * |1 - S_l|^2
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_l[:, ell] = (
                (2 * ell + 1) * (np.pi / k_vec**2) * np.abs(1 - S_l_k) ** 2
            )
            sigma_l[:, ell] = np.where(np.isfinite(sigma_l[:, ell]), sigma_l[:, ell], 0)

        # Time delay: tau_l = d_delta_dk / k
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_l[:, ell] = d_delta_dk / k_vec
            tau_l[:, ell] = np.where(np.isfinite(tau_l[:, ell]), tau_l[:, ell], 0)

        S_l[:, ell] = S_l_k

    return CrossSectionResult(
        S_l=S_l,
        sigma_l=sigma_l,
        tau_l=tau_l,
        alpha=alpha,
        k_n_l=k_n_l,
        E_vec=E_vec,
    )
