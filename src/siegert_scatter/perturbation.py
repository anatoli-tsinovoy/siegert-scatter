"""First-order perturbation theory for SPS (ports V_2 pathway from calc_cross_section_by_SPS.m).

This module implements the perturbation theory calculation for transition cross sections,
completely decoupled from the elastic scattering calculation in cross_section.py.
"""

from typing import Callable

import numpy as np

from .tise import tise_by_sps


def calc_e_l_z(z: np.ndarray, z_l_p: np.ndarray, ell: int) -> np.ndarray:
    """Calculate the asymptotic spherical Bessel factor e_l(z).

    This constructs the function:
        e_l(z) = P_l(-iz) / (-iz)^l * exp(iz)

    where P_l is the polynomial with roots at z_l_p (the spherical Bessel zeros).

    Parameters
    ----------
    z : np.ndarray
        Points at which to evaluate (typically k*a).
    z_l_p : np.ndarray
        Spherical Bessel zeros for angular momentum l.
    ell : int
        Angular momentum quantum number.

    Returns
    -------
    np.ndarray
        Values of e_l(z) at the given points.
    """
    z = np.atleast_1d(z)

    if len(z_l_p) == 0:
        # For l=0, no zeros, polynomial is just 1
        return np.exp(1j * z)

    # poly() gives coefficients of polynomial with given roots
    # polyval evaluates it
    coeffs = np.poly(z_l_p)
    poly_val = np.polyval(coeffs, -1j * z)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = (poly_val / (-1j * z) ** ell) * np.exp(1j * z)
        # Handle z=0 case
        result = np.where(np.isfinite(result), result, 0.0)

    return result


def calc_mittag_leffler_coeffs(
    k: np.ndarray,
    k_n: np.ndarray,
    z_l_p: np.ndarray,
    ell: int,
    a: float,
) -> np.ndarray:
    """Calculate Mittag-Leffler expansion coefficients M(k).

    M_n(k) = -ik / (e_l(ka) * k_n * (k_n - k))

    Parameters
    ----------
    k : np.ndarray
        Wavenumbers, shape (n_k,).
    k_n : np.ndarray
        Pole positions (possibly augmented), shape (n_poles,).
    z_l_p : np.ndarray
        Spherical Bessel zeros for this l.
    ell : int
        Angular momentum.
    a : float
        Cutoff radius.

    Returns
    -------
    np.ndarray
        Coefficients M_n(k), shape (n_k, n_poles).
    """
    k = np.atleast_1d(k)
    n_k = len(k)
    n_poles = len(k_n)

    # e_l(k*a) for all k values
    e_l_ka = calc_e_l_z(k * a, z_l_p, ell)  # shape (n_k,)

    # M_n(k) = -ik / (e_l(ka) * k_n * (k_n - k))
    # Shape: (n_k, n_poles)
    M = np.zeros((n_k, n_poles), dtype=np.complex128)

    for i, ki in enumerate(k):
        with np.errstate(divide="ignore", invalid="ignore"):
            M[i, :] = (-1j * ki / e_l_ka[i]) / (k_n * (k_n - ki))
            M[i, :] = np.where(np.isfinite(M[i, :]), M[i, :], 0.0)

    return M


def augment_poles(k_n: np.ndarray, dGamma: float) -> np.ndarray:
    """Augment poles with decay width.

    k_n_aug = Re(k_n) + i*(Im(k_n) - dGamma/|k_n|)  for Re(k_n) != 0

    Parameters
    ----------
    k_n : np.ndarray
        Original pole positions.
    dGamma : float
        Width parameter (natural energy units). Actual k-shift is dGamma/|k_n|.

    Returns
    -------
    np.ndarray
        Augmented poles.
    """
    if dGamma == 0:
        return k_n.copy()

    real_mask = np.real(k_n) != 0
    denom = np.abs(k_n) + (~real_mask).astype(float)
    return np.real(k_n) + 1j * (np.imag(k_n) - real_mask * dGamma / denom)


def calc_T_matrix_element(
    k: float,
    k_prime: float,
    M_k: np.ndarray,
    M_k_prime: np.ndarray,
    Cf: np.ndarray,
    CVC: np.ndarray,
) -> complex:
    """Calculate a single T-matrix element for the perturbation.

    T = (-1/k) * (M(k)·Cf)^T @ CVC @ (M(k')·Cf)

    Parameters
    ----------
    k : float
        Incoming wavenumber.
    k_prime : float
        Outgoing wavenumber (may differ due to threshold B).
    M_k : np.ndarray
        Mittag-Leffler coefficients at k, shape (n_poles,).
    M_k_prime : np.ndarray
        Mittag-Leffler coefficients at k', shape (n_poles,).
    Cf : np.ndarray
        C @ f_i_a product, shape (n_poles,).
    CVC : np.ndarray
        C @ V2_tilde @ C.T matrix, shape (n_poles, n_poles).

    Returns
    -------
    complex
        T-matrix element.
    """
    MCf_k = M_k * Cf  # element-wise, shape (n_poles,)
    MCf_k_prime = M_k_prime * Cf

    with np.errstate(divide="ignore", invalid="ignore"):
        T = (-1.0 / k) * (MCf_k @ CVC @ MCf_k_prime)
        if not np.isfinite(T):
            T = 0.0

    return T


class PerturbationResult:
    """Results from perturbation theory cross section calculation.

    Attributes
    ----------
    B_vals : np.ndarray
        Threshold/field values used in the sweep.
    k_vecs : list[np.ndarray]
        Wavenumber grids for each B value.
    E_vecs : list[np.ndarray]
        Energy grids for each B value.
    T_l : list[np.ndarray]
        T-matrix elements for each B, shape (n_k, l_max+1).
    sigma : list[np.ndarray]
        Cross sections for each B value, shape (n_k,).
    """

    def __init__(
        self,
        B_vals: np.ndarray,
        k_vecs: list[np.ndarray],
        T_l: list[np.ndarray],
        sigma: list[np.ndarray],
        l_max: int,
    ):
        self.B_vals = B_vals
        self.k_vecs = k_vecs
        self.E_vecs = [k**2 / 2 for k in k_vecs]
        self.T_l = T_l
        self.sigma = sigma
        self.l_max = l_max


def calc_perturbation(
    f_x: Callable[[np.ndarray], np.ndarray],
    V_2: Callable[[np.ndarray], np.ndarray],
    N: int,
    a: float,
    l_max: int,
    E_vec: np.ndarray,
    B_vals: np.ndarray | None = None,
    dGamma: float = 0.0,
    a0_sq: float = 1.0,
    verbose: bool = False,
) -> PerturbationResult:
    """Calculate perturbation theory cross sections.

    This implements the V_2 pathway from MATLAB's calc_cross_section_by_SPS.m,
    computing transition cross sections due to a perturbation potential V_2.

    Parameters
    ----------
    f_x : callable
        Base radial potential V(r) in atomic units.
    V_2 : callable
        Perturbation potential V_2(r) in atomic units.
    N : int
        Number of basis functions.
    a : float
        Potential cutoff radius (a.u.).
    l_max : int
        Maximum angular momentum.
    E_vec : np.ndarray
        Energies to compute at (a.u.).
    B_vals : np.ndarray, optional
        Threshold/field values to sweep. If None, uses MATLAB default:
        [-logspace(-1,-5,50), 0, logspace(-5,-1,50)].
    dGamma : float, default=0.0
        Width parameter for pole augmentation (natural energy units).
        Poles are shifted in imaginary direction by dGamma/|k_n|.
        Caller converts from lifetime: dGamma = t_au * (mu/m_e) / tau_seconds.
    a0_sq : float, default=1.0
        Bohr radius squared for unit conversion in output (pi*a0^2/k^2 prefactor).
        Set to 1.0 for atomic units, or (0.529e-8)**2 for cm^2.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    PerturbationResult
        Container with T-matrices and cross sections for each B value.
    """
    E_vec = np.atleast_1d(np.asarray(E_vec, dtype=np.float64))
    k_vec_base = np.sqrt(2 * E_vec)

    # Default B_vals from MATLAB
    if B_vals is None:
        B_vals = np.concatenate(
            [
                -np.flip(np.logspace(-5, -1, 50)),
                [0.0],
                np.logspace(-5, -1, 50),
            ]
        )
    B_vals = np.atleast_1d(np.asarray(B_vals, dtype=np.float64))

    # Solve TISE and compute PT matrices for each l
    if verbose:
        print("Solving TISE for each l...")

    k_n_l: list[np.ndarray] = []
    z_l_p_l: list[np.ndarray] = []
    CVC_l: list[np.ndarray] = []
    Cf_l: list[np.ndarray] = []

    for ell in range(l_max + 1):
        if verbose:
            print(f"  l = {ell}")

        result = tise_by_sps(f_x, N, a, ell)

        # Compute PT matrices: C, CVC, Cf
        C = result.c_ctilde_zeta[: result.N, :].T  # (n_states, N)
        V2_tilde = np.diag(V_2(result.r_i))
        CVC = C @ V2_tilde @ C.T
        Cf = C @ result.f_i_a

        k_n_l.append(result.k_n)
        z_l_p_l.append(result.z_l_p)
        CVC_l.append(CVC)
        Cf_l.append(Cf)

    # Sweep over B values
    if verbose:
        print("Computing T-matrices for each B...")

    result_k_vecs: list[np.ndarray] = []
    result_T_l: list[np.ndarray] = []
    result_sigma: list[np.ndarray] = []

    for B_ind, B in enumerate(B_vals):
        if verbose and B_ind % 20 == 0:
            print(f"  B[{B_ind}] = {B:.2e}")

        # Build k-vector: union of k and sqrt(2*(E - B)) where real
        k_shifted = np.sqrt(2 * (k_vec_base**2 / 2 - B))
        # Keep only real values
        real_mask = np.imag(k_shifted) == 0
        k_shifted_real = np.real(k_shifted[real_mask])

        k_vec = np.unique(np.concatenate([k_vec_base, k_shifted_real]))
        k_vec = k_vec[k_vec > 0]  # positive k only

        n_k = len(k_vec)
        T_l = np.zeros((n_k, l_max + 1), dtype=np.complex128)

        for ell in range(l_max + 1):
            # Augment poles
            k_n_aug = augment_poles(k_n_l[ell], dGamma)

            # Precompute M coefficients for all k in k_vec
            M_all = calc_mittag_leffler_coeffs(k_vec, k_n_aug, z_l_p_l[ell], ell, a)

            for i, k in enumerate(k_vec):
                # Outgoing energy: E' = E + B = k^2/2 + B
                E_prime = k**2 / 2 + B
                if E_prime <= 0:
                    continue

                k_prime = np.sqrt(2 * E_prime)

                # Get M coefficients
                M_k = M_all[i, :]

                # M at k_prime (need to compute separately if k_prime not in k_vec)
                M_k_prime = calc_mittag_leffler_coeffs(
                    np.array([k_prime]), k_n_aug, z_l_p_l[ell], ell, a
                )[0, :]

                T_l[i, ell] = calc_T_matrix_element(
                    k, k_prime, M_k, M_k_prime, Cf_l[ell], CVC_l[ell]
                )

        # Cross section: sigma = (pi * a0^2 / k^2) * sum_l (2l+1) |T_l|^2
        ell_weights = 2 * np.arange(l_max + 1) + 1  # (2l+1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = (np.pi * a0_sq / k_vec**2) * np.sum(
                ell_weights * np.abs(T_l) ** 2, axis=1
            )
            sigma = np.where(np.isfinite(sigma), sigma, 0.0)

        result_k_vecs.append(k_vec)
        result_T_l.append(T_l)
        result_sigma.append(sigma)

    return PerturbationResult(
        B_vals=B_vals,
        k_vecs=result_k_vecs,
        T_l=result_T_l,
        sigma=result_sigma,
        l_max=l_max,
    )
