"""Time-independent Schrödinger equation solver via SPS (ports TISE_by_SPS.m)."""

from typing import Callable

import numpy as np
from scipy import linalg

from .bessel_zeros import calc_z_l
from .polynomials import j_polynomial
from .quadrature import get_gaussian_quadrature


class TISEResult:
    """Results from TISE_by_SPS solver.

    Attributes
    ----------
    k_n : np.ndarray
        Complex wavenumbers (shape (n_states,)).
    c_ctilde_zeta : np.ndarray
        Eigenvectors in extended basis.
    eigenmodes : np.ndarray
        Eigenmodes at quadrature points r_i.
    r_i : np.ndarray
        Quadrature points (radial coordinates).
    z_l_p : np.ndarray
        Spherical Bessel zeros used.
    CVC : np.ndarray | None
        C @ V2_tilde @ C.T (if V_2 provided).
    Cf : np.ndarray | None
        C @ f (if V_2 provided).
    f : np.ndarray | None
        Boundary function values (if V_2 provided).
    C : np.ndarray | None
        Normalized coefficient matrix (if V_2 provided).
    V2_tilde : np.ndarray | None
        Perturbation potential matrix (if V_2 provided).
    """

    def __init__(
        self,
        k_n: np.ndarray,
        c_ctilde_zeta: np.ndarray,
        eigenmodes: np.ndarray,
        r_i: np.ndarray,
        z_l_p: np.ndarray,
        CVC: np.ndarray | None = None,
        Cf: np.ndarray | None = None,
        f: np.ndarray | None = None,
        C: np.ndarray | None = None,
        V2_tilde: np.ndarray | None = None,
    ):
        self.k_n = k_n
        self.c_ctilde_zeta = c_ctilde_zeta
        self.eigenmodes = eigenmodes
        self.r_i = r_i
        self.z_l_p = z_l_p
        self.CVC = CVC
        self.Cf = Cf
        self.f = f
        self.C = C
        self.V2_tilde = V2_tilde


def tise_by_sps(
    V: Callable[[np.ndarray], np.ndarray],
    N: int,
    a: float,
    ell: int,
    V_2: Callable[[np.ndarray], np.ndarray] | None = None,
) -> TISEResult:
    """Solve the radial TISE using Siegert pseudostates.

    Parameters
    ----------
    V : callable
        Potential function V(r), returns values at radial points.
    N : int
        Number of basis functions.
    a : float
        Cutoff radius (potential assumed zero for r > a).
    ell : int
        Angular momentum quantum number.
    V_2 : callable, optional
        Perturbation potential for computing transition matrix elements.

    Returns
    -------
    TISEResult
        Container with eigenvalues, eigenvectors, and auxiliary data.
    """

    # Effective potential including centrifugal term
    def V_eff(z: np.ndarray) -> np.ndarray:
        return V(z) + ell * (ell + 1) / (2 * z**2)

    # Jacobi polynomial normalization
    def jacobi_normalization(n: np.ndarray) -> np.ndarray:
        return 1.0 / np.sqrt(2 ** (1 + 2 * ell) / (1 + 2 * n + 2 * ell))

    # Get quadrature points and weights
    omega_i, x_i = get_gaussian_quadrature(N, 0, 2 * ell, -1, 1)

    # Evaluate Jacobi polynomials at quadrature points and at x=1
    x_extended = np.concatenate([x_i, [1.0]])
    JP_xi_n = j_polynomial(N + 1, N - 1, 0, 2 * ell, x_extended)

    # Basis functions at quadrature points: phi_n(x_i)
    n_vec = np.arange(N)
    norm = jacobi_normalization(n_vec)  # shape (N,)

    # phi_n_xi has shape (N orders, N points)
    # JP_xi_n[:N, :] is (N points × N orders)
    # Multiply (1+x)^l column-wise, then norm row-wise, then transpose
    JP_at_xi = JP_xi_n[:N, :]  # (N points × N orders)
    factor_1 = ((1 + x_i) ** ell)[:, None]  # (N, 1)
    factor_2 = factor_1 * JP_at_xi  # (N points × N orders)
    factor_3 = norm[None, :] * factor_2  # (N points × N orders)
    phi_n_xi = factor_3.T  # (N orders × N points)
    phi_n_1 = norm * (2**ell) * JP_xi_n[N, :]  # at x=1, shape (N,)

    # T_n_i matrix (transformation)
    T_n_i = phi_n_xi * (np.sqrt(omega_i) / (1 + x_i) ** ell)[None, :]

    # Construct matrices H_tilde_l and F
    pi_i_1 = T_n_i.T @ phi_n_1  # shape (N,)
    f_i_a = np.sqrt(2 / a) * (2 / (1 + x_i)) * pi_i_1  # shape (N,)
    F = np.outer(f_i_a, f_i_a)

    # Potential matrix
    r_i = (a / 2) * (x_i + 1)
    U_tilde_l = np.diag(V_eff(r_i))

    # Kinetic energy matrix K_tilde_l
    K_tilde_nm_phi_frac = np.zeros(N)
    diag_K_tilde_nm_phi = np.zeros(N)

    phi_sq_cumsum = np.cumsum(phi_n_1**2)
    for n in range(N):
        prev_sum = phi_sq_cumsum[n - 1] if n > 0 else 0.0
        K_tilde_nm_phi_frac[n] = (2 * prev_sum + phi_n_1[n] ** 2 - 0.5) * phi_n_1[n]
        diag_K_tilde_nm_phi[n] = (
            2 * phi_n_1[n] ** 2 * prev_sum + 0.5 * (phi_n_1[n] ** 2 - 0.5) ** 2
        )

    K_tilde_nm_phi = np.outer(K_tilde_nm_phi_frac, phi_n_1)
    K_tilde_nm_phi = np.triu(K_tilde_nm_phi) + np.triu(K_tilde_nm_phi).T
    K_tilde_nm_phi = (
        K_tilde_nm_phi - np.diag(np.diag(K_tilde_nm_phi)) + np.diag(diag_K_tilde_nm_phi)
    )

    # Build K_tilde_l
    scale = (2 / a) / (1 + x_i)
    K_tilde_l = np.outer(scale, scale) * (
        T_n_i.T @ K_tilde_nm_phi @ T_n_i + np.outer(pi_i_1, pi_i_1)
    )

    H_tilde_l = U_tilde_l + K_tilde_l

    # Get spherical Bessel zeros and sort them
    z_l_p_raw = calc_z_l(ell, False)

    # Sort: pure real roots first (for odd l), then complex sorted by real part
    if ell % 2 == 0:
        pure_real_z = np.array([], dtype=np.complex128)
    else:
        pure_real_mask = np.imag(z_l_p_raw) == 0
        pure_real_z = z_l_p_raw[pure_real_mask]

    complex_mask = np.imag(z_l_p_raw) != 0
    complex_z = z_l_p_raw[complex_mask]
    sort_idx = np.argsort(np.real(complex_z))
    z_l_p = np.concatenate([pure_real_z, complex_z[sort_idx]])

    # Build block-diagonal unitary U for real-valued similarity transform
    if ell % 2 == 0:
        U = np.array([], dtype=np.complex128).reshape(0, 0)
    else:
        U = np.array([[1.0]], dtype=np.complex128)

    for _ in range(ell // 2):
        block = np.array([[1, 1j], [1, -1j]], dtype=np.complex128)
        if U.size == 0:
            U = block
        else:
            U = linalg.block_diag(U, block)

    bigU = linalg.block_diag(np.eye(N), np.eye(N), U if U.size > 0 else np.eye(0))

    # Build the full matrix
    n_z = len(z_l_p)
    z_f_i_a_mat = -(z_l_p[:, None] * f_i_a[None, :]) / a  # shape (n_z, N)
    diag_z_mat = -np.diag(z_l_p) / a
    diag_rz_mat = np.diag(-1.0 / z_l_p) if n_z > 0 else np.zeros((0, 0))

    # Assemble tot_mat
    tot_mat = np.zeros((2 * N + n_z, 2 * N + n_z), dtype=np.complex128)
    tot_mat[:N, N : 2 * N] = np.eye(N)
    tot_mat[N : 2 * N, :N] = -2 * H_tilde_l
    tot_mat[N : 2 * N, N : 2 * N] = F
    if n_z > 0:
        tot_mat[N : 2 * N, 2 * N :] = np.tile(f_i_a[:, None], (1, n_z)) / a
        tot_mat[2 * N :, :N] = z_f_i_a_mat
        tot_mat[2 * N :, 2 * N :] = diag_z_mat

    # W matrix for normalization
    W = np.zeros((2 * N + n_z, 2 * N + n_z), dtype=np.complex128)
    W[:N, :N] = -F
    W[:N, N : 2 * N] = np.eye(N)
    W[N : 2 * N, :N] = np.eye(N)
    if n_z > 0:
        W[2 * N :, 2 * N :] = diag_rz_mat

    # Apply similarity transform to make matrices real
    if bigU.shape[0] > 0:
        bigU_inv = np.linalg.inv(bigU)
        tot_mat = np.real(bigU_inv @ tot_mat @ bigU)
        W = np.real(bigU.T @ W @ bigU)

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(tot_mat)
    k_n = -1j * eigenvalues

    # Normalize eigenvectors under W
    c_ctilde_zeta = eigenvectors
    c_norm_W = np.diag(c_ctilde_zeta.T @ W @ c_ctilde_zeta)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_factor = np.sqrt(2 * eigenvalues / c_norm_W)
        norm_factor = np.where(np.isfinite(norm_factor), norm_factor, 0)
    c_ctilde_zeta_normalized = c_ctilde_zeta * norm_factor[None, :]

    # Compute eigenmodes at quadrature points
    pi_i_xi = T_n_i.T @ phi_n_xi  # shape (N, N)
    f_i_xi = np.sqrt(2 / a) * ((1 + x_i)[None, :] / (1 + x_i)[:, None]) * pi_i_xi
    eigenmodes = c_ctilde_zeta_normalized[:N, :].T @ f_i_xi

    # Optional: compute CVC, Cf if V_2 is provided
    CVC = None
    Cf = None
    f = None
    C = None
    V2_tilde = None

    if V_2 is not None:
        V2_tilde = np.diag(V_2(r_i))
        C = c_ctilde_zeta_normalized[:N, :].T
        CVC = C @ V2_tilde @ C.T
        Cf = C @ f_i_a
        f = f_i_a

    return TISEResult(
        k_n=k_n,
        c_ctilde_zeta=c_ctilde_zeta,
        eigenmodes=eigenmodes,
        r_i=r_i,
        z_l_p=z_l_p,
        CVC=CVC,
        Cf=Cf,
        f=f,
        C=C,
        V2_tilde=V2_tilde,
    )
