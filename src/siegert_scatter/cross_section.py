"""Cross section calculation via SPS (ports calc_cross_section_by_SPS.m)."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np


def _solve_single_channel(
    args: tuple[Callable[[np.ndarray], np.ndarray], int, float, int],
) -> tuple[int, np.ndarray]:
    """Worker function for parallel TISE solving."""
    from .tise import tise_by_sps

    potential_func, N, a, ell = args
    return (ell, tise_by_sps(potential_func, N, a, ell).k_n)


def _compute_s_matrix_single(
    args: tuple[np.ndarray, np.ndarray, float, int, float],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Worker function for parallel S-matrix computation."""
    k_vec, k_n, a, ell, dGamma = args
    k_n_aug = _augment_poles(k_n, dGamma)
    S, sigma, tau = _compute_s_matrix_from_poles(
        k_vec, k_n_aug, a, prefactor=2 * ell + 1
    )
    return (ell, S, sigma, tau)


def _augment_poles(k_n: np.ndarray, dGamma: float) -> np.ndarray:
    """Augment scattering poles with decay width.

    k_n_aug = Re(k_n) + i*(Im(k_n) - dGamma/|k_n|)  for scattering poles only.
    Bound state poles (Re(k_n) == 0) are left unchanged.

    Parameters
    ----------
    k_n : np.ndarray
        Original pole positions.
    dGamma : float
        Width parameter (natural energy units).

    Returns
    -------
    np.ndarray
        Augmented poles.
    """
    if dGamma <= 0:
        return k_n

    is_scattering = np.real(k_n) != 0
    im_shift = np.where(is_scattering, dGamma / np.abs(k_n), 0.0)
    return np.real(k_n) + 1j * (np.imag(k_n) - im_shift)


def _compute_s_matrix_from_poles(
    k_vec: np.ndarray,
    k_n: np.ndarray,
    a: float,
    prefactor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute S-matrix, cross section, and time delay from poles.

    Parameters
    ----------
    k_vec : np.ndarray
        Wavenumber grid, shape (n_E,).
    k_n : np.ndarray
        Pole positions (possibly augmented with decay width).
    a : float
        Cutoff radius.
    prefactor : float
        Degeneracy factor: (2*l+1) or (2*Jtag+1).

    Returns
    -------
    S : np.ndarray
        S-matrix values, shape (n_E,).
    sigma : np.ndarray
        Cross section, shape (n_E,).
    tau : np.ndarray
        Time delay, shape (n_E,).
    """
    S = np.exp(-2j * k_vec * a)
    d_delta_dk = -a * np.ones_like(k_vec)

    for k_pole in k_n:
        S = S * (k_pole + k_vec) / (k_pole - k_vec)

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

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = prefactor * (np.pi / k_vec**2) * np.abs(1 - S) ** 2
        sigma = np.where(np.isfinite(sigma), sigma, 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = d_delta_dk / k_vec
        tau = np.where(np.isfinite(tau), tau, 0)

    return S, sigma, tau


def _solve_and_compute(
    channels: list[tuple[Callable[[np.ndarray], np.ndarray], int]],
    N: int,
    a: float,
    E_vec_input: np.ndarray,
    dGamma: float,
    adaptive_grid: bool,
    verbose: bool,
    max_workers: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n_channels = len(channels)
    potential = channels[0][0]
    l_max = n_channels - 1

    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)

    if verbose:
        print(f"Solving TISE for l=0..{l_max}")

    channel_args = [(potential, N, a, ell) for ell in range(l_max + 1)]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        results = list(executor.map(_solve_single_channel, channel_args))
    results.sort(key=lambda x: x[0])
    k_n_list = [k_n for _, k_n in results]

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.real(a + np.sum(1j / k_n_list[0]))

    k_vec_base = np.sqrt(2 * E_vec_input)
    if adaptive_grid:
        dE = (E_vec_input[-1] - E_vec_input[0]) / max(len(E_vec_input) - 1, 1)
        k_vec = _build_resonant_k_grid(k_vec_base, k_n_list, dGamma, dE)
        E_vec = k_vec**2 / 2
        if verbose:
            print(f"Adaptive grid: {len(E_vec_input)} -> {len(E_vec)} points")
    else:
        k_vec = k_vec_base
        E_vec = E_vec_input

    if verbose:
        print(f"Computing S-matrices for l=0..{l_max}")

    n_E = len(k_vec)
    S = np.zeros((n_E, n_channels), dtype=np.complex128)
    sigma = np.zeros((n_E, n_channels), dtype=np.float64)
    tau = np.zeros((n_E, n_channels), dtype=np.float64)

    s_matrix_args = [
        (k_vec, k_n_list[ell], a, ell, dGamma) for ell in range(n_channels)
    ]
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        s_results = list(executor.map(_compute_s_matrix_single, s_matrix_args))
    s_results.sort(key=lambda x: x[0])
    for ell, S_l, sigma_l, tau_l in s_results:
        S[:, ell] = S_l
        sigma[:, ell] = sigma_l
        tau[:, ell] = tau_l

    return k_n_list, S, sigma, tau, E_vec, alpha


class CrossSectionResult:
    """Results from cross section calculation.

    Attributes
    ----------
    S_l : np.ndarray
        S-matrix elements, shape (n_E, n_l).
    sigma_l : np.ndarray
        Partial cross sections, shape (n_E, n_l).
    tau_l : np.ndarray
        Time delays, shape (n_E, n_l).
    k_n_l : list[np.ndarray]
        Eigenvalues k_n for each angular momentum.
    E_vec : np.ndarray
        Energy grid used for calculations (may differ from input if adaptive).
    E_vec_input : np.ndarray
        Original energy grid provided by user.
    l_vec : np.ndarray
        Angular momentum quantum numbers (0, 1, ..., l_max).
    alpha : float
        Scattering length (l=0 channel).
    """

    def __init__(
        self,
        S_l: np.ndarray,
        sigma_l: np.ndarray,
        tau_l: np.ndarray,
        k_n_l: list[np.ndarray],
        E_vec: np.ndarray,
        E_vec_input: np.ndarray,
        l_vec: np.ndarray,
        alpha: float,
    ):
        self.S_l = S_l
        self.sigma_l = sigma_l
        self.tau_l = tau_l
        self.k_n_l = k_n_l
        self.E_vec = E_vec
        self.E_vec_input = E_vec_input
        self.l_vec = l_vec
        self.alpha = alpha


def _build_resonant_k_grid(
    k_vec_base: np.ndarray,
    k_n_l: list[np.ndarray],
    dGamma: float,
    dE: float,
    min_points_per_width: int = 10,
) -> np.ndarray:
    """Build k-grid with extra density around narrow resonances.

    Parameters
    ----------
    k_vec_base : np.ndarray
        Base k-vector from user-provided energies.
    k_n_l : list[np.ndarray]
        Pole positions for each angular momentum.
    dGamma : float
        Width parameter (natural energy units). Local k-width is dGamma/k.
    dE : float
        Energy grid spacing from user input (used to determine local k spacing).
    min_points_per_width : int, default=10
        Minimum number of grid points required to resolve a resonance width.
        Resonances with fewer than this many points get extra grid density.

    Returns
    -------
    np.ndarray
        Refined k-vector with extra points around resonances.
    """
    k_min = k_vec_base.min()
    k_max = k_vec_base.max()

    k_points = []

    # Scan all poles for narrow resonances
    for k_n in k_n_l:
        for pole in k_n:
            re_k = np.real(pole)
            im_k = np.imag(pole)

            # Only consider scattering poles (Re(k) > 0)
            if re_k > 0:
                # Local k-spacing at this resonance energy: dk = dE / k
                dk_local = dE / re_k

                # Resonance is "narrow" if width < min_points_per_width * dk_local
                resonance_width = np.abs(im_k)
                if resonance_width < min_points_per_width * dk_local:
                    # Width for grid construction: max of decay width and intrinsic
                    width = max(dGamma / re_k, resonance_width)

                    # Build nested linspace segments around resonance
                    min_val = re_k - 3 * width
                    far_min_val = re_k - 10 * width
                    super_far_min_val = re_k - 100 * width

                    max_val = re_k + 3 * width
                    far_max_val = re_k + 10 * width
                    super_far_max_val = re_k + 100 * width
                    ultra_far_max_val = re_k + 1000 * width

                    k_points.extend(np.linspace(super_far_min_val, far_min_val, 19))
                    k_points.extend(np.linspace(far_min_val, min_val, 19))
                    k_points.extend(np.linspace(min_val, max_val, 19))
                    k_points.extend(np.linspace(max_val, far_max_val, 19))
                    k_points.extend(np.linspace(far_max_val, super_far_max_val, 19))
                    k_points.extend(
                        np.linspace(super_far_max_val, ultra_far_max_val, 19)
                    )

    # Start with the original input grid, add resonance points
    k_points.extend(k_vec_base)

    # Sort, unique, and crop to original range
    k_vec_out = np.array(k_points)
    k_vec_out = k_vec_out[(k_vec_out >= k_min) & (k_vec_out <= k_max)]
    k_vec_out = np.unique(k_vec_out)

    return k_vec_out


def calc_cross_section(
    f_x: Callable[[np.ndarray], np.ndarray],
    N: int,
    a: float,
    l_max: int,
    E_vec: np.ndarray,
    dGamma: float = 0.0,
    verbose: bool = False,
    adaptive_grid: bool = True,
    max_workers: int | None = None,
) -> CrossSectionResult:
    """Compute elastic scattering cross sections using Siegert pseudostates.

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
        Energies to compute scattering at (a.u.). Defines the range; the actual
        grid may be refined if adaptive_grid=True.
    dGamma : float, default=0.0
        Width parameter for pole augmentation (natural energy units).
        Poles are shifted in imaginary direction by dGamma/|k_n|.
        Caller converts from lifetime: dGamma = t_au * (mu/m_e) / tau_seconds.
    verbose : bool, default=False
        Print progress.
    adaptive_grid : bool, default=True
        If True (recommended), builds a resonance-adaptive energy grid with
        extra density around narrow resonances. The output E_vec will differ
        from the input. If False, uses the user-provided E_vec exactly, but
        this is NOT RECOMMENDED as it may miss narrow resonance features.
    max_workers : int | None, default=None
        Maximum worker processes for parallel execution. None uses all CPUs.
        Use max_workers=1 to disable parallelism.

    Returns
    -------
    CrossSectionResult
        Container with S-matrix, cross sections, time delays, etc.
        The E_vec attribute contains the actual grid used (adaptive or input).
        The E_vec_input attribute contains the original user-provided grid.
    """
    E_vec_input = np.atleast_1d(np.asarray(E_vec, dtype=np.float64))

    channels = [(f_x, ell) for ell in range(l_max + 1)]

    k_n_list, S_mat, sigma, tau, E_vec_used, alpha = _solve_and_compute(
        channels,
        N,
        a,
        E_vec_input,
        dGamma,
        adaptive_grid,
        verbose,
        max_workers,
    )

    return CrossSectionResult(
        S_l=S_mat,
        sigma_l=sigma,
        tau_l=tau,
        k_n_l=k_n_list,
        E_vec=E_vec_used,
        E_vec_input=E_vec_input,
        l_vec=np.arange(l_max + 1),
        alpha=alpha,
    )
