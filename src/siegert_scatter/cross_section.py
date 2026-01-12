"""Cross section calculation via SPS (ports calc_cross_section_by_SPS.m)."""

from typing import Callable

import numpy as np

from .tise import tise_by_sps


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
        Energy grid used for calculations (may differ from input if adaptive).
    E_vec_input : np.ndarray
        Original energy grid provided by user.
    """

    def __init__(
        self,
        S_l: np.ndarray,
        sigma_l: np.ndarray,
        tau_l: np.ndarray,
        alpha: float,
        k_n_l: list[np.ndarray],
        E_vec: np.ndarray,
        E_vec_input: np.ndarray | None = None,
    ):
        self.S_l = S_l
        self.sigma_l = sigma_l
        self.tau_l = tau_l
        self.alpha = alpha
        self.k_n_l = k_n_l
        self.E_vec = E_vec
        self.E_vec_input = E_vec_input if E_vec_input is not None else E_vec


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

    Returns
    -------
    CrossSectionResult
        Container with S-matrix, cross sections, time delays, etc.
        The E_vec attribute contains the actual grid used (adaptive or input).
        The E_vec_input attribute contains the original user-provided grid.
    """
    E_vec_input = np.atleast_1d(np.asarray(E_vec, dtype=np.float64))

    # Solve TISE for each l, only need k_n
    k_n_l: list[np.ndarray] = []

    for ell in range(l_max + 1):
        if verbose:
            print(f"l = {ell}")
        result = tise_by_sps(f_x, N, a, ell)
        k_n_l.append(result.k_n)

    # Compute scattering length from l=0 poles
    # alpha = real(a + sum(1i / k_n_l{1}))
    k_n_0 = k_n_l[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.real(a + np.sum(1j / k_n_0))

    # Build k vector - either adaptive or from user input
    k_vec_base = np.sqrt(2 * E_vec_input)

    if adaptive_grid:
        # Compute mean energy spacing for adaptive grid threshold
        dE = (E_vec_input[-1] - E_vec_input[0]) / max(len(E_vec_input) - 1, 1)
        k_vec = _build_resonant_k_grid(k_vec_base, k_n_l, dGamma, dE)
        E_vec_used = k_vec**2 / 2
        if verbose:
            print(
                f"Adaptive grid: {len(E_vec_input)} input points -> "
                f"{len(E_vec_used)} output points"
            )
    else:
        k_vec = k_vec_base
        E_vec_used = E_vec_input

    n_E = len(k_vec)
    S_l = np.zeros((n_E, l_max + 1), dtype=np.complex128)
    sigma_l = np.zeros((n_E, l_max + 1), dtype=np.float64)
    tau_l = np.zeros((n_E, l_max + 1), dtype=np.float64)

    for ell in range(l_max + 1):
        if verbose:
            print(f"Computing S-matrix for l = {ell}")

        k_n = k_n_l[ell]

        # Augment poles with decay width if dGamma > 0
        if dGamma > 0:
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
        E_vec=E_vec_used,
        E_vec_input=E_vec_input,
    )
