#!/usr/bin/env python3
"""Compute SPS cross-sections for K-He scattering from electron structure outputs."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
from schema import ElectronStructureOutputs
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt

from siegert_scatter.cross_section import calc_cross_section, s_matrix_from_poles_all_l
from siegert_scatter.units import reduced_mass, tau_to_dGamma

HBAR = 1.05457173e-34
A0 = 5.2917721092e-11
HARTREE_TO_KELVIN = 315777.1
HARTREE_TO_HZ = 6.57966e15
JOULE_TO_EV = 6.24150913e18
HARTREE_TO_EV = 27.211399
KB = 1.38064852e-23
M_P = 1.66e-27
M1 = 3 * M_P
M2 = 39 * M_P
MU = 1 / (1 / M1 + 1 / M2)
HARTREE_TO_JOULE = 4.35974e-18
JOULE_TO_HZ = JOULE_TO_EV * 241.8e12
JOULE_TO_AU = MU * A0**2 / HBAR**2

G_N = -4.255248
G_E = -2.00231930436153
MU_B = 9.274009682e-24
MU_N = 5.0507835311e-27
MU_0 = 4 * np.pi * 1e-7
A_PREFACTOR = (2 / 3) * MU_0 * G_E * MU_B * G_N * MU_N


def _spin_potential(
    r: np.ndarray,
    potential: np.ndarray,
    rho_spin: np.ndarray,
    spin_factor: float,
) -> Callable:
    energy_joule = potential * HARTREE_TO_JOULE + spin_factor * rho_spin
    spline = CubicSpline(r, JOULE_TO_AU * energy_joule, extrapolate=True)

    return spline


def _interpolate_s_matrix(
    s_l: np.ndarray,
    e_vec_source: np.ndarray,
    e_vec_target: np.ndarray,
) -> np.ndarray:
    n_l = s_l.shape[1]
    result = np.zeros((len(e_vec_target), n_l), dtype=np.complex128)
    for l_idx in range(n_l):
        spline_re = CubicSpline(e_vec_source, np.real(s_l[:, l_idx]), extrapolate=True)
        spline_im = CubicSpline(e_vec_source, np.imag(s_l[:, l_idx]), extrapolate=True)
        result[:, l_idx] = spline_re(e_vec_target) + 1j * spline_im(e_vec_target)
    return result


def _interpolate_real_matrix(
    arr: np.ndarray,
    e_vec_source: np.ndarray,
    e_vec_target: np.ndarray,
) -> np.ndarray:
    n_l = arr.shape[1]
    result = np.zeros((len(e_vec_target), n_l), dtype=np.float64)
    for l_idx in range(n_l):
        spline = CubicSpline(e_vec_source, arr[:, l_idx], extrapolate=True)
        result[:, l_idx] = spline(e_vec_target)
    return result


def _build_union_grid(
    e_vec_s: np.ndarray,
    e_vec_t: np.ndarray,
) -> np.ndarray:
    union = np.union1d(e_vec_s, e_vec_t)
    return np.sort(union)


def _sigma_spin_exchange(
    s_l_s: np.ndarray,
    s_l_t: np.ndarray,
    l_max: int,
    e_vec_joule: np.ndarray,
) -> np.ndarray:
    l_vec = np.arange(l_max + 1)
    prefactor = 2 * l_vec + 1
    diff = (s_l_s - s_l_t) / 2
    return (
        A0**2
        * ((np.pi / 2) / (JOULE_TO_AU * e_vec_joule))
        * np.sum(prefactor * np.abs(diff) ** 2, axis=1)
    )


def _median_filter(values: np.ndarray, width: int) -> np.ndarray:
    kernel = width if width % 2 == 1 else width + 1
    return medfilt(values, kernel_size=kernel)


def compute_all_identities(
    r: np.ndarray,
    all_v: np.ndarray,
    all_rho_alpha: np.ndarray,
    all_rho_beta: np.ndarray,
    identities: list[str],
    indices_to_compute: list[int] | None = None,
    decay_times: list[float] | None = None,
    max_workers: int | None = None,
    verbose: bool = False,
):
    """Compute SPS cross-sections for specified identities.

    Parameters
    ----------
    r : np.ndarray
        Radial grid.
    all_v : np.ndarray
        Potentials array, shape (n_r, n_identities).
    all_rho_alpha : np.ndarray
        Alpha spin density, shape (n_r, n_identities).
    all_rho_beta : np.ndarray
        Beta spin density, shape (n_r, n_identities).
    identities : list[str]
        Full list of identity labels.
    indices_to_compute : list[int] | None
        Specific indices to compute. If None, computes all.
    decay_times : list[float] | None
        External decay times in seconds. If None or empty, uses dGamma=0.
    max_workers : int | None
        Number of parallel workers.
    verbose : bool
        Enable verbose output from cross-section calculations.
    """
    n = len(identities)
    if indices_to_compute is None:
        indices_to_compute = list(range(n))

    mu_amu = reduced_mass(3.016, 38.964)

    # Always include dGamma=0 (tau=inf), then add user-specified decay times
    dGamma_values = [0.0]
    decay_time_labels = ["inf"]
    if decay_times is not None and len(decay_times) > 0:
        dGamma_values.extend(tau_to_dGamma(tau, mu_amu) for tau in decay_times)
        decay_time_labels.extend(f"{tau:.2e}s" for tau in decay_times)

    n_compute = len(indices_to_compute)
    n_basis = 300
    cutoff = 40
    l_max = 150
    temperature = 373  # 100°C, matching paper (arxiv:2201.01255 Figure 3c)
    e_vec_joule = np.linspace(1e-6, 0.6, 30000) / JOULE_TO_EV
    f_e = (
        (2 / np.sqrt(np.pi))
        * (1 / (KB * temperature) ** 1.5)
        * np.exp(-e_vec_joule / (KB * temperature))
    )
    mfw = 700

    spin_factor_s = -(3 / 4) * A_PREFACTOR * (1 / A0) ** 3
    spin_factor_t = (1 / 4) * A_PREFACTOR * (1 / A0) ** 3
    e_vec_au = e_vec_joule * JOULE_TO_AU

    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)

    # Phase 1: Solve TISE once for each identity to get poles (dGamma=0)
    print(f"Phase 1: Computing poles for {n_compute} identities...", flush=True)
    poles_data: list[dict] = []

    start_phase1 = time.perf_counter()
    for compute_idx, i in enumerate(indices_to_compute):
        start_identity = time.perf_counter()
        print(
            f"[{compute_idx + 1:02d}/{n_compute}] {identities[i]}: solving TISE"
            f" (parallel, {workers} workers)",
            flush=True,
        )
        rho_spin = all_rho_alpha[:, i] - all_rho_beta[:, i]
        potential = all_v[:, i]

        f_x_s = _spin_potential(r, potential, rho_spin, spin_factor_s)
        f_x_t = _spin_potential(r, potential, rho_spin, spin_factor_t)

        start_tise_s = time.perf_counter()
        result_s = calc_cross_section(
            f_x_s,
            n_basis,
            cutoff,
            l_max,
            e_vec_au,
            dGamma=0.0,
            verbose=verbose,
            adaptive_grid=True,
            max_workers=max_workers,
        )
        elapsed_s = time.perf_counter() - start_tise_s

        start_tise_t = time.perf_counter()
        result_t = calc_cross_section(
            f_x_t,
            n_basis,
            cutoff,
            l_max,
            e_vec_au,
            dGamma=0.0,
            verbose=verbose,
            adaptive_grid=True,
            max_workers=max_workers,
        )
        elapsed_t = time.perf_counter() - start_tise_t

        print(
            f"[{compute_idx + 1:02d}/{n_compute}] {identities[i]}: "
            f"TISE singlet {elapsed_s:.1f}s, triplet {elapsed_t:.1f}s",
            flush=True,
        )

        e_vec_s_joule = result_s.E_vec / JOULE_TO_AU
        e_vec_t_joule = result_t.E_vec / JOULE_TO_AU
        e_union_joule = _build_union_grid(e_vec_s_joule, e_vec_t_joule)
        k_union = np.sqrt(2 * e_union_joule * JOULE_TO_AU)

        poles_data.append(
            {
                "identity_idx": i,
                "k_n_l_s": result_s.k_n_l,
                "k_n_l_t": result_t.k_n_l,
                "e_union_joule": e_union_joule,
                "k_union": k_union,
                "alpha_s": result_s.alpha * A0,
                "alpha_t": result_t.alpha * A0,
            }
        )

        elapsed = time.perf_counter() - start_identity
        print(
            f"[{compute_idx + 1:02d}/{n_compute}] {identities[i]}: poles computed ({elapsed:.1f}s)",
            flush=True,
        )

    phase1_elapsed = time.perf_counter() - start_phase1
    print(f"Phase 1 complete: {phase1_elapsed:.1f}s\n", flush=True)

    # Phase 2: For each decay time, recompute S-matrices from stored poles
    all_results = {}

    for dGamma_idx, (dGamma, decay_label) in enumerate(
        zip(dGamma_values, decay_time_labels)
    ):
        print(
            f"=== Phase 2: Decay time {decay_label} (dGamma={dGamma:.3e}) ===",
            flush=True,
        )

        gamma_se = np.zeros(n)
        gamma_se_mf = np.zeros(n)
        alpha_s = np.zeros(n)
        alpha_t = np.zeros(n)
        sigma_se = np.zeros((n, e_vec_joule.size))
        s_l_s_list: list[np.ndarray] = []
        s_l_t_list: list[np.ndarray] = []
        e_vec_union_list: list[np.ndarray] = []

        start_phase2 = time.perf_counter()
        for compute_idx, pdata in enumerate(poles_data):
            i = pdata["identity_idx"]
            k_union = pdata["k_union"]
            e_union_joule = pdata["e_union_joule"]

            s_l_s_union = s_matrix_from_poles_all_l(
                pdata["k_n_l_s"], k_union, cutoff, dGamma
            )
            s_l_t_union = s_matrix_from_poles_all_l(
                pdata["k_n_l_t"], k_union, cutoff, dGamma
            )

            s_l_s_list.append(s_l_s_union)
            s_l_t_list.append(s_l_t_union)
            e_vec_union_list.append(e_union_joule)
            alpha_s[i] = pdata["alpha_s"]
            alpha_t[i] = pdata["alpha_t"]

            sigma_se_union = _sigma_spin_exchange(
                s_l_s_union, s_l_t_union, l_max, e_union_joule
            )
            sigma_se_on_input = CubicSpline(
                e_union_joule, sigma_se_union, extrapolate=True
            )(e_vec_joule)
            sigma_se[i, :] = sigma_se_on_input

            gamma_se[i] = 1e6 * np.trapezoid(
                np.sqrt(2 / MU) * e_vec_joule * sigma_se[i, :] * f_e, e_vec_joule
            )
            sigma_se_mf = _median_filter(sigma_se[i, :], mfw)
            gamma_se_mf[i] = 1e6 * np.trapezoid(
                np.sqrt(2 / MU) * e_vec_joule * sigma_se_mf * f_e, e_vec_joule
            )

            print(
                f"  [{compute_idx + 1:02d}/{n_compute}] {identities[i]}: "
                f"Gamma_SE={gamma_se[i]:.3e}, Gamma_SE_MF={gamma_se_mf[i]:.3e}",
                flush=True,
            )

        phase2_elapsed = time.perf_counter() - start_phase2
        print(f"Decay={decay_label} complete: {phase2_elapsed:.1f}s\n", flush=True)

        all_results[decay_label] = {
            "e_vec_joule": e_vec_joule,
            "e_vec_union_list": e_vec_union_list,
            "sigma_se": sigma_se.copy(),
            "gamma_se": gamma_se.copy(),
            "gamma_se_mf": gamma_se_mf.copy(),
            "alpha_s": alpha_s.copy(),
            "alpha_t": alpha_t.copy(),
            "s_l_s": s_l_s_list,
            "s_l_t": s_l_t_list,
            "dGamma": dGamma,
            "indices_computed": indices_to_compute,
            "poles_data": poles_data,
        }

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SPS cross-sections from electron structure outputs."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to electron structure outputs JSON file",
    )
    parser.add_argument(
        "--start-identity",
        type=int,
        default=0,
        help="Zero-based start index for all-identities section",
    )
    parser.add_argument(
        "--max-identities",
        type=int,
        default=None,
        help="Maximum identities to process in all-identities section",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count, use 1 to disable parallelism)",
    )
    parser.add_argument(
        "--identities",
        type=str,
        default=None,
        help=(
            "Specific identities to calculate, separated by '+' (e.g., '5s' or '4s+5s'). "
            "If not specified, uses --start-identity and --max-identities for index-based selection."
        ),
    )
    parser.add_argument(
        "--decay-times",
        type=str,
        default=None,
        help=(
            "External decay times in seconds, comma-separated (e.g., '1e-12,1e-11,1e-10,1e-9'). "
            "If not specified, no external decay is applied (dGamma=0)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results as a pickle file (e.g., 'results.pkl')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from cross-section calculations",
    )
    args = parser.parse_args()

    data = ElectronStructureOutputs.from_json_file(args.input)
    r = data.r_array()
    all_v = data.v_array()
    all_rho_alpha = data.rho_alpha_array()
    all_rho_beta = data.rho_beta_array()
    identities = list(data.all_identities)

    indices_to_compute: list[int] | None = None
    if args.identities:
        requested = [s.strip() for s in args.identities.split("+")]
        indices_to_compute = []
        for name in requested:
            if name not in identities:
                print(f"Error: Identity '{name}' not found. Available: {identities}")
                return
            indices_to_compute.append(identities.index(name))
    elif args.start_identity != 0 or args.max_identities is not None:
        start = args.start_identity
        end = (
            len(identities)
            if args.max_identities is None
            else min(len(identities), start + args.max_identities)
        )
        indices_to_compute = list(range(start, end))

    decay_times: list[float] | None = None
    if args.decay_times:
        decay_times = [float(t.strip()) for t in args.decay_times.split(",")]

    all_results = compute_all_identities(
        r,
        all_v,
        all_rho_alpha,
        all_rho_beta,
        identities,
        indices_to_compute=indices_to_compute,
        decay_times=decay_times,
        max_workers=args.workers,
        verbose=args.verbose,
    )

    if args.output is not None:
        import pickle

        with open(args.output, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
