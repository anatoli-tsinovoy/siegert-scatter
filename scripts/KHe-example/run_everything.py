#!/usr/bin/env python3
"""Compute SPS cross-sections for K-He scattering from electron structure outputs."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt

from schema import ElectronStructureOutputs
from siegert_scatter.cross_section import calc_cross_section

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
    start_index: int = 0,
    max_identities: int | None = None,
    max_workers: int | None = None,
):
    n = len(identities)
    end_index = n if max_identities is None else min(n, start_index + max_identities)
    n_basis = 300
    cutoff = 40
    l_max = 150
    temperature = 473
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

    gamma_se = np.zeros(n)
    gamma_se_mf = np.zeros(n)
    alpha_s = np.zeros(n)
    alpha_t = np.zeros(n)
    sigma_se = np.zeros((n, e_vec_joule.size))

    s_l_s: list[np.ndarray] = []
    s_l_t: list[np.ndarray] = []
    tau_l_s: list[np.ndarray] = []
    tau_l_t: list[np.ndarray] = []
    e_vec_union_list: list[np.ndarray] = []
    k_n_l_s: list[list[np.ndarray]] = []
    k_n_l_t: list[list[np.ndarray]] = []

    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)

    start_total = time.perf_counter()
    for i in range(start_index, end_index):
        start_identity = time.perf_counter()
        print(
            f"[{i + 1:02d}/{n}] {identities[i]}: starting SPS solve"
            f" (parallel, {workers} workers)",
            flush=True,
        )
        rho_spin = all_rho_alpha[:, i] - all_rho_beta[:, i]
        potential = all_v[:, i]

        f_x_s = _spin_potential(r, potential, rho_spin, spin_factor_s)
        f_x_t = _spin_potential(r, potential, rho_spin, spin_factor_t)

        result_s = calc_cross_section(
            f_x_s,
            n_basis,
            cutoff,
            l_max,
            e_vec_au,
            adaptive_grid=True,
            max_workers=max_workers,
        )
        result_t = calc_cross_section(
            f_x_t,
            n_basis,
            cutoff,
            l_max,
            e_vec_au,
            adaptive_grid=True,
            max_workers=max_workers,
        )

        e_vec_s_joule = result_s.E_vec / JOULE_TO_AU
        e_vec_t_joule = result_t.E_vec / JOULE_TO_AU
        e_union_joule = _build_union_grid(e_vec_s_joule, e_vec_t_joule)

        s_l_s_union = _interpolate_s_matrix(result_s.S_l, e_vec_s_joule, e_union_joule)
        s_l_t_union = _interpolate_s_matrix(result_t.S_l, e_vec_t_joule, e_union_joule)

        s_l_s.append(s_l_s_union)
        s_l_t.append(s_l_t_union)
        e_vec_union_list.append(e_union_joule)
        k_n_l_s.append(result_s.k_n_l)
        k_n_l_t.append(result_t.k_n_l)

        tau_s_interp = _interpolate_real_matrix(
            result_s.tau_l, e_vec_s_joule, e_union_joule
        )
        tau_t_interp = _interpolate_real_matrix(
            result_t.tau_l, e_vec_t_joule, e_union_joule
        )
        tau_l_s.append(tau_s_interp * JOULE_TO_AU * HBAR)
        tau_l_t.append(tau_t_interp * JOULE_TO_AU * HBAR)
        alpha_s[i] = result_s.alpha * A0
        alpha_t[i] = result_t.alpha * A0

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
        elapsed = time.perf_counter() - start_identity
        print(
            f"[{i + 1:02d}/{n}] {identities[i]}: "
            f"Gamma_SE={gamma_se[i]:.3e}, Gamma_SE_MF={gamma_se_mf[i]:.3e} "
            f"(elapsed {elapsed:.1f}s)",
            flush=True,
        )

    total_elapsed = time.perf_counter() - start_total
    print(
        f"Completed identities {start_index + 1}-{end_index} in {total_elapsed:.1f}s",
        flush=True,
    )

    return {
        "e_vec_joule": e_vec_joule,
        "e_vec_union_list": e_vec_union_list,
        "sigma_se": sigma_se,
        "gamma_se": gamma_se,
        "gamma_se_mf": gamma_se_mf,
        "alpha_s": alpha_s,
        "alpha_t": alpha_t,
        "s_l_s": s_l_s,
        "s_l_t": s_l_t,
        "tau_l_s": tau_l_s,
        "tau_l_t": tau_l_t,
        "k_n_l_s": k_n_l_s,
        "k_n_l_t": k_n_l_t,
    }


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
        "--sections",
        nargs="+",
        default=["all-identities"],
        choices=["all-identities"],
        help="Which SPS sections to run",
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
    args = parser.parse_args()

    sections = set(args.sections)
    if "all" in sections:
        sections = {"all-identities", "4p", "4s"}

    data = ElectronStructureOutputs.from_json_file(args.input)
    r = data.r_array()
    all_v = data.v_array()
    all_rho_alpha = data.rho_alpha_array()
    all_rho_beta = data.rho_beta_array()
    identities = list(data.all_identities)

    if "all-identities" in sections:
        compute_all_identities(
            r,
            all_v,
            all_rho_alpha,
            all_rho_beta,
            identities,
            start_index=args.start_identity,
            max_identities=args.max_identities,
            max_workers=args.workers,
        )

    print(
        "Total AM calculation (MATLAB: calc_cross_section_by_SPS_with_total_AM) "
        "not ported; outputs unused."
    )


if __name__ == "__main__":
    main()
