"""Command-line interface for siegert-scatter."""

import argparse
import json
import sys

import numpy as np

from .bessel_zeros import calc_z_l
from .cross_section import calc_cross_section_by_sps
from .polynomials import j_polynomial
from .quadrature import get_gaussian_quadrature
from .tise import tise_by_sps


def cmd_quick(args: argparse.Namespace) -> int:
    """Run quick validation tests (matches SPS_test_quick.m)."""
    print("=== SPS QUICK VALIDATION TEST ===")
    print()

    # Test 1: Jacobi polynomial spot check
    x = np.array([0, 0.5, 1])
    JP = j_polynomial(3, 2, 0, 4, x)
    print("Jacobi P_2(0,4) at x=0,0.5,1:")
    if args.json:
        result = {"jacobi_P2": JP[:, 2].tolist()}
    else:
        print(f"  {JP[:, 2]}")

    # Test 2: Gauss-Jacobi quadrature
    w, nodes = get_gaussian_quadrature(5, 0, 0, -1, 1)
    print("\nGauss-Legendre N=5 nodes:")
    if args.json:
        result["quadrature_nodes"] = nodes.tolist()
    else:
        print(f"  {nodes}")

    # Test 3: Bessel zeros
    z3 = calc_z_l(3, False)
    print("\nSpherical Bessel zeros l=3:")
    if args.json:
        result["bessel_zeros_l3"] = [{"real": z.real, "imag": z.imag} for z in z3]
    else:
        for z in z3:
            print(f"  {z.real:.12e} + {z.imag:.12e}i")

    # Test 4: TISE eigenvalues for Pöschl-Teller potential
    LAMBDA_PT = 4
    a_pt = 20

    # Radial potential (centered at origin)
    def f_pt_radial(r: np.ndarray) -> np.ndarray:
        return -LAMBDA_PT * (LAMBDA_PT + 1) / 2 / np.cosh(r) ** 2

    result_tise = tise_by_sps(f_pt_radial, 100, a_pt, 0)
    k_n = result_tise.k_n

    # Bound states: small real part, positive imaginary part
    bound_mask = (np.abs(np.real(k_n)) < 1e-6) & (np.imag(k_n) > 0.5)
    bound = k_n[bound_mask]
    E_bound = -(np.abs(bound) ** 2) / 2
    E_bound = np.sort(E_bound)[::-1]

    print("\nTest 4a: Radial l=0 bound states (psi(0)=0, odd parity only):")
    if args.json:
        result["bound_energies_radial"] = E_bound.tolist()
    else:
        print(f"  {E_bound}")
        print("Exact: -0.5, -4.5")

    # Test 4b: Full potential (shifted)
    def f_pt_full(r: np.ndarray) -> np.ndarray:
        return -LAMBDA_PT * (LAMBDA_PT + 1) / 2 / np.cosh(r - a_pt / 2) ** 2

    result_tise_full = tise_by_sps(f_pt_full, 100, a_pt, 0)
    k_n_full = result_tise_full.k_n

    bound_mask_full = (np.abs(np.real(k_n_full)) < 1e-6) & (np.imag(k_n_full) > 0.5)
    bound_full = k_n_full[bound_mask_full]
    E_bound_full = -(np.abs(bound_full) ** 2) / 2
    E_bound_full = np.sort(E_bound_full)[::-1]

    print("\nTest 4b: Full Pöschl-Teller (centered at a/2):")
    if args.json:
        result["bound_energies_full"] = E_bound_full.tolist()
    else:
        print(f"  {E_bound_full}")
        print("Exact: -0.5, -2, -4.5, -8")

    # Test 5: Scattering length
    cs_result = calc_cross_section_by_sps(
        f_pt_radial, 100, a_pt, 0, np.array([0.01]), np.inf
    )
    print(f"\nScattering length: {cs_result.alpha:.10e}")
    if args.json:
        result["scattering_length"] = cs_result.alpha

    # Test 6: S-matrix at E=1
    E_test = np.array([1.0])
    cs_result_E1 = calc_cross_section_by_sps(f_pt_radial, 100, a_pt, 2, E_test, np.inf)

    print("\nAt E=1:")
    S_0 = cs_result_E1.S_l[0, 0]
    sigma_0 = cs_result_E1.sigma_l[0, 0]
    print(f"  S_0 = {S_0.real:.10e} + {S_0.imag:.10e}i")
    print(f"  sigma_0 = {sigma_0:.10e}")
    print(f"  |S_0| = {np.abs(S_0):.15e}")

    if args.json:
        result["S_0_at_E1"] = {"real": S_0.real, "imag": S_0.imag}
        result["sigma_0_at_E1"] = sigma_0
        result["S_0_magnitude"] = np.abs(S_0)
        print("\n=== JSON OUTPUT ===")
        print(json.dumps(result, indent=2))

    print("\n=== VALIDATION COMPLETE ===")
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    """Run full test suite (matches SPS_test_suite.m)."""
    print("=" * 64)
    print("SPS TEST SUITE - Reference outputs for Python implementation")
    print("=" * 64)
    print()

    results: dict = {}

    # Test 1: Jacobi Polynomials
    print("=== TEST 1: Jacobi Polynomials ===")
    x_test = np.array([-1, -0.5, 0, 0.5, 1])
    alpha_j, beta_j = 0, 4
    JP = j_polynomial(len(x_test), 5, alpha_j, beta_j, x_test)
    print(f"Test 1.1: j_polynomial(5, 5, {alpha_j}, {beta_j}, [-1,-0.5,0,0.5,1])")
    print(f"Shape: {JP.shape}")
    print("Values:")
    print(JP)

    results["test_jacobi_polynomial"] = {
        "alpha": alpha_j,
        "beta": beta_j,
        "x": x_test.tolist(),
        "P_0": JP[:, 0].tolist(),
        "P_1": JP[:, 1].tolist(),
    }

    # Test 2: Gaussian Quadrature
    print("\n=== TEST 2: Gaussian Quadrature ===")
    omega_i, x_i = get_gaussian_quadrature(5, 0, 0, -1, 1)
    print("Test 2.1: Gauss-Jacobi quadrature N=5, alpha=0, beta=0")
    print(f"Nodes: {x_i}")
    print(f"Weights: {omega_i}")

    results["test_quadrature_N5_l0"] = {
        "nodes": x_i.tolist(),
        "weights": omega_i.tolist(),
    }

    # Test 3: Spherical Bessel Zeros
    print("\n=== TEST 3: Spherical Bessel Zeros ===")
    for ell in range(6):
        z_l = calc_z_l(ell, False)
        print(f"Test 3.{ell + 1}: calc_z_l({ell}, false)")
        print(f"  Number of zeros: {len(z_l)}")
        if len(z_l) > 0:
            for i, z in enumerate(z_l):
                print(f"  z_{ell}_{i + 1} = {z.real:.15e} + {z.imag:.15e}i")

    # Test 4: Median filter
    print("\n=== TEST 4: Median Filter ===")
    from .median_filter import medfilt1

    x_med = np.array([1, 5, 2, 8, 3, 7, 4, 6, 9, 0], dtype=float)
    y_med3 = medfilt1(x_med, 3)
    y_med5 = medfilt1(x_med, 5)
    print(f"Input: {x_med}")
    print(f"Output (window=3): {y_med3}")
    print(f"Output (window=5): {y_med5}")

    # Test 5: TISE Eigenvalue Solver
    print("\n=== TEST 5: TISE Eigenvalue Solver ===")
    LAMBDA_PT = 4

    def f_test(x: np.ndarray) -> np.ndarray:
        return -LAMBDA_PT * (LAMBDA_PT + 1) / 2 / np.cosh(x) ** 2

    N_tise, a_tise, l_tise = 10, 10, 0
    result_tise = tise_by_sps(f_test, N_tise, a_tise, l_tise)
    k_n = result_tise.k_n

    print(f"Test 5.1: TISE_by_SPS with N={N_tise}, a={a_tise}, l={l_tise}")
    print(f"Number of eigenvalues: {len(k_n)}")
    sort_idx = np.argsort(np.real(k_n))
    k_n_sorted = k_n[sort_idx]
    print("First 10 k_n values (sorted by real part):")
    for i in range(min(10, len(k_n_sorted))):
        print(
            f"  k_{i + 1} = {np.real(k_n_sorted[i]):.10e} + {np.imag(k_n_sorted[i]):.10e}i"
        )

    # Bound states
    bound_mask = (np.abs(np.real(k_n)) < 1e-10) & (np.imag(k_n) > 0)
    k_bound = k_n[bound_mask]
    print(f"Number of bound states: {len(k_bound)}")
    for i, kb in enumerate(k_bound):
        E_b = -(np.abs(kb) ** 2) / 2
        print(f"  k_bound_{i + 1} = {np.imag(kb):.10e}i (E = {E_b:.10e})")

    # Test 5.3: Larger N
    N_tise = 30
    result_tise_30 = tise_by_sps(f_test, N_tise, a_tise, 0)
    k_n_30 = result_tise_30.k_n
    bound_mask_30 = (np.abs(np.real(k_n_30)) < 1e-10) & (np.imag(k_n_30) > 0)
    k_bound_30 = k_n_30[bound_mask_30]
    print(f"\nTest 5.3: TISE_by_SPS with N={N_tise}, a={a_tise}, l=0")
    print(f"Number of bound states: {len(k_bound_30)}")
    for i, kb in enumerate(k_bound_30):
        E_b = -(np.abs(kb) ** 2) / 2
        print(f"  E_{i + 1} = {E_b:.10e}")
    print(f"Quadrature points r_i (first 5): {result_tise_30.r_i[:5]}")

    # Test 6: Cross Section
    print("\n=== TEST 6: Cross Section Calculation ===")
    N_cs, a_cs, l_max_cs = 20, 10, 3
    E_vec_test = np.linspace(1e-4, 5, 50)

    cs_result = calc_cross_section_by_sps(
        f_test, N_cs, a_cs, l_max_cs, E_vec_test, np.inf
    )
    print(f"\nScattering length (alpha): {cs_result.alpha:.10e}")

    E_samples = [0, 9, 24, 49]
    print("\n--- S-matrix at selected energies ---")
    for i_e in E_samples:
        if i_e < len(E_vec_test):
            print(f"E = {E_vec_test[i_e]:.4e}")
            for ell in range(l_max_cs + 1):
                S_val = cs_result.S_l[i_e, ell]
                print(
                    f"  S_{ell} = {S_val.real:.10e} + {S_val.imag:.10e}i  "
                    f"(|S|={np.abs(S_val):.10e})"
                )

    print("\n--- Cross sections at selected energies ---")
    for i_e in E_samples:
        if i_e < len(E_vec_test):
            print(f"E = {E_vec_test[i_e]:.4e}")
            for ell in range(l_max_cs + 1):
                print(f"  sigma_{ell} = {cs_result.sigma_l[i_e, ell]:.10e}")
            print(f"  sigma_total = {np.sum(cs_result.sigma_l[i_e, :]):.10e}")

    # Test 7: Full calculation
    print("\n=== TEST 7: Full SPS Calculation ===")
    a, N, l_max = 10, 50, 5
    E_vec = np.linspace(1e-6, 10, 1000)

    def f_x_1(x: np.ndarray) -> np.ndarray:
        return f_test(x) - 1e-6 * (1 / 4) / np.cosh(x)

    def f_x_2(x: np.ndarray) -> np.ndarray:
        return f_test(x) + 1e-6 * (3 / 4) / np.cosh(x)

    print(f"Parameters: N={N}, a={a}, l_max={l_max}")
    print(f"Energy grid: {len(E_vec)} points")

    print("\nComputing potential 1...")
    cs1 = calc_cross_section_by_sps(f_x_1, N, a, l_max, E_vec, np.inf)
    print("Computing potential 2...")
    cs2 = calc_cross_section_by_sps(f_x_2, N, a, l_max, E_vec, np.inf)

    print(f"\nScattering length (alpha_1): {cs1.alpha:.10e}")
    print(f"Scattering length (alpha_2): {cs2.alpha:.10e}")

    results["test_scattering_length"] = {
        "alpha_1": cs1.alpha,
        "alpha_2": cs2.alpha,
    }

    # Rate coefficient
    kbT_natural = 5
    f_E = (
        (2 / np.sqrt(np.pi))
        * (1 / kbT_natural) ** (3 / 2)
        * np.exp(-E_vec / kbT_natural)
    )
    l_vec = np.arange(l_max + 1)
    sigma_12 = (
        np.pi
        / (2 * E_vec)
        * np.sum((2 * l_vec + 1) * np.abs((cs1.S_l - cs2.S_l) / 2) ** 2, axis=1)
    )
    Gamma_12 = 1e6 * np.trapezoid(np.sqrt(2 * E_vec) * sigma_12 * f_E, E_vec)

    print(f"\nRate coefficient Gamma_12: {Gamma_12:.10e} cm^3/sec")

    results["test_rate_coefficient"] = {
        "kbT_natural": kbT_natural,
        "Gamma_12_cm3_per_sec": Gamma_12,
    }

    # Test 8: Unitarity
    print("\n=== TEST 8: Numerical Consistency ===")
    S_mag_sq = np.abs(cs1.S_l[:, 0]) ** 2
    print(f"max(|S_0|^2 - 1) = {np.max(np.abs(S_mag_sq - 1)):.10e}")
    print(f"mean(|S_0|^2) = {np.mean(S_mag_sq):.10e}")

    if args.json:
        print("\n=== SUMMARY (JSON format) ===")
        print(json.dumps(results, indent=2))

    print("\n" + "=" * 64)
    print("TEST SUITE COMPLETE")
    print("=" * 64)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run demo calculation (like SPS_script.m)."""
    print("Running Pöschl-Teller demo calculation...")

    LAMBDA_PT = 4
    a = 10
    N = 50
    l_max = 5

    def f_test(x: np.ndarray) -> np.ndarray:
        return -LAMBDA_PT * (LAMBDA_PT + 1) / 2 / np.cosh(x) ** 2

    E_vec = np.linspace(1e-6, 10, 1000)

    print(f"Parameters: lambda={LAMBDA_PT}, N={N}, a={a}, l_max={l_max}")
    print(f"Energy grid: {len(E_vec)} points from {E_vec[0]:.2e} to {E_vec[-1]:.2e}")

    result = calc_cross_section_by_sps(f_test, N, a, l_max, E_vec, np.inf, verbose=True)

    print(f"\nScattering length: {result.alpha:.10e}")
    print(f"Total cross section at E=1: {np.sum(result.sigma_l[500, :]):.10e}")

    # Find bound states
    for ell in range(min(3, l_max + 1)):
        k_n = result.k_n_l[ell]
        bound_mask = (np.abs(np.real(k_n)) < 1e-8) & (np.imag(k_n) > 0)
        k_bound = k_n[bound_mask]
        if len(k_bound) > 0:
            E_bound = -(np.abs(k_bound) ** 2) / 2
            E_bound = np.sort(E_bound)[::-1]
            print(f"\nl={ell} bound state energies: {E_bound}")

    if args.output:
        try:
            from scipy.io import savemat

            data = {
                "S_l": result.S_l,
                "sigma_l": result.sigma_l,
                "tau_l": result.tau_l,
                "E_vec": E_vec,
                "alpha": result.alpha,
            }
            savemat(args.output, data)
            print(f"\nSaved results to {args.output}")
        except ImportError:
            print("\nWarning: scipy.io.savemat not available, skipping .mat output")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="siegert-scatter",
        description="Scattering calculations using Siegert pseudostates",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # quick command
    p_quick = subparsers.add_parser("quick", help="Run quick validation tests")
    p_quick.add_argument("--json", action="store_true", help="Output results as JSON")
    p_quick.set_defaults(func=cmd_quick)

    # suite command
    p_suite = subparsers.add_parser("suite", help="Run full test suite")
    p_suite.add_argument("--json", action="store_true", help="Output summary as JSON")
    p_suite.add_argument("--fast", action="store_true", help="Use reduced parameters")
    p_suite.set_defaults(func=cmd_suite)

    # demo command
    p_demo = subparsers.add_parser("demo", help="Run demo calculation")
    p_demo.add_argument("-o", "--output", help="Output .mat file path")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
