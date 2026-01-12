#!/usr/bin/env python3
"""Generate spherical Hankel zeros lookup table.

Computes complex roots of the polynomial:
    P_ℓ(z) = Σₘ₌₀^ℓ [(2ℓ-m)! / (m!(ℓ-m)!)] * 2^m * z^m

These roots are the complex zeros of spherical Hankel functions θₗ(z).

WARNING: ℓ > 75 often fails to converge with default settings. Increasing
--maxsteps may help, but computation time grows prohibitively (hours for ℓ ~ 150).
For high ℓ, consider using the original Mathematica notebook instead.

Usage:
    uv run --group calc python scripts/generate_bessel_zeros.py --lmax 60
    uv run --group calc python scripts/generate_bessel_zeros.py --lmax 60 --check-against-current
"""

from __future__ import annotations

import argparse
import multiprocessing as mp_proc
import os
import sys
from math import factorial
from pathlib import Path

try:
    from mpmath import mp, mpc
    from sympy import Poly, Symbol, nroots
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Install with: uv sync --group calc", file=sys.stderr)
    sys.exit(1)


# Global config for worker processes (set via initializer)
_worker_config: dict = {}


def compute_poly_coeffs(ell: int) -> list[int]:
    """Compute integer coefficients of P_ℓ(z).

    Coefficient of z^m is: (2ℓ-m)! / (m!(ℓ-m)!) * 2^m
    """
    coeffs = []
    for m in range(ell + 1):
        coeff = factorial(2 * ell - m) // (factorial(m) * factorial(ell - m)) * (2**m)
        coeffs.append(coeff)
    return coeffs


def compute_roots(ell: int, digits: int, maxsteps: int = 200) -> list[complex]:
    """Compute all roots of P_ℓ(z) at given precision."""
    if ell == 0:
        return []

    coeffs = compute_poly_coeffs(ell)
    z = Symbol("z")

    # Build polynomial: coeffs[m] * z^m
    poly_expr = sum(c * z**m for m, c in enumerate(coeffs))
    poly = Poly(poly_expr, z)

    # Compute roots with requested precision
    # Use higher maxsteps for higher-degree polynomials
    roots = nroots(poly, n=digits, maxsteps=maxsteps)

    # Convert to Python complex
    result = [complex(r) for r in roots]

    return result


def sort_roots(roots: list[complex]) -> list[complex]:
    """Sort roots canonically: by real part, then imaginary part."""
    return sorted(roots, key=lambda z: (z.real, z.imag))


def validate_roots(ell: int, roots: list[complex], digits: int) -> bool:
    """Validate roots by checking polynomial residuals and count."""
    if ell == 0:
        return len(roots) == 0

    # Count check
    if len(roots) != ell:
        print(
            f"  ERROR: ℓ={ell} has {len(roots)} roots, expected {ell}", file=sys.stderr
        )
        return False

    # Residual check at high precision
    # The polynomial has huge coefficients (factorials), so absolute residuals
    # can be large. We check relative to the polynomial's scale at the root.
    mp.dps = digits + 10
    coeffs = compute_poly_coeffs(ell)

    for root in roots:
        z = mpc(root.real, root.imag)
        val = sum(c * z**m for m, c in enumerate(coeffs))
        # Scale by leading coefficient * |z|^ell to get relative residual
        scale = coeffs[-1] * abs(z) ** ell if abs(z) > 1 else coeffs[-1]
        rel_residual = abs(val) / scale if scale > 0 else abs(val)
        # Warn only if relative residual is worse than ~1e-15 (well above float precision)
        tol = mp.mpf(1e-15)
        if rel_residual > tol:
            print(
                f"  WARNING: ℓ={ell} root {root} has relative residual {float(rel_residual):.2e}",
                file=sys.stderr,
            )

    return True


def _init_worker(digits: int, maxsteps: int) -> None:
    """Initialize worker process with config."""
    _worker_config["digits"] = digits
    _worker_config["maxsteps"] = maxsteps


def _compute_one(ell: int) -> tuple[int, list[complex] | None, str | None]:
    """Worker function: compute roots for single ℓ. Returns (ell, roots, error)."""
    digits = _worker_config["digits"]
    maxsteps = _worker_config["maxsteps"]

    try:
        roots = compute_roots(ell, digits, maxsteps)
        roots = sort_roots(roots)
        if not validate_roots(ell, roots, digits):
            return (ell, None, f"Validation failed for ℓ={ell}")
        return (ell, roots, None)
    except Exception as e:
        return (ell, None, f"ℓ={ell}: {e}")


def _load_existing_module():
    """Load the existing bessel_zeros module. Returns (module, error_msg)."""
    import importlib.util

    src_path = (
        Path(__file__).parent.parent / "src" / "siegert_scatter" / "bessel_zeros.py"
    )
    if not src_path.exists():
        return None, f"Existing module not found: {src_path}"

    spec = importlib.util.spec_from_file_location("bessel_zeros_existing", src_path)
    if spec is None or spec.loader is None:
        return None, "Could not load existing bessel_zeros module"

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, None


def get_existing_max_ell() -> int | None:
    """Get the maximum ℓ in the existing table, or None if not found."""
    module, error = _load_existing_module()
    if module is None:
        return None
    return max(module._Z_L_P.keys())


def compare_with_existing(
    new_zeros: dict[int, list[complex]], rtol: float = 1e-12
) -> bool:
    """Compare new zeros against existing table."""
    module, error = _load_existing_module()
    if module is None:
        print(error, file=sys.stderr)
        return False

    calc_z_l = module.calc_z_l

    all_match = True
    for ell, new_roots in new_zeros.items():
        try:
            existing = calc_z_l(ell)
        except ValueError:
            print(f"  ℓ={ell}: not in existing table (new extension)")
            continue

        if len(new_roots) != len(existing):
            print(f"  ℓ={ell}: count mismatch ({len(new_roots)} vs {len(existing)})")
            all_match = False
            continue

        # Sort both for comparison
        new_sorted = sort_roots(new_roots)
        existing_sorted = sort_roots(list(existing))

        for i, (new_z, old_z) in enumerate(zip(new_sorted, existing_sorted)):
            rel_diff = abs(new_z - old_z) / max(abs(old_z), 1e-100)
            if rel_diff > rtol:
                print(f"  ℓ={ell} root {i}: relative diff {rel_diff:.2e} > {rtol:.0e}")
                all_match = False

    return all_match


def format_complex(z: complex) -> str:
    """Format complex number for output with full precision."""
    return f"{z.real:+.20e}{z.imag:+.20e}j"


def generate_module(zeros: dict[int, list[complex]], output_path: Path) -> None:
    """Write the zeros to a Python module."""
    lines = [
        '"""Spherical Bessel zeros lookup table (ports calc_z_l.m).',
        "",
        "Generated from MATLAB calc_z_l.m (Mathematica-computed values).",
        '"""',
        "",
        "import numpy as np",
        "",
        "# Complex zeros of spherical Hankel functions",
        f"# Generated from Mathematica up to ell={max(zeros.keys())} (used in TISE)",
        "_Z_L_P: dict[int, np.ndarray] = {",
    ]

    for ell in sorted(zeros.keys()):
        roots = zeros[ell]
        if len(roots) == 0:
            lines.append(f"    {ell}: np.array([], dtype=np.complex128),")
        elif len(roots) == 1:
            z = roots[0]
            lines.append(f"    {ell}: np.array(")
            lines.append(f"        [{format_complex(z)}], dtype=np.complex128")
            lines.append("    ),")
        else:
            lines.append(f"    {ell}: np.array(")
            lines.append("        [")
            for z in roots:
                lines.append(f"            {format_complex(z)},")
            lines.append("        ],")
            lines.append("        dtype=np.complex128,")
            lines.append("    ),")

    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def calc_z_l(ell: int, real_only: bool = False) -> np.ndarray:")
    lines.append('    """Return spherical Bessel zeros for angular momentum ell.')
    lines.append("")
    lines.append("    These are the complex zeros of the spherical Hankel function")
    lines.append("    or equivalently of theta_ell(z) = (d/dz)^ell [sin(z)/z] / z.")
    lines.append("")
    lines.append("    Parameters")
    lines.append("    ----------")
    lines.append("    ell : int")
    lines.append("        Angular momentum quantum number (0 to max in table).")
    lines.append("    real_only : bool, default=False")
    lines.append(
        "        If True, return unique real parts and absolute imaginary parts"
    )
    lines.append("        (for constructing real-valued matrices).")
    lines.append("")
    lines.append("    Returns")
    lines.append("    -------")
    lines.append("    np.ndarray")
    lines.append(
        "        Complex zeros (shape (ell,) typically), or real array if real_only=True."
    )
    lines.append('    """')
    lines.append("    max_ell = max(_Z_L_P.keys())")
    lines.append("    if ell < 0 or ell > max_ell:")
    lines.append('        raise ValueError(f"ell must be 0-{max_ell}, got {ell}")')
    lines.append("")
    lines.append("    z_l_p = _Z_L_P[ell].copy()")
    lines.append("")
    lines.append("    if real_only:")
    lines.append("        # Return unique real and |imag| values, excluding zeros")
    lines.append("        combined = np.concatenate([z_l_p.real, np.abs(z_l_p.imag)])")
    lines.append("        result = np.unique(combined)")
    lines.append("        result = result[result != 0]")
    lines.append("        return result")
    lines.append("")
    lines.append("    return z_l_p")
    lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate spherical Hankel zeros lookup table"
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=60,
        help="Maximum ℓ to compute (default: 60, >75 may fail)",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=30,
        help="Working decimal digits for root computation (default: 30)",
    )
    parser.add_argument(
        "--maxsteps",
        type=int,
        default=200,
        help="Max iterations for root finder (default: 200, increase for high ℓ)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/siegert_scatter/bessel_zeros.py"),
        help="Output Python module path",
    )
    parser.add_argument(
        "--check-against-current",
        action="store_true",
        help="Compare against existing table before overwriting",
    )
    args = parser.parse_args()

    if args.lmax > 75:
        print(
            f"WARNING: ℓ > 75 often fails to converge. Increase --maxsteps to help,\n"
            f"         but expect very long runtimes (hours for ℓ ~ 150).",
            file=sys.stderr,
        )

    n_workers = args.workers or os.cpu_count() or 1
    print(
        f"Computing zeros for ℓ = 0..{args.lmax} at {args.digits} digits ({n_workers} workers)"
    )

    zeros: dict[int, list[complex]] = {}
    ells = list(range(args.lmax + 1))

    with mp_proc.Pool(
        n_workers, initializer=_init_worker, initargs=(args.digits, args.maxsteps)
    ) as pool:
        results = pool.imap_unordered(_compute_one, ells)
        for ell, roots, error in tqdm(results, total=len(ells), desc="Computing roots"):
            if error:
                print(error, file=sys.stderr)
                pool.terminate()
                sys.exit(1)
            zeros[ell] = roots  # type: ignore[assignment]

    if args.check_against_current:
        existing_max = get_existing_max_ell()

        print("\nComparing against existing table...", flush=True)
        if compare_with_existing(zeros):
            print("All values match existing table within tolerance", flush=True)
        else:
            print("Some values differ from existing table", flush=True)

        if existing_max is not None and args.lmax < existing_max:
            print(
                f"\nWARNING: New table (ℓ ≤ {args.lmax}) would discard "
                f"existing values up to ℓ = {existing_max}.",
                file=sys.stderr,
            )
            try:
                alt_path = input(
                    "Enter alternative path to save (or press Enter to discard): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                alt_path = ""

            if alt_path:
                args.output = Path(alt_path)
                print(f"Will save to: {args.output}")
            else:
                print("Discarding computed results.", file=sys.stderr)
                sys.exit(1)

    generate_module(zeros, args.output)


if __name__ == "__main__":
    main()
