from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from poetry_gp.backends.scalapack_fit import fit_exact_gp_scalapack


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_spd_matrix(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 16))
    K = x @ x.T + 1e-2 * np.eye(n)
    y = rng.normal(size=n)
    return K, y


def scipy_reference(K: np.ndarray, y: np.ndarray) -> dict[str, float]:
    c_and_lower = cho_factor(K, lower=True, check_finite=False)
    alpha = cho_solve(c_and_lower, y, check_finite=False)
    l_tri = np.tril(c_and_lower[0])
    logdet = float(2.0 * np.sum(np.log(np.diag(l_tri))))
    return {
        "alpha_norm": float(np.linalg.norm(alpha)),
        "logdet": logdet,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="32,64,128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--launcher", choices=["srun", "mpirun"], default="srun")
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--executable", default="native/build/scalapack_gp_fit")
    parser.add_argument("--output", type=Path, default=Path("results/bench_scalapack_fit.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = parse_int_list(args.n_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "n",
        "nprocs",
        "block_size",
        "factor_seconds",
        "solve_seconds",
        "gather_seconds",
        "total_seconds",
        "logdet",
        "alpha_norm",
        "implemented",
        "message",
        "logdet_abs_error_vs_scipy",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for offset, n in enumerate(n_values):
            K, y = make_spd_matrix(n, seed=args.seed + offset)
            scipy_result = scipy_reference(K, y)
            writer.writerow(
                {
                    "backend": "scipy_serial",
                    "n": n,
                    "nprocs": 1,
                    "block_size": "",
                    "factor_seconds": "",
                    "solve_seconds": "",
                    "gather_seconds": "",
                    "total_seconds": "",
                    "logdet": scipy_result["logdet"],
                    "alpha_norm": scipy_result["alpha_norm"],
                    "implemented": True,
                    "message": "SciPy reference",
                    "logdet_abs_error_vs_scipy": 0.0,
                }
            )
            native_result = fit_exact_gp_scalapack(
                K,
                y,
                launcher=args.launcher,
                nprocs=args.nprocs,
                executable=args.executable,
                block_size=args.block_size,
            )
            writer.writerow(
                {
                    "backend": native_result.backend,
                    "n": n,
                    "nprocs": args.nprocs,
                    "block_size": args.block_size,
                    "factor_seconds": native_result.factor_seconds,
                    "solve_seconds": native_result.solve_seconds,
                    "gather_seconds": native_result.gather_seconds,
                    "total_seconds": native_result.total_seconds,
                    "logdet": native_result.logdet,
                    "alpha_norm": float(np.linalg.norm(native_result.alpha)),
                    "implemented": native_result.implemented,
                    "message": native_result.message,
                    "logdet_abs_error_vs_scipy": abs(native_result.logdet - scipy_result["logdet"]),
                }
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
