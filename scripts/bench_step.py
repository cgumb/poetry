from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.backends.naive import run_naive_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["naive", "blocked"], required=True)
    parser.add_argument("--fit-backend", choices=["python", "native_reference"], default="python")
    parser.add_argument("--score-backend", choices=["python", "daemon", "auto", "gpu", "none"], default="python",
                        help="Scoring backend: python (CPU BLAS), daemon (MPI parallel), auto (try daemon), gpu (CUDA/CuPy), none (skip scoring)")
    parser.add_argument("--n-poems", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--m-rated", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--optimize-hyperparameters", action="store_true")
    parser.add_argument("--optimizer-maxiter", type=int, default=50)
    parser.add_argument("--scalapack-launcher", choices=["srun", "mpirun"], default="srun")
    parser.add_argument("--scalapack-nprocs", type=int, default=4)
    parser.add_argument("--scalapack-executable", default="native/build/scalapack_gp_fit")
    parser.add_argument("--scalapack-block-size", type=int, default=128)
    parser.add_argument("--scalapack-native-backend", choices=["auto", "mpi", "scalapack"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None, help="Output results to CSV file")
    parser.add_argument("--append", action="store_true", help="Append to CSV file instead of overwriting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    embeddings = rng.normal(size=(args.n_poems, args.dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12

    rated_indices = rng.choice(args.n_poems, size=args.m_rated, replace=False)
    ratings = rng.normal(size=args.m_rated)

    if args.backend == "naive":
        if args.fit_backend != "python":
            raise ValueError("naive backend currently supports only fit-backend=python")
        result = run_naive_step(
            embeddings,
            rated_indices,
            ratings,
            length_scale=args.length_scale,
            variance=args.variance,
            noise=args.noise,
            optimize_hyperparameters=args.optimize_hyperparameters,
            optimizer_maxiter=args.optimizer_maxiter,
        )
    else:
        result = run_blocked_step(
            embeddings,
            rated_indices,
            ratings,
            length_scale=args.length_scale,
            variance=args.variance,
            noise=args.noise,
            block_size=args.block_size,
            optimize_hyperparameters=args.optimize_hyperparameters,
            optimizer_maxiter=args.optimizer_maxiter,
            fit_backend=args.fit_backend,
            score_backend=args.score_backend,
            scalapack_launcher=args.scalapack_launcher,
            scalapack_nprocs=args.scalapack_nprocs,
            scalapack_executable=args.scalapack_executable,
            scalapack_block_size=args.scalapack_block_size,
            scalapack_native_backend=args.scalapack_native_backend,
        )

    optimization_result = result.state.optimization_result or {}
    profile = {
        "backend": args.backend,
        "fit_backend": args.fit_backend,
        "native_backend": optimization_result.get("native_backend"),
        "n_poems": args.n_poems,
        "dim": args.dim,
        "m_rated": args.m_rated,
        "block_size": args.block_size,
        "optimize_hyperparameters": args.optimize_hyperparameters,
        "optimizer_maxiter": args.optimizer_maxiter,
        "scalapack_nprocs": args.scalapack_nprocs,
        "scalapack_block_size": args.scalapack_block_size,
        "fit_seconds": result.profile.fit_seconds,
        "optimize_seconds": result.profile.optimize_seconds,
        "score_seconds": result.profile.score_seconds,
        "select_seconds": result.profile.select_seconds,
        "total_seconds": result.profile.total_seconds,
        "length_scale": result.state.length_scale,
        "variance": result.state.variance,
        "noise": result.state.noise,
        "log_marginal_likelihood": result.state.log_marginal_likelihood,
        "optimization_success": optimization_result.get("success", ""),
        "used_optimized_params": optimization_result.get("used_optimized_params", False),
        "optimization_nit": optimization_result.get("nit", ""),
        "exploit_index": result.exploit_index,
        "explore_index": result.explore_index,
    }

    # Output JSON if requested
    if args.output is not None:
        text = json.dumps(profile, indent=2)
        args.output.write_text(text)
        print(f"wrote benchmark result to {args.output}")
    elif args.output_csv is None:
        # Default: print JSON to stdout
        text = json.dumps(profile, indent=2)
        print(text)

    # Output CSV if requested
    if args.output_csv is not None:
        fieldnames = [
            "timestamp",
            "backend",
            "fit_backend",
            "native_backend",
            "n_poems",
            "dim",
            "m_rated",
            "block_size",
            "scalapack_nprocs",
            "scalapack_block_size",
            "fit_seconds",
            "optimize_seconds",
            "score_seconds",
            "select_seconds",
            "total_seconds",
            "length_scale",
            "variance",
            "noise",
            "log_marginal_likelihood",
            "optimize_hyperparameters",
            "optimization_success",
            "used_optimized_params",
        ]

        csv_row = {
            "timestamp": datetime.now().isoformat(),
            "backend": profile["backend"],
            "fit_backend": profile["fit_backend"],
            "native_backend": profile.get("native_backend", ""),
            "n_poems": profile["n_poems"],
            "dim": profile["dim"],
            "m_rated": profile["m_rated"],
            "block_size": profile["block_size"],
            "scalapack_nprocs": profile.get("scalapack_nprocs", ""),
            "scalapack_block_size": profile.get("scalapack_block_size", ""),
            "fit_seconds": profile["fit_seconds"],
            "optimize_seconds": profile["optimize_seconds"],
            "score_seconds": profile["score_seconds"],
            "select_seconds": profile["select_seconds"],
            "total_seconds": profile["total_seconds"],
            "length_scale": profile["length_scale"],
            "variance": profile["variance"],
            "noise": profile["noise"],
            "log_marginal_likelihood": profile.get("log_marginal_likelihood", ""),
            "optimize_hyperparameters": profile["optimize_hyperparameters"],
            "optimization_success": profile.get("optimization_success", ""),
            "used_optimized_params": profile.get("used_optimized_params", ""),
        }

        # Write or append to CSV
        file_exists = args.output_csv.exists()
        mode = "a" if args.append and file_exists else "w"

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w" or not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)

        print(f"appended benchmark result to {args.output_csv}" if args.append else f"wrote benchmark result to {args.output_csv}")


if __name__ == "__main__":
    main()
