from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.backends.naive import run_naive_step


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="naive,blocked")
    parser.add_argument("--fit-backends", default="python")
    parser.add_argument("--n-poems", default="1000,2000,5000")
    parser.add_argument("--m-rated", default="5,10,20,40")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--optimize-hyperparameters", action="store_true")
    parser.add_argument("--optimizer-maxiter", type=int, default=50)
    parser.add_argument("--scalapack-launcher", choices=["srun", "mpirun"], default="srun")
    parser.add_argument("--scalapack-nprocs", type=int, default=4)
    parser.add_argument("--scalapack-executable", default="native/build/scalapack_gp_fit")
    parser.add_argument("--scalapack-block-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/bench_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = parse_str_list(args.backends)
    fit_backends = parse_str_list(args.fit_backends)
    n_values = parse_int_list(args.n_poems)
    m_values = parse_int_list(args.m_rated)
    rng = np.random.default_rng(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "fit_backend",
        "native_backend",
        "n_poems",
        "m_rated",
        "dim",
        "block_size",
        "scalapack_nprocs",
        "scalapack_block_size",
        "optimize_hyperparameters",
        "optimizer_maxiter",
        "fit_seconds",
        "optimize_seconds",
        "score_seconds",
        "select_seconds",
        "total_seconds",
        "length_scale",
        "variance",
        "noise",
        "log_marginal_likelihood",
        "optimization_success",
        "used_optimized_params",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n in n_values:
            embeddings = rng.normal(size=(n, args.dim))
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            for m in m_values:
                if m >= n:
                    continue
                rated_indices = rng.choice(n, size=m, replace=False)
                ratings = rng.normal(size=m)
                for backend in backends:
                    candidate_fit_backends = fit_backends if backend == "blocked" else ["python"]
                    for fit_backend in candidate_fit_backends:
                        if backend == "naive":
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
                        elif backend == "blocked":
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
                                fit_backend=fit_backend,
                                scalapack_launcher=args.scalapack_launcher,
                                scalapack_nprocs=args.scalapack_nprocs,
                                scalapack_executable=args.scalapack_executable,
                                scalapack_block_size=args.scalapack_block_size,
                            )
                        else:
                            raise ValueError(f"Unknown backend: {backend}")
                        optimization_result = result.state.optimization_result or {}
                        row = {
                            "backend": backend,
                            "fit_backend": fit_backend,
                            "native_backend": optimization_result.get("native_backend"),
                            "n_poems": n,
                            "m_rated": m,
                            "dim": args.dim,
                            "block_size": args.block_size,
                            "scalapack_nprocs": args.scalapack_nprocs if fit_backend == "native_reference" else "",
                            "scalapack_block_size": args.scalapack_block_size if fit_backend == "native_reference" else "",
                            "optimize_hyperparameters": args.optimize_hyperparameters,
                            "optimizer_maxiter": args.optimizer_maxiter,
                            "fit_seconds": result.profile.fit_seconds,
                            "optimize_seconds": result.profile.optimize_seconds,
                            "score_seconds": result.profile.score_seconds,
                            "select_seconds": result.profile.select_seconds,
                            "total_seconds": result.profile.total_seconds,
                            "length_scale": result.state.length_scale,
                            "variance": result.state.variance,
                            "noise": result.state.noise,
                            "log_marginal_likelihood": result.state.log_marginal_likelihood,
                            "optimization_success": optimization_result.get("success"),
                            "used_optimized_params": optimization_result.get("used_optimized_params", False),
                        }
                        writer.writerow(row)
                        print(row)
    print(f"wrote sweep results to {args.output}")


if __name__ == "__main__":
    main()
