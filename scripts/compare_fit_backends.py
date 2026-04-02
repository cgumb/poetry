from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-poems", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--m-rated", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--scalapack-launcher", choices=["srun", "mpirun", "local"], default="srun")
    parser.add_argument("--scalapack-nprocs", type=int, default=4)
    parser.add_argument("--scalapack-executable", default="native/build/scalapack_gp_fit")
    parser.add_argument("--scalapack-block-size", type=int, default=128)
    parser.add_argument("--scalapack-native-backend", choices=["auto", "mpi", "scalapack"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _safe_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a - b) / denom)


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def main() -> None:
    args = parse_args()
    total_start = perf_counter()
    _log(args.verbose, "[compare] generating synthetic embeddings and ratings")
    rng = np.random.default_rng(args.seed)

    embeddings = rng.normal(size=(args.n_poems, args.dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    rated_indices = rng.choice(args.n_poems, size=args.m_rated, replace=False)
    ratings = rng.normal(size=args.m_rated)
    _log(args.verbose, f"[compare] data ready: n_poems={args.n_poems} dim={args.dim} m_rated={args.m_rated}")

    _log(args.verbose, "[compare] running blocked step with python fit backend")
    python_start = perf_counter()
    python_result = run_blocked_step(
        embeddings,
        rated_indices,
        ratings,
        length_scale=args.length_scale,
        variance=args.variance,
        noise=args.noise,
        block_size=args.block_size,
        fit_backend="python",
    )
    python_end = perf_counter()
    _log(args.verbose, f"[compare] python fit finished in {python_end - python_start:.3f}s (fit={python_result.profile.fit_seconds:.3f}s score={python_result.profile.score_seconds:.3f}s)")

    _log(args.verbose, "[compare] running blocked step with native fit backend")
    native_start = perf_counter()
    native_result = run_blocked_step(
        embeddings,
        rated_indices,
        ratings,
        length_scale=args.length_scale,
        variance=args.variance,
        noise=args.noise,
        block_size=args.block_size,
        fit_backend="native_reference",
        scalapack_launcher=args.scalapack_launcher,
        scalapack_nprocs=args.scalapack_nprocs,
        scalapack_executable=args.scalapack_executable,
        scalapack_block_size=args.scalapack_block_size,
        scalapack_native_backend=args.scalapack_native_backend,
        scalapack_verbose=args.verbose,
    )
    native_end = perf_counter()
    _log(args.verbose, f"[compare] native fit finished in {native_end - native_start:.3f}s (fit={native_result.profile.fit_seconds:.3f}s score={native_result.profile.score_seconds:.3f}s)")

    _log(args.verbose, "[compare] computing agreement metrics")
    mean_diff = python_result.mean - native_result.mean
    var_diff = python_result.variance - native_result.variance
    alpha_diff = python_result.state.alpha - native_result.state.alpha

    summary = {
        "n_poems": args.n_poems,
        "dim": args.dim,
        "m_rated": args.m_rated,
        "block_size": args.block_size,
        "length_scale": args.length_scale,
        "variance": args.variance,
        "noise": args.noise,
        "scalapack_nprocs": args.scalapack_nprocs,
        "scalapack_block_size": args.scalapack_block_size,
        "scalapack_native_backend": args.scalapack_native_backend,
        "python_fit_seconds": python_result.profile.fit_seconds,
        "native_fit_seconds": native_result.profile.fit_seconds,
        "python_score_seconds": python_result.profile.score_seconds,
        "native_score_seconds": native_result.profile.score_seconds,
        "max_abs_mean_diff": float(np.max(np.abs(mean_diff))),
        "max_abs_variance_diff": float(np.max(np.abs(var_diff))),
        "rel_mean_diff": _safe_rel_diff(python_result.mean, native_result.mean),
        "rel_variance_diff": _safe_rel_diff(python_result.variance, native_result.variance),
        "max_abs_alpha_diff": float(np.max(np.abs(alpha_diff))),
        "rel_alpha_diff": _safe_rel_diff(python_result.state.alpha, native_result.state.alpha),
        "log_marginal_likelihood_diff": float((python_result.state.log_marginal_likelihood or 0.0) - (native_result.state.log_marginal_likelihood or 0.0)),
        "exploit_index_equal": bool(python_result.exploit_index == native_result.exploit_index),
        "explore_index_equal": bool(python_result.explore_index == native_result.explore_index),
        "python_exploit_index": int(python_result.exploit_index),
        "native_exploit_index": int(native_result.exploit_index),
        "python_explore_index": int(python_result.explore_index),
        "native_explore_index": int(native_result.explore_index),
        "native_backend": (native_result.state.optimization_result or {}).get("native_backend"),
        "native_message": (native_result.state.optimization_result or {}).get("message"),
        "native_workdir": (native_result.state.optimization_result or {}).get("workdir"),
        "requested_native_backend": (native_result.state.optimization_result or {}).get("requested_native_backend"),
        "compiled_with_scalapack": (native_result.state.optimization_result or {}).get("compiled_with_scalapack"),
        "total_script_seconds": perf_counter() - total_start,
    }

    text = json.dumps(summary, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text)
        print(f"wrote backend agreement summary to {args.output}")


if __name__ == "__main__":
    main()
