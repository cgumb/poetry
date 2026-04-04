from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from pathlib import Path

import numpy as np

from ..gp_exact import GPState, fit_exact_gp, predict_block
from .scalapack_fit import fit_exact_gp_scalapack_from_rated
from .scoring import score_all_with_fallback, try_create_daemon_client
from .gpu_scoring import score_all_gpu, is_gpu_available


@dataclass
class StepProfile:
    fit_seconds: float
    optimize_seconds: float
    score_seconds: float
    select_seconds: float
    total_seconds: float


@dataclass
class BlockedStepResult:
    mean: np.ndarray
    variance: np.ndarray
    exploit_index: int
    explore_index: int
    profile: StepProfile
    state: GPState


def run_blocked_step(
    embeddings: np.ndarray,
    rated_indices: np.ndarray,
    ratings: np.ndarray,
    excluded_mask: np.ndarray | None = None,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-3,
    block_size: int = 2048,
    optimize_hyperparameters: bool = False,
    optimizer_maxiter: int = 50,
    fit_backend: str = "python",
    score_backend: str = "python",  # "python", "daemon", "auto"
    daemon_client: object | None = None,  # Reusable daemon client
    daemon_nprocs: int = 4,
    daemon_launcher: str = "mpirun",
    scalapack_launcher: str = "srun",
    scalapack_nprocs: int = 4,
    scalapack_executable: str = "native/build/scalapack_gp_fit",
    scalapack_block_size: int = 128,
    scalapack_grid_rows: int | None = None,
    scalapack_grid_cols: int | None = None,
    scalapack_native_backend: str = "auto",
    scalapack_workdir: Path | None = None,
    scalapack_verbose: bool = False,
) -> BlockedStepResult:
    t0 = perf_counter()
    rated_indices = np.asarray(rated_indices, dtype=np.int64)
    ratings = np.asarray(ratings, dtype=np.float64)
    embeddings = np.asarray(embeddings, dtype=np.float64)

    if excluded_mask is None:
        excluded_mask = np.zeros(embeddings.shape[0], dtype=bool)
    else:
        excluded_mask = np.asarray(excluded_mask, dtype=bool).copy()
    excluded_mask[rated_indices] = True

    x_rated = embeddings[rated_indices]

    fit_start = perf_counter()
    if fit_backend == "python":
        state: GPState = fit_exact_gp(
            x_rated,
            ratings,
            length_scale=length_scale,
            variance=variance,
            noise=noise,
            optimize_hyperparameters=optimize_hyperparameters,
            optimizer_maxiter=optimizer_maxiter,
        )
    elif fit_backend == "native_reference":
        if optimize_hyperparameters:
            raise ValueError("native_reference fit backend does not support hyperparameter optimization yet")
        state = fit_exact_gp_scalapack_from_rated(
            x_rated,
            ratings,
            length_scale=length_scale,
            variance=variance,
            noise=noise,
            launcher=scalapack_launcher,
            nprocs=scalapack_nprocs,
            executable=scalapack_executable,
            block_size=scalapack_block_size,
            grid_rows=scalapack_grid_rows,
            grid_cols=scalapack_grid_cols,
            native_backend=scalapack_native_backend,
            workdir=scalapack_workdir,
            verbose=scalapack_verbose,
        )
    else:
        raise ValueError(f"Unknown fit_backend: {fit_backend}")
    fit_end = perf_counter()

    n = embeddings.shape[0]
    score_start = perf_counter()

    if score_backend == "none":
        mean = np.empty(0, dtype=np.float64)
        variance_arr = np.empty(0, dtype=np.float64)
        score_seconds_inner = 0.0

    elif score_backend == "python":
        # Pure Python scoring (serial)
        mean = np.empty(n, dtype=np.float64)
        variance_arr = np.empty(n, dtype=np.float64)
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            mu_block, var_block = predict_block(state, embeddings[start:stop])
            mean[start:stop] = mu_block
            variance_arr[start:stop] = var_block
        score_seconds_inner = 0.0

    elif score_backend == "daemon":
        # Daemon scoring (parallel) - fail if unavailable
        # If daemon_client was provided, reuse it; otherwise create one
        daemon_to_shutdown = None
        if daemon_client is None:
            daemon = try_create_daemon_client(nprocs=daemon_nprocs, launcher=daemon_launcher)
            if daemon is None:
                raise RuntimeError("Daemon scoring requested but daemon unavailable")
            daemon_to_shutdown = daemon
        else:
            daemon = daemon_client

        try:
            mean, variance_arr, score_seconds_inner = score_all_with_fallback(
                state, embeddings, block_size, daemon_client=daemon
            )
        finally:
            if daemon_to_shutdown is not None:
                daemon_to_shutdown.shutdown()

    elif score_backend == "auto":
        # Try daemon, fall back to Python automatically
        # If daemon_client was provided, reuse it; otherwise create one
        daemon_to_shutdown = None
        if daemon_client is None:
            daemon = try_create_daemon_client(nprocs=daemon_nprocs, launcher=daemon_launcher, verbose=True)
            daemon_to_shutdown = daemon
        else:
            daemon = daemon_client

        try:
            mean, variance_arr, score_seconds_inner = score_all_with_fallback(
                state, embeddings, block_size, daemon_client=daemon
            )
        finally:
            if daemon_to_shutdown is not None:
                daemon_to_shutdown.shutdown()

    elif score_backend == "gpu":
        # GPU scoring with CuPy
        if not is_gpu_available():
            raise RuntimeError("GPU scoring requested but GPU/CuPy not available")
        mean, variance_arr, score_seconds_inner = score_all_gpu(state, embeddings, block_size)

    else:
        raise ValueError(f"Unknown score_backend: {score_backend}")

    score_end = perf_counter()

    select_start = perf_counter()
    if score_backend == "none":
        exploit_index = -1
        explore_index = -1
    else:
        masked_mean = mean.copy()
        masked_var = variance_arr.copy()
        masked_mean[excluded_mask] = -np.inf
        masked_var[excluded_mask] = -np.inf
        exploit_index = int(np.argmax(masked_mean))
        explore_index = int(np.argmax(masked_var))
    select_end = perf_counter()

    optimize_seconds = 0.0
    if state.optimization_result is not None:
        optimize_seconds = float(state.optimization_result.get("optimize_seconds", 0.0))

    return BlockedStepResult(
        mean=mean,
        variance=variance_arr,
        exploit_index=exploit_index,
        explore_index=explore_index,
        profile=StepProfile(
            fit_seconds=fit_end - fit_start,
            optimize_seconds=optimize_seconds,
            score_seconds=score_end - score_start,
            select_seconds=select_end - select_start,
            total_seconds=select_end - t0,
        ),
        state=state,
    )
