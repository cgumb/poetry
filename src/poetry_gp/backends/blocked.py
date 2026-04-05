from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from pathlib import Path

import numpy as np
from scipy.stats import norm

from ..gp_exact import GPState, fit_exact_gp, predict_block
from ..kernel import rbf_kernel
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


def _compute_spatial_variance_reduction_scores(
    embeddings: np.ndarray,
    variance_arr: np.ndarray,
    state: GPState,
    excluded_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute spatial variance reduction for each candidate.

    This measures how much total uncertainty would be reduced by observing each point,
    considering spatial correlation across ALL candidates.

    For a GP, observing y* at x* updates the posterior variance at another point x_i as:
        σ²_new(x_i) = σ²(x_i) - k(x_i, x*)² / (σ²(x*) + σ_n²)

    The total expected variance reduction from observing x* is:
        SVR(x*) = Σ_i k(x_i, x*)² / (σ²(x*) + σ_n²)

    This differs from max_variance (entropy reduction) by:
    - max_variance: maximizes log(σ²(x*)) - information-theoretic optimality
    - spatial_variance: maximizes trace reduction - spatial diversity

    Complexity: O(n² × d) for pairwise kernel + O(n²) for scoring
    This is expensive but gives more spatially diverse exploration.

    Args:
        embeddings: All candidate embeddings (n, d)
        variance_arr: Posterior variances for all candidates (n,)
        state: GP state with hyperparameters
        excluded_mask: Mask of points to exclude (n,)

    Returns:
        scores: Spatial variance reduction score for each candidate (n,)
    """
    n = embeddings.shape[0]

    # Compute pairwise kernel matrix K(candidates, candidates)
    # This is the expensive part: O(n² × d)
    k_cc = rbf_kernel(
        embeddings,
        embeddings,
        length_scale=state.length_scale,
        variance=state.variance,
    )

    # For each candidate x*, compute SVR(x*) = Σ_i k(x_i, x*)² / (σ²(x*) + σ_n²)
    spatial_variance_reduction = np.zeros(n, dtype=np.float64)
    noise_var = state.noise ** 2

    for i in range(n):
        if excluded_mask[i]:
            continue
        # Denominator: σ²(x*) + σ_n²
        denom = variance_arr[i] + noise_var
        if denom <= 0:
            continue
        # Numerator: Σ_j k(x_j, x*)²
        k_col = k_cc[:, i]
        spatial_variance_reduction[i] = np.sum(k_col ** 2) / denom

    return spatial_variance_reduction


def _compute_expected_improvement_scores(
    mean: np.ndarray,
    variance_arr: np.ndarray,
    best_observed: float,
    excluded_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute Expected Improvement (EI) for each candidate.

    EI balances exploitation (high mean) and exploration (high variance).
    It measures the expected amount by which f(x) exceeds the current best.

    For Gaussian posterior N(μ(x), σ²(x)):
        EI(x) = (μ(x) - f_best) * Φ(Z) + σ(x) * φ(Z)

    where:
        Z = (μ(x) - f_best) / σ(x)
        Φ = standard normal CDF
        φ = standard normal PDF

    This is a classic balanced acquisition function that:
    - Returns 0 if μ(x) ≤ f_best and σ(x) = 0 (no hope of improvement)
    - Increases with both μ(x) and σ(x)
    - Naturally trades off exploitation vs exploration

    Args:
        mean: Posterior means for all candidates (n,)
        variance_arr: Posterior variances for all candidates (n,)
        best_observed: Best rating observed so far
        excluded_mask: Mask of points to exclude (n,)

    Returns:
        ei_scores: Expected improvement for each candidate (n,)
    """
    n = len(mean)
    ei_scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if excluded_mask[i]:
            continue

        std = np.sqrt(variance_arr[i])
        if std <= 0:
            continue

        # Compute Z-score
        z = (mean[i] - best_observed) / std

        # EI = (μ - f_best) * Φ(Z) + σ * φ(Z)
        ei_scores[i] = (mean[i] - best_observed) * norm.cdf(z) + std * norm.pdf(z)

    return ei_scores


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
    score_backend: str = "python",  # "python", "daemon", "auto", "gpu", "none"
    exploitation_strategy: str = "max_mean",  # "max_mean", "ucb", "lcb", "thompson"
    exploration_strategy: str = "max_variance",  # "max_variance", "spatial_variance", "expected_improvement"
    ucb_beta: float = 2.0,  # Confidence parameter for UCB/LCB strategies
    compute_mean: bool = True,  # Set False to skip mean (only for explore-only workflows)
    compute_variance: bool = True,  # Set False to skip variance (exploit-only workflows)
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

    # Optimize: Skip variance/chol gathering if score_backend="none"
    # (no scoring = no need for variance or cholesky factor)
    if score_backend == "none":
        compute_variance = False
        compute_mean = False

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
        # Only gather outputs needed for subsequent scoring
        # return_alpha: Always True (needed for mean computation or to store in state)
        # return_chol: Only if variance will be computed
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
            return_alpha=True,  # Always needed
            return_chol=compute_variance,  # Only if variance will be computed
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
        mean = np.empty(n, dtype=np.float64) if compute_mean else np.empty(0, dtype=np.float64)
        variance_arr = np.empty(n, dtype=np.float64) if compute_variance else np.empty(0, dtype=np.float64)

        # Determine what to compute based on flags
        need_variance_for_scoring = compute_variance

        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            mu_block, var_block = predict_block(
                state,
                embeddings[start:stop],
                compute_variance=need_variance_for_scoring,
            )
            if compute_mean:
                mean[start:stop] = mu_block
            if compute_variance and var_block is not None:
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
        mean, variance_arr, score_seconds_inner = score_all_gpu(
            state, embeddings, block_size, compute_variance=compute_variance
        )
        if not compute_variance:
            variance_arr = np.empty(0, dtype=np.float64)
        if not compute_mean:
            # GPU always computes mean currently, but we can discard it
            mean = np.empty(0, dtype=np.float64)

    else:
        raise ValueError(f"Unknown score_backend: {score_backend}")

    score_end = perf_counter()

    select_start = perf_counter()
    if score_backend == "none":
        exploit_index = -1
        explore_index = -1
    else:
        # Exploitation: Use specified strategy
        if compute_mean:
            if exploitation_strategy == "max_mean":
                # Simple: pick point with highest posterior mean
                masked_mean = mean.copy()
                masked_mean[excluded_mask] = -np.inf
                exploit_index = int(np.argmax(masked_mean))

            elif exploitation_strategy == "ucb":
                # Upper Confidence Bound: μ(x) + β·σ(x)
                # Balances mean (exploitation) and uncertainty (exploration)
                if not compute_variance:
                    raise ValueError("UCB requires variance computation")
                ucb_scores = mean + ucb_beta * np.sqrt(variance_arr)
                ucb_scores[excluded_mask] = -np.inf
                exploit_index = int(np.argmax(ucb_scores))

            elif exploitation_strategy == "lcb":
                # Lower Confidence Bound: μ(x) - β·σ(x)
                # Pessimistic/conservative recommendation
                if not compute_variance:
                    raise ValueError("LCB requires variance computation")
                lcb_scores = mean - ucb_beta * np.sqrt(variance_arr)
                lcb_scores[excluded_mask] = -np.inf
                exploit_index = int(np.argmax(lcb_scores))

            elif exploitation_strategy == "thompson":
                # Thompson Sampling: sample from posterior, return max
                if not compute_variance:
                    raise ValueError("Thompson sampling requires variance computation")
                rng = np.random.default_rng()
                samples = rng.normal(mean, np.sqrt(variance_arr))
                samples[excluded_mask] = -np.inf
                exploit_index = int(np.argmax(samples))

            else:
                raise ValueError(f"Unknown exploitation_strategy: {exploitation_strategy}")
        else:
            exploit_index = -1

        # Exploration: Use specified strategy
        if compute_variance:
            if exploration_strategy == "max_variance":
                # Entropy reduction: pick point with highest posterior variance
                # Information-theoretically optimal for minimizing H(f|D)
                masked_var = variance_arr.copy()
                masked_var[excluded_mask] = -np.inf
                explore_index = int(np.argmax(masked_var))

            elif exploration_strategy == "spatial_variance":
                # Spatial variance reduction: pick point that reduces mean uncertainty most
                # Considers spatial correlation between candidates
                # Tends to give more diverse exploration than max_variance
                svr_scores = _compute_spatial_variance_reduction_scores(
                    embeddings, variance_arr, state, excluded_mask
                )
                explore_index = int(np.argmax(svr_scores))

            elif exploration_strategy == "expected_improvement":
                # Expected Improvement: balanced exploitation + exploration
                # Classic acquisition function from Bayesian optimization
                if not compute_mean:
                    raise ValueError("Expected Improvement requires mean computation")
                best_observed = float(np.max(ratings))
                ei_scores = _compute_expected_improvement_scores(
                    mean, variance_arr, best_observed, excluded_mask
                )
                explore_index = int(np.argmax(ei_scores))

            else:
                raise ValueError(f"Unknown exploration_strategy: {exploration_strategy}")
        else:
            explore_index = -1

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
