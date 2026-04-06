"""
GPU-accelerated GP scoring using CuPy.

Provides fast posterior prediction using CUDA-enabled GPUs for large-scale
recommendation scenarios where m (rated points) is large.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular as solve_triangular_gpu
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    solve_triangular_gpu = None


def _pairwise_squared_distances_gpu(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
    """Compute pairwise squared distances on GPU."""
    x_norm = cp.sum(x * x, axis=1, keepdims=True)  # (n, 1)
    y_norm = cp.sum(y * y, axis=1, keepdims=True).T  # (1, m)
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return cp.maximum(d2, 0.0)


def _rbf_kernel_gpu(
    x: cp.ndarray,
    y: cp.ndarray,
    length_scale: float,
    variance: float
) -> cp.ndarray:
    """Compute RBF kernel on GPU."""
    d2 = _pairwise_squared_distances_gpu(x, y)
    return variance * cp.exp(-0.5 * d2 / (length_scale * length_scale))


def predict_block_gpu(
    x_rated_gpu: cp.ndarray,
    alpha_gpu: cp.ndarray,
    cho_lower_gpu: cp.ndarray,
    x_query: np.ndarray,
    length_scale: float,
    variance: float,
    compute_variance: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Predict mean and (optionally) variance for query points using GPU.

    Args:
        x_rated_gpu: Training points on GPU (m, d)
        alpha_gpu: GP weights on GPU (m,)
        cho_lower_gpu: Lower Cholesky factor on GPU (m, m)
        x_query: Query points on CPU (n_query, d)
        length_scale: RBF length scale
        variance: RBF variance
        compute_variance: Whether to compute variance (expensive O(n × m²))

    Returns:
        mean: Posterior mean on CPU (n_query,)
        var: Posterior variance on CPU (n_query,) or None if not computed
    """
    # Transfer query points to GPU
    x_query_gpu = cp.asarray(x_query, dtype=cp.float64)

    # Compute kernel matrix: K(x_query, x_rated)
    k_qr = _rbf_kernel_gpu(x_query_gpu, x_rated_gpu, length_scale, variance)

    # Compute mean: k_qr @ alpha
    mean_gpu = k_qr @ alpha_gpu
    mean = cp.asnumpy(mean_gpu)

    if not compute_variance:
        return mean, None

    # Compute variance: solve L @ V = K^T, then var = variance - ||V||^2
    v = solve_triangular_gpu(cho_lower_gpu, k_qr.T, lower=True)
    var_gpu = variance - cp.sum(v * v, axis=0)
    var_gpu = cp.maximum(var_gpu, 0.0)

    # Transfer results back to CPU
    var = cp.asnumpy(var_gpu)

    return mean, var


def score_all_gpu(
    state,
    embeddings: np.ndarray,
    block_size: int = 2048,
    compute_variance: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    Score all embeddings using GPU acceleration.

    Args:
        state: GPState with x_rated, alpha, cho_factor_data
        embeddings: All candidate embeddings (n, d)
        block_size: Process candidates in blocks to manage GPU memory
        compute_variance: Whether to compute variance (set False for exploit-only)

    Returns:
        mean: Posterior mean for all candidates (n,)
        variance: Posterior variance for all candidates (n,) or None if not computed
        gpu_seconds: Time spent on GPU computation (excluding initial transfer)
    """
    import time

    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available - cannot use GPU scoring")

    n = embeddings.shape[0]

    # Transfer GP state to GPU once (amortized cost)
    transfer_start = time.perf_counter()
    x_rated_gpu = cp.asarray(state.x_rated, dtype=cp.float64)
    alpha_gpu = cp.asarray(state.alpha, dtype=cp.float64)
    cho_lower_gpu = cp.asarray(np.tril(state.cho_factor_data[0]), dtype=cp.float64)
    transfer_end = time.perf_counter()

    # Allocate output arrays
    mean = np.empty(n, dtype=np.float64)
    variance = np.empty(n, dtype=np.float64) if compute_variance else None

    # Process in blocks to manage GPU memory
    compute_start = time.perf_counter()
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        mean_block, var_block = predict_block_gpu(
            x_rated_gpu,
            alpha_gpu,
            cho_lower_gpu,
            embeddings[start:stop],
            state.length_scale,
            state.variance,
            compute_variance=compute_variance,
        )
        mean[start:stop] = mean_block
        if compute_variance:
            variance[start:stop] = var_block

    # Ensure all GPU operations complete
    cp.cuda.Stream.null.synchronize()
    compute_end = time.perf_counter()

    # Report compute time (excluding initial state transfer)
    gpu_seconds = compute_end - compute_start

    return mean, variance, gpu_seconds


def compute_spatial_variance_reduction_gpu(
    embeddings: np.ndarray,
    variance_arr: np.ndarray,
    state,
    excluded_mask: np.ndarray,
    block_size: int = 1000,
) -> tuple[np.ndarray, float]:
    """
    Compute spatial variance reduction scores on GPU (exact, no approximation).

    For each candidate x*, computes:
        SVR(x*) = Σ_i k(x_i, x*)² / (σ²(x*) + σ_n²)

    This measures total uncertainty reduction across all candidates if we observe x*.

    Memory-efficient implementation:
    - Computes K(candidates, candidates) in blocks to avoid OOM
    - For n=85k: processes ~85 blocks of 1000 rows each
    - Each block: 85k × 1000 × 8 bytes = 680 MB (manageable)

    Expected speedup vs CPU:
    - n=85k: ~10-50× faster (8s → 160-800ms)

    Args:
        embeddings: All candidate embeddings (n, d)
        variance_arr: Posterior variances (n,)
        state: GP state with hyperparameters
        excluded_mask: Mask of points to exclude (n,)
        block_size: Rows to process per block (default 1000)

    Returns:
        svr_scores: Spatial variance reduction for each candidate (n,)
        gpu_seconds: Time spent on GPU computation
    """
    import time

    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available - cannot use GPU")

    n = embeddings.shape[0]
    noise_var = state.noise ** 2

    # Transfer data to GPU once
    transfer_start = time.perf_counter()
    embeddings_gpu = cp.asarray(embeddings, dtype=cp.float64)
    variance_gpu = cp.asarray(variance_arr, dtype=cp.float64)
    excluded_gpu = cp.asarray(excluded_mask, dtype=cp.bool_)
    transfer_end = time.perf_counter()

    # Allocate output on GPU
    svr_scores_gpu = cp.zeros(n, dtype=cp.float64)

    # Process in blocks to manage memory
    # Each block: compute K(block, all) and accumulate SVR scores
    compute_start = time.perf_counter()
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)

        # Compute kernel for this block: K(block, all)
        # Shape: (block_size, n)
        k_block = _rbf_kernel_gpu(
            embeddings_gpu[start:stop],
            embeddings_gpu,
            state.length_scale,
            state.variance,
        )

        # For each candidate i in block, compute SVR(i) = Σ_j k(j, i)² / (σ²(i) + σ_n²)
        # k_block[i-start, :] = k(i, all candidates)
        for i in range(start, stop):
            if excluded_gpu[i]:
                continue

            denom = variance_gpu[i] + noise_var
            if denom <= 0:
                continue

            # Get k(i, all): this is row (i-start) of k_block
            k_row = k_block[i - start, :]

            # SVR(i) = Σ_j k(i, j)² / denom
            svr_scores_gpu[i] = cp.sum(k_row ** 2) / denom

    # Ensure all GPU operations complete
    cp.cuda.Stream.null.synchronize()
    compute_end = time.perf_counter()

    # Transfer results back to CPU
    svr_scores = cp.asnumpy(svr_scores_gpu)

    gpu_seconds = compute_end - compute_start

    return svr_scores, gpu_seconds


def is_gpu_available() -> bool:
    """Check if GPU scoring is available."""
    if not CUPY_AVAILABLE:
        return False
    try:
        # Try to allocate a small array to verify CUDA is working
        test = cp.array([1.0])
        del test
        return True
    except Exception:
        return False
