"""
Native LAPACK backend via PyBind11 in-memory bridge.

Eliminates subprocess and file I/O overhead for small-to-medium problems (m < 5000).

Usage:
    from poetry_gp.backends.native_lapack import fit_exact_gp_native, is_native_available

    if is_native_available():
        state = fit_exact_gp_native(x_rated, y_rated, ...)
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..gp_exact import GPState
from ..kernel import rbf_kernel as rbf_kernel_python


def is_native_available() -> bool:
    """Check if poetry_gp_native module is available."""
    try:
        import poetry_gp_native  # noqa: F401
        return True
    except ImportError:
        return False


def fit_exact_gp_native(
    x_rated: np.ndarray,
    y_rated: np.ndarray,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-3,
    return_chol: bool = True,
    verbose: bool = False,
) -> GPState:
    """
    Fit GP using native LAPACK via PyBind11 (in-memory, zero overhead).

    Args:
        x_rated: (m × d) training features
        y_rated: (m,) training observations
        length_scale: RBF length scale
        variance: RBF variance (output scale)
        noise: Observation noise (added to diagonal)
        return_chol: Whether to return Cholesky factor (needed for variance)
        verbose: Print timing information

    Returns:
        GPState with fitted parameters

    Raises:
        ImportError: If poetry_gp_native module not available
        RuntimeError: If LAPACK factorization fails

    Complexity: O(m³) for Cholesky, O(m²) for solve

    Note: Suitable for m < 5000. For larger problems, use fit_backend="native_reference"
          (ScaLAPACK with MPI).
    """
    if not is_native_available():
        raise ImportError(
            "poetry_gp_native module not available. "
            "Build with: make native-build (requires PyBind11)"
        )

    import poetry_gp_native

    x_rated = np.asarray(x_rated, dtype=np.float64)
    y_rated = np.asarray(y_rated, dtype=np.float64)

    if x_rated.ndim != 2:
        raise ValueError("x_rated must be 2D")
    if y_rated.ndim != 1:
        raise ValueError("y_rated must be 1D")
    if x_rated.shape[0] != y_rated.shape[0]:
        raise ValueError("x_rated and y_rated length mismatch")
    if noise <= 0:
        raise ValueError("noise must be positive")

    m = x_rated.shape[0]

    t0 = perf_counter()

    # Compute kernel matrix: K_rr = rbf_kernel(x_rated, x_rated) + noise^2 * I
    K_rr = rbf_kernel_python(x_rated, x_rated, length_scale=length_scale, variance=variance)
    K_rr.flat[:: K_rr.shape[0] + 1] += noise * noise

    kernel_seconds = perf_counter() - t0

    # Fit using native LAPACK
    fit_start = perf_counter()
    result = poetry_gp_native.fit_gp_lapack(K_rr, y_rated, return_chol=return_chol)
    fit_seconds = perf_counter() - fit_start

    total_seconds = perf_counter() - t0

    if verbose:
        print(f"[native-lapack] m={m} kernel={kernel_seconds:.3f}s fit={fit_seconds:.3f}s total={total_seconds:.3f}s")

    # Compute log marginal likelihood
    alpha = result["alpha"]
    logdet = result["logdet"]
    lml = -0.5 * float(y_rated @ alpha) - 0.5 * float(logdet) - 0.5 * m * np.log(2.0 * np.pi)

    # Build GPState
    cho_factor_data = None
    if return_chol and "chol_lower" in result:
        chol_lower = result["chol_lower"]
        if np.any(chol_lower != 0):
            cho_factor_data = (chol_lower, True)  # (chol, lower=True)

    return GPState(
        x_rated=x_rated,
        y_rated=y_rated,
        alpha=alpha,
        cho_factor_data=cho_factor_data,
        length_scale=length_scale,
        variance=variance,
        noise=noise,
        log_marginal_likelihood=float(lml),
        optimization_result={
            "fit_backend": "native_lapack",
            "fit_total_seconds": total_seconds,
            "fit_kernel_seconds": kernel_seconds,
            "fit_cholesky_seconds": fit_seconds,
            "return_chol": return_chol,
        },
    )


def predict_native(
    state: GPState,
    x_query: np.ndarray,
    compute_variance: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Predict GP posterior using native LAPACK (in-memory, zero overhead).

    Args:
        state: Fitted GP state (from fit_exact_gp_native)
        x_query: (n × d) query points
        compute_variance: Whether to compute variance (expensive O(n × m²))

    Returns:
        (mean, variance) where variance is None if compute_variance=False

    Raises:
        ImportError: If poetry_gp_native module not available
        RuntimeError: If variance requested but Cholesky not available

    Complexity:
        Mean only: O(n × m × d) + O(n × m)
        With variance: O(n × m × d) + O(n × m²)
    """
    if not is_native_available():
        raise ImportError(
            "poetry_gp_native module not available. "
            "Build with: make native-build (requires PyBind11)"
        )

    import poetry_gp_native

    x_query = np.asarray(x_query, dtype=np.float64)

    if compute_variance and state.cho_factor_data is None:
        raise RuntimeError(
            "Cannot compute variance: GPState was created without Cholesky factor. "
            "Use return_chol=True when fitting, or set compute_variance=False."
        )

    # Prepare Cholesky factor (may be None if not needed)
    chol_lower = None
    if compute_variance:
        chol_lower = state.cho_factor_data[0]

    # Call native prediction
    result = poetry_gp_native.predict_gp_lapack(
        x_query,
        state.x_rated,
        state.alpha,
        chol_lower if chol_lower is not None else np.zeros((1, 1)),  # Dummy if not needed
        state.length_scale,
        state.variance,
        compute_variance=compute_variance,
    )

    mean = result["mean"]
    variance = result.get("variance", None)

    return mean, variance
