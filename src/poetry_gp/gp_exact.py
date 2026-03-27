from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .kernel import rbf_kernel


@dataclass
class GPState:
    x_rated: np.ndarray
    y_rated: np.ndarray
    alpha: np.ndarray
    cho_factor_data: tuple[np.ndarray, bool]
    length_scale: float
    variance: float
    noise: float


def fit_exact_gp(
    x_rated: np.ndarray,
    y_rated: np.ndarray,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-3,
) -> GPState:
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

    k_rr = rbf_kernel(x_rated, x_rated, length_scale=length_scale, variance=variance)
    k_rr.flat[:: k_rr.shape[0] + 1] += noise * noise
    c_and_lower = cho_factor(k_rr, lower=True, check_finite=False)
    alpha = cho_solve(c_and_lower, y_rated, check_finite=False)
    return GPState(
        x_rated=x_rated,
        y_rated=y_rated,
        alpha=alpha,
        cho_factor_data=c_and_lower,
        length_scale=length_scale,
        variance=variance,
        noise=noise,
    )


def predict_block(state: GPState, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_query = np.asarray(x_query, dtype=np.float64)
    k_qr = rbf_kernel(
        x_query,
        state.x_rated,
        length_scale=state.length_scale,
        variance=state.variance,
    )
    mean = k_qr @ state.alpha

    l_tri = np.tril(state.cho_factor_data[0])
    v = solve_triangular(l_tri, k_qr.T, lower=True, check_finite=False)
    var = state.variance - np.sum(v * v, axis=0)
    var = np.maximum(var, 0.0)
    return mean, var
