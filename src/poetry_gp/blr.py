from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class BLRState:
    posterior_mean: np.ndarray
    posterior_cov: np.ndarray
    prior_precision: float
    noise_variance: float
    fit_intercept: bool


def _prepare_design_matrix(x: np.ndarray, fit_intercept: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if not fit_intercept:
        return x
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    return np.concatenate([ones, x], axis=1)


def fit_bayesian_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prior_precision: float = 1.0,
    noise_variance: float = 1.0,
    fit_intercept: bool = True,
) -> BLRState:
    if prior_precision <= 0:
        raise ValueError("prior_precision must be positive")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")
    design = _prepare_design_matrix(x, fit_intercept)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if design.shape[0] != y.shape[0]:
        raise ValueError("x and y row counts do not match")

    p = design.shape[1]
    posterior_precision = prior_precision * np.eye(p, dtype=np.float64) + (design.T @ design) / noise_variance
    rhs = (design.T @ y) / noise_variance
    cho = cho_factor(posterior_precision, lower=True, check_finite=False)
    posterior_mean = cho_solve(cho, rhs, check_finite=False)
    posterior_cov = cho_solve(cho, np.eye(p, dtype=np.float64), check_finite=False)
    return BLRState(
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        prior_precision=prior_precision,
        noise_variance=noise_variance,
        fit_intercept=fit_intercept,
    )


def predict_bayesian_linear_regression(state: BLRState, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = _prepare_design_matrix(x_query, state.fit_intercept)
    mean = design @ state.posterior_mean
    quad = np.sum((design @ state.posterior_cov) * design, axis=1)
    variance = state.noise_variance + quad
    variance = np.maximum(variance, 0.0)
    return mean, variance


def exploit_explore_indices(state: BLRState, x_query: np.ndarray, *, excluded_mask: np.ndarray | None = None) -> tuple[int, int, np.ndarray, np.ndarray]:
    mean, variance = predict_bayesian_linear_regression(state, x_query)
    if excluded_mask is None:
        excluded_mask = np.zeros(len(mean), dtype=bool)
    else:
        excluded_mask = np.asarray(excluded_mask, dtype=bool)
    masked_mean = mean.copy()
    masked_var = variance.copy()
    masked_mean[excluded_mask] = -np.inf
    masked_var[excluded_mask] = -np.inf
    exploit_index = int(np.argmax(masked_mean))
    explore_index = int(np.argmax(masked_var))
    return exploit_index, explore_index, mean, variance
