from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

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
    log_marginal_likelihood: float | None = None
    optimization_result: dict[str, object] | None = None


def _kernel_rr(
    x_rated: np.ndarray,
    *,
    length_scale: float,
    variance: float,
    noise: float,
) -> np.ndarray:
    k_rr = rbf_kernel(x_rated, x_rated, length_scale=length_scale, variance=variance)
    k_rr.flat[:: k_rr.shape[0] + 1] += noise * noise
    return k_rr


def _solve_gp_state(
    x_rated: np.ndarray,
    y_rated: np.ndarray,
    *,
    length_scale: float,
    variance: float,
    noise: float,
    optimization_result: dict[str, object] | None = None,
) -> GPState:
    k_rr = _kernel_rr(
        x_rated,
        length_scale=length_scale,
        variance=variance,
        noise=noise,
    )
    c_and_lower = cho_factor(k_rr, lower=True, check_finite=False)
    alpha = cho_solve(c_and_lower, y_rated, check_finite=False)
    l_tri = np.tril(c_and_lower[0])
    logdet = 2.0 * np.sum(np.log(np.diag(l_tri)))
    lml = -0.5 * float(y_rated @ alpha) - 0.5 * float(logdet) - 0.5 * len(y_rated) * np.log(2.0 * np.pi)
    return GPState(
        x_rated=x_rated,
        y_rated=y_rated,
        alpha=alpha,
        cho_factor_data=c_and_lower,
        length_scale=length_scale,
        variance=variance,
        noise=noise,
        log_marginal_likelihood=float(lml),
        optimization_result=optimization_result,
    )


def optimize_gp_hyperparameters(
    x_rated: np.ndarray,
    y_rated: np.ndarray,
    *,
    length_scale: float,
    variance: float,
    noise: float,
    optimizer_maxiter: int = 50,
) -> dict[str, object]:
    lower = np.asarray([1e-3, 1e-4, 1e-5], dtype=np.float64)
    upper = np.asarray([10.0, 10.0, 1.0], dtype=np.float64)

    init_params = np.asarray([length_scale, variance, noise], dtype=np.float64)
    init_params = np.clip(init_params, lower, upper)
    init_log = np.log(init_params)

    def objective(log_params: np.ndarray) -> float:
        params = np.exp(log_params)
        try:
            state = _solve_gp_state(
                x_rated,
                y_rated,
                length_scale=float(params[0]),
                variance=float(params[1]),
                noise=float(params[2]),
            )
            if state.log_marginal_likelihood is None:
                return 1e12
            return -float(state.log_marginal_likelihood)
        except (LinAlgError, ValueError, FloatingPointError):
            return 1e12

    init_objective = float(objective(init_log))
    optimize_start = perf_counter()
    result = minimize(
        objective,
        init_log,
        method="L-BFGS-B",
        bounds=[(float(np.log(lo)), float(np.log(hi))) for lo, hi in zip(lower, upper)],
        options={"maxiter": int(optimizer_maxiter)},
    )
    optimize_seconds = perf_counter() - optimize_start

    use_optimized = False
    chosen_log = init_log
    chosen_objective = init_objective
    if hasattr(result, "x") and np.all(np.isfinite(result.x)):
        candidate_log = np.asarray(result.x, dtype=np.float64)
        candidate_objective = float(objective(candidate_log))
        if candidate_objective <= init_objective or bool(getattr(result, "success", False)):
            chosen_log = candidate_log
            chosen_objective = candidate_objective
            use_optimized = True

    chosen_params = np.exp(chosen_log)
    return {
        "success": bool(getattr(result, "success", False)),
        "used_optimized_params": bool(use_optimized),
        "message": str(getattr(result, "message", "")),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "optimize_seconds": float(optimize_seconds),
        "length_scale": float(chosen_params[0]),
        "variance": float(chosen_params[1]),
        "noise": float(chosen_params[2]),
        "initial_log_marginal_likelihood": float(-init_objective),
        "final_log_marginal_likelihood": float(-chosen_objective),
    }


def fit_exact_gp(
    x_rated: np.ndarray,
    y_rated: np.ndarray,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-3,
    optimize_hyperparameters: bool = False,
    optimizer_maxiter: int = 50,
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

    optimization_result = None
    if optimize_hyperparameters and x_rated.shape[0] >= 2:
        optimization_result = optimize_gp_hyperparameters(
            x_rated,
            y_rated,
            length_scale=length_scale,
            variance=variance,
            noise=noise,
            optimizer_maxiter=optimizer_maxiter,
        )
        length_scale = float(optimization_result["length_scale"])
        variance = float(optimization_result["variance"])
        noise = float(optimization_result["noise"])

    return _solve_gp_state(
        x_rated,
        y_rated,
        length_scale=length_scale,
        variance=variance,
        noise=noise,
        optimization_result=optimization_result,
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
