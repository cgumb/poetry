from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..gp_exact import GPState, fit_exact_gp, predict_block


@dataclass
class StepProfile:
    fit_seconds: float
    score_seconds: float
    select_seconds: float
    total_seconds: float


@dataclass
class NaiveStepResult:
    mean: np.ndarray
    variance: np.ndarray
    exploit_index: int
    explore_index: int
    profile: StepProfile


def run_naive_step(
    embeddings: np.ndarray,
    rated_indices: np.ndarray,
    ratings: np.ndarray,
    excluded_mask: np.ndarray | None = None,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-3,
) -> NaiveStepResult:
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
    state: GPState = fit_exact_gp(
        x_rated,
        ratings,
        length_scale=length_scale,
        variance=variance,
        noise=noise,
    )
    fit_end = perf_counter()

    n = embeddings.shape[0]
    mean = np.empty(n, dtype=np.float64)
    variance_arr = np.empty(n, dtype=np.float64)

    score_start = perf_counter()
    for i in range(n):
        mu_i, var_i = predict_block(state, embeddings[i : i + 1])
        mean[i] = mu_i[0]
        variance_arr[i] = var_i[0]
    score_end = perf_counter()

    select_start = perf_counter()
    masked_mean = mean.copy()
    masked_var = variance_arr.copy()
    masked_mean[excluded_mask] = -np.inf
    masked_var[excluded_mask] = -np.inf
    exploit_index = int(np.argmax(masked_mean))
    explore_index = int(np.argmax(masked_var))
    select_end = perf_counter()

    return NaiveStepResult(
        mean=mean,
        variance=variance_arr,
        exploit_index=exploit_index,
        explore_index=explore_index,
        profile=StepProfile(
            fit_seconds=fit_end - fit_start,
            score_seconds=score_end - score_start,
            select_seconds=select_end - select_start,
            total_seconds=select_end - t0,
        ),
    )
