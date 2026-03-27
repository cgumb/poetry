from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from mpi4py import MPI

from ..gp_exact import fit_exact_gp, predict_block


@dataclass
class MPIStepProfile:
    fit_seconds: float
    broadcast_seconds: float
    local_score_seconds: float
    reduce_seconds: float
    total_seconds: float


@dataclass
class MPIStepResult:
    exploit_index: int
    explore_index: int
    exploit_score: float
    explore_score: float
    profile: MPIStepProfile


def run_mpi_step(
    embeddings: np.ndarray,
    rated_indices: np.ndarray,
    ratings: np.ndarray,
    excluded_mask: np.ndarray | None = None,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-3,
    block_size: int = 2048,
    comm: MPI.Comm | None = None,
) -> MPIStepResult:
    comm = MPI.COMM_WORLD if comm is None else comm
    rank = comm.Get_rank()
    size = comm.Get_size()
    t0 = perf_counter()

    embeddings = np.asarray(embeddings, dtype=np.float64)
    rated_indices = np.asarray(rated_indices, dtype=np.int64)
    ratings = np.asarray(ratings, dtype=np.float64)

    if excluded_mask is None:
        excluded_mask = np.zeros(embeddings.shape[0], dtype=bool)
    else:
        excluded_mask = np.asarray(excluded_mask, dtype=bool).copy()
    excluded_mask[rated_indices] = True

    fit_start = perf_counter()
    if rank == 0:
        state = fit_exact_gp(
            embeddings[rated_indices],
            ratings,
            length_scale=length_scale,
            variance=variance,
            noise=noise,
        )
        payload = {
            "x_rated": state.x_rated,
            "alpha": state.alpha,
            "chol": state.cho_factor_data[0],
            "length_scale": state.length_scale,
            "variance": state.variance,
            "noise": state.noise,
        }
    else:
        payload = None
    fit_end = perf_counter()

    bcast_start = perf_counter()
    payload = comm.bcast(payload, root=0)
    bcast_end = perf_counter()

    n = embeddings.shape[0]
    counts = [(n + i) // size for i in range(size)]
    offsets = np.cumsum([0] + counts[:-1])
    start = offsets[rank]
    stop = start + counts[rank]

    local_embeddings = embeddings[start:stop]
    local_excluded = excluded_mask[start:stop]

    class _State:
        pass

    state = _State()
    state.x_rated = payload["x_rated"]
    state.alpha = payload["alpha"]
    state.cho_factor_data = (payload["chol"], True)
    state.length_scale = payload["length_scale"]
    state.variance = payload["variance"]
    state.noise = payload["noise"]

    local_score_start = perf_counter()
    best_mean = (-np.inf, -1)
    best_var = (-np.inf, -1)
    for block_start in range(0, local_embeddings.shape[0], block_size):
        block_stop = min(block_start + block_size, local_embeddings.shape[0])
        mu, var_arr = predict_block(state, local_embeddings[block_start:block_stop])
        valid = ~local_excluded[block_start:block_stop]
        if np.any(valid):
            valid_mu = np.where(valid, mu, -np.inf)
            valid_var = np.where(valid, var_arr, -np.inf)
            local_mean_idx = int(np.argmax(valid_mu))
            local_var_idx = int(np.argmax(valid_var))
            local_global_offset = start + block_start
            candidate_mean = (float(valid_mu[local_mean_idx]), local_global_offset + local_mean_idx)
            candidate_var = (float(valid_var[local_var_idx]), local_global_offset + local_var_idx)
            if candidate_mean[0] > best_mean[0]:
                best_mean = candidate_mean
            if candidate_var[0] > best_var[0]:
                best_var = candidate_var
    local_score_end = perf_counter()

    reduce_start = perf_counter()
    gathered_means = comm.gather(best_mean, root=0)
    gathered_vars = comm.gather(best_var, root=0)
    reduce_end = perf_counter()

    if rank == 0:
        exploit_score, exploit_index = max(gathered_means, key=lambda x: x[0])
        explore_score, explore_index = max(gathered_vars, key=lambda x: x[0])
    else:
        exploit_score, exploit_index = (-np.inf, -1)
        explore_score, explore_index = (-np.inf, -1)

    result = MPIStepResult(
        exploit_index=int(exploit_index),
        explore_index=int(explore_index),
        exploit_score=float(exploit_score),
        explore_score=float(explore_score),
        profile=MPIStepProfile(
            fit_seconds=fit_end - fit_start,
            broadcast_seconds=bcast_end - bcast_start,
            local_score_seconds=local_score_end - local_score_start,
            reduce_seconds=reduce_end - reduce_start,
            total_seconds=perf_counter() - t0,
        ),
    )
    return result
