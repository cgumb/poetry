"""
Scoring backend with graceful fallback.

Provides unified interface for scoring that automatically tries:
1. MPI daemon (if available and launched) - parallel, fast
2. Python fallback (always available) - serial, slower
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..gp_exact import GPState, predict_block


def score_all_with_fallback(
    state: GPState,
    embeddings: np.ndarray,
    block_size: int = 2048,
    daemon_client: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Score all embeddings, using daemon if available, falling back to Python.

    Args:
        state: GP state from fit
        embeddings: All embeddings to score (n × d)
        block_size: Block size for chunked scoring
        daemon_client: Optional ScaLAPACKDaemonClient instance

    Returns:
        (mean, variance, score_seconds)
    """
    import time

    # Try daemon first if provided
    if daemon_client is not None:
        try:
            result = daemon_client.score_all(
                x_query=embeddings,
                x_rated=state.x_rated,
                alpha=state.alpha,
                L_factor=state.cho_factor_data[0],
                length_scale=state.length_scale,
                variance=state.variance,
            )
            return result["mean"], result["variance"], result["score_seconds"]
        except Exception as e:
            # Log the error but continue with fallback
            import sys
            print(f"[Warning] Daemon scoring failed: {e}", file=sys.stderr)
            print("[Warning] Falling back to Python scoring", file=sys.stderr)

    # Fallback: Python scoring (serial, but always works)
    start = time.time()
    n = embeddings.shape[0]
    mean = np.empty(n, dtype=np.float64)
    variance = np.empty(n, dtype=np.float64)

    for start_idx in range(0, n, block_size):
        stop_idx = min(start_idx + block_size, n)
        mu_block, var_block = predict_block(state, embeddings[start_idx:stop_idx])
        mean[start_idx:stop_idx] = mu_block
        variance[start_idx:stop_idx] = var_block

    elapsed = time.time() - start
    return mean, variance, elapsed


def try_create_daemon_client(
    nprocs: int = 4,
    launcher: str = "mpirun",
    verbose: bool = False,
) -> Any | None:
    """
    Try to create and start a daemon client.

    Returns None if:
    - MPI not available
    - Daemon executable not found
    - Daemon fails to start

    Args:
        nprocs: Number of MPI processes
        launcher: MPI launcher (mpirun, srun, etc.)
        verbose: Print status messages

    Returns:
        ScaLAPACKDaemonClient instance or None
    """
    try:
        from .scalapack_daemon_client import ScaLAPACKDaemonClient
        from pathlib import Path

        # Check if daemon executable exists
        daemon_exe = Path("native/build/scalapack_daemon")
        if not daemon_exe.exists():
            if verbose:
                print(f"[Info] Daemon not available: {daemon_exe} not found")
            return None

        # Try to start daemon
        client = ScaLAPACKDaemonClient(
            nprocs=nprocs,
            launcher=launcher,
            daemon_exe=daemon_exe,
        )
        client.start()

        if verbose:
            print(f"[Info] Started daemon with {nprocs} processes")

        return client

    except ImportError as e:
        if verbose:
            print(f"[Info] Daemon not available: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"[Info] Failed to start daemon: {e}")
        return None
