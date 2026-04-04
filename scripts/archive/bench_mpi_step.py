from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from poetry_gp.backends.mpi import run_mpi_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-poems", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--m-rated", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    embeddings = rng.normal(size=(args.n_poems, args.dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    rated_indices = rng.choice(args.n_poems, size=args.m_rated, replace=False)
    ratings = rng.normal(size=args.m_rated)

    result = run_mpi_step(
        embeddings,
        rated_indices,
        ratings,
        block_size=args.block_size,
        comm=comm,
    )

    if rank == 0:
        payload = {
            "backend": "mpi",
            "ranks": comm.Get_size(),
            "n_poems": args.n_poems,
            "dim": args.dim,
            "m_rated": args.m_rated,
            "block_size": args.block_size,
            "fit_seconds": result.profile.fit_seconds,
            "broadcast_seconds": result.profile.broadcast_seconds,
            "local_score_seconds": result.profile.local_score_seconds,
            "reduce_seconds": result.profile.reduce_seconds,
            "total_seconds": result.profile.total_seconds,
            "exploit_index": result.exploit_index,
            "explore_index": result.explore_index,
        }
        text = json.dumps(payload, indent=2)
        if args.output is None:
            print(text)
        else:
            args.output.write_text(text)
            print(f"wrote benchmark result to {args.output}")


if __name__ == "__main__":
    main()
