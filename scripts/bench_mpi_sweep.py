from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from mpi4py import MPI

from poetry_gp.backends.mpi import run_mpi_step


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-poems", default="5000,10000,20000")
    parser.add_argument("--m-rated", default="5,10,20,40")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/mpi_bench_results.csv"))
    return parser.parse_args()


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = parse_args()
    n_values = parse_int_list(args.n_poems)
    m_values = parse_int_list(args.m_rated)
    rng = np.random.default_rng(args.seed)

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        f = args.output.open("w", newline="")
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "ranks",
                "n_poems",
                "m_rated",
                "dim",
                "block_size",
                "fit_seconds",
                "broadcast_seconds",
                "local_score_seconds",
                "reduce_seconds",
                "total_seconds",
            ],
        )
        writer.writeheader()
    else:
        f = None
        writer = None

    for n in n_values:
        embeddings = rng.normal(size=(n, args.dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        for m in m_values:
            if m >= n:
                continue
            rated_indices = rng.choice(n, size=m, replace=False)
            ratings = rng.normal(size=m)
            result = run_mpi_step(
                embeddings,
                rated_indices,
                ratings,
                block_size=args.block_size,
                comm=comm,
            )
            if rank == 0:
                row = {
                    "backend": "mpi",
                    "ranks": size,
                    "n_poems": n,
                    "m_rated": m,
                    "dim": args.dim,
                    "block_size": args.block_size,
                    "fit_seconds": result.profile.fit_seconds,
                    "broadcast_seconds": result.profile.broadcast_seconds,
                    "local_score_seconds": result.profile.local_score_seconds,
                    "reduce_seconds": result.profile.reduce_seconds,
                    "total_seconds": result.profile.total_seconds,
                }
                writer.writerow(row)
                print(row)
            comm.Barrier()

    if rank == 0 and f is not None:
        f.close()
        print(f"wrote sweep results to {args.output}")


if __name__ == "__main__":
    main()
