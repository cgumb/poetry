from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.backends.naive import run_naive_step


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="naive,blocked")
    parser.add_argument("--n-poems", default="1000,2000,5000")
    parser.add_argument("--m-rated", default="5,10,20,40")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/bench_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = [x.strip() for x in args.backends.split(",") if x.strip()]
    n_values = parse_int_list(args.n_poems)
    m_values = parse_int_list(args.m_rated)
    rng = np.random.default_rng(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "n_poems",
        "m_rated",
        "dim",
        "block_size",
        "fit_seconds",
        "score_seconds",
        "select_seconds",
        "total_seconds",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n in n_values:
            embeddings = rng.normal(size=(n, args.dim))
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            for m in m_values:
                if m >= n:
                    continue
                rated_indices = rng.choice(n, size=m, replace=False)
                ratings = rng.normal(size=m)
                for backend in backends:
                    if backend == "naive":
                        result = run_naive_step(embeddings, rated_indices, ratings)
                    elif backend == "blocked":
                        result = run_blocked_step(embeddings, rated_indices, ratings, block_size=args.block_size)
                    else:
                        raise ValueError(f"Unknown backend: {backend}")
                    row = {
                        "backend": backend,
                        "n_poems": n,
                        "m_rated": m,
                        "dim": args.dim,
                        "block_size": args.block_size,
                        "fit_seconds": result.profile.fit_seconds,
                        "score_seconds": result.profile.score_seconds,
                        "select_seconds": result.profile.select_seconds,
                        "total_seconds": result.profile.total_seconds,
                    }
                    writer.writerow(row)
                    print(row)
    print(f"wrote sweep results to {args.output}")


if __name__ == "__main__":
    main()
