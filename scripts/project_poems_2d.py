from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from poetry_gp.reducer_2d import default_umap_jobs, fit_umap_projection, save_reducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--output", type=Path, default=Path("data/proj2d.npy"))
    parser.add_argument("--reducer-output", type=Path, default=Path("data/proj2d_reducer.pkl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--n-jobs", type=int, default=default_umap_jobs())
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = np.load(args.input)
    if args.limit is not None:
        x = x[: args.limit]
    reducer, z = fit_umap_projection(
        x,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        deterministic=args.deterministic,
        n_jobs=args.n_jobs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, z)
    save_reducer(reducer, args.reducer_output)
    print(f"wrote UMAP projection with shape {z.shape} to {args.output}")
    print(f"wrote fitted reducer to {args.reducer_output}")
    print(
        "UMAP settings: "
        f"metric={args.metric} n_neighbors={args.n_neighbors} min_dist={args.min_dist} "
        f"deterministic={args.deterministic} n_jobs={args.n_jobs} dtype=float32"
    )


if __name__ == "__main__":
    main()
