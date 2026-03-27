from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.backends.naive import run_naive_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["naive", "blocked"], required=True)
    parser.add_argument("--n-poems", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--m-rated", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    embeddings = rng.normal(size=(args.n_poems, args.dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12

    rated_indices = rng.choice(args.n_poems, size=args.m_rated, replace=False)
    ratings = rng.normal(size=args.m_rated)

    if args.backend == "naive":
        result = run_naive_step(embeddings, rated_indices, ratings)
    else:
        result = run_blocked_step(
            embeddings,
            rated_indices,
            ratings,
            block_size=args.block_size,
        )

    profile = {
        "backend": args.backend,
        "n_poems": args.n_poems,
        "dim": args.dim,
        "m_rated": args.m_rated,
        "block_size": args.block_size,
        "fit_seconds": result.profile.fit_seconds,
        "score_seconds": result.profile.score_seconds,
        "select_seconds": result.profile.select_seconds,
        "total_seconds": result.profile.total_seconds,
        "exploit_index": result.exploit_index,
        "explore_index": result.explore_index,
    }

    text = json.dumps(profile, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text)
        print(f"wrote benchmark result to {args.output}")


if __name__ == "__main__":
    main()
