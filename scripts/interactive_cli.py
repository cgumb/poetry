from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from poetry_gp.backends.blocked import run_blocked_step


TEXT_CANDIDATES = ["text", "poem", "content", "body"]
TITLE_CANDIDATES = ["title", "poem_title", "name"]
POET_CANDIDATES = ["poet", "author", "poet_name"]


def pick_column(columns: list[str], candidates: list[str], fallback: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def show_poem(df: pd.DataFrame, idx: int, title_col: str, poet_col: str, text_col: str) -> None:
    row = df.iloc[idx]
    print("\n" + "=" * 80)
    print(f"[{idx}] {row[title_col]}")
    print(f"by {row[poet_col]}")
    print("-" * 80)
    print(str(row[text_col])[:4000])
    print("=" * 80)


def main() -> None:
    args = parse_args()
    poems = pd.read_parquet(args.poems)
    embeddings = np.load(args.embeddings)
    if len(poems) != embeddings.shape[0]:
        raise ValueError("poem and embedding row counts do not match")

    cols = list(poems.columns)
    text_col = pick_column(cols, TEXT_CANDIDATES, cols[-1])
    title_col = pick_column(cols, TITLE_CANDIDATES, cols[0])
    poet_col = pick_column(cols, POET_CANDIDATES, cols[0])

    rng = np.random.default_rng(args.seed)
    current_idx = int(rng.integers(len(poems)))
    rated_indices: list[int] = []
    ratings: list[float] = []

    while True:
        show_poem(poems, current_idx, title_col, poet_col, text_col)
        cmd = input("Rate [l]ike/[n]eutral/[d]islike, [e]xploit, e[x]plore, [q]uit: ").strip().lower()
        if cmd == "q":
            break
        if cmd in {"l", "n", "d"}:
            if current_idx not in rated_indices:
                rated_indices.append(current_idx)
                ratings.append({"l": 1.0, "n": 0.0, "d": -1.0}[cmd])
                print(f"Recorded rating {ratings[-1]} for poem {current_idx}")
            else:
                print("Poem already rated.")
            continue
        if cmd not in {"e", "x"}:
            print("Unknown command.")
            continue
        if not rated_indices:
            print("Rate at least one poem first.")
            continue
        result = run_blocked_step(
            embeddings,
            np.array(rated_indices, dtype=np.int64),
            np.array(ratings, dtype=np.float64),
        )
        current_idx = result.exploit_index if cmd == "e" else result.explore_index
        print(
            f"Timing: fit={result.profile.fit_seconds:.4f}s score={result.profile.score_seconds:.4f}s total={result.profile.total_seconds:.4f}s"
        )


if __name__ == "__main__":
    main()
