from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--title", default=None)
    parser.add_argument("--poem-id", default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--exclude-self", action="store_true")
    return parser.parse_args()


def find_query_index(poems: pd.DataFrame, title: str | None, poem_id: str | None) -> int:
    if poem_id is not None:
        if "poem_id" not in poems.columns:
            raise ValueError("poem_id column not found in poems table")
        matches = poems.index[poems["poem_id"].astype(str) == str(poem_id)].tolist()
        if not matches:
            raise ValueError(f"No poem found with poem_id={poem_id}")
        return int(matches[0])
    if title is not None:
        if "title" not in poems.columns:
            raise ValueError("title column not found in poems table")
        mask = poems["title"].astype(str).str.contains(title, case=False, regex=False)
        matches = poems.index[mask].tolist()
        if not matches:
            raise ValueError(f"No poem found with title containing {title!r}")
        return int(matches[0])
    raise ValueError("Provide either --title or --poem-id")


def main() -> None:
    args = parse_args()
    poems = pd.read_parquet(args.poems)
    emb = np.load(args.embeddings)
    if len(poems) != emb.shape[0]:
        raise ValueError("poem and embedding row counts do not match")

    q_idx = find_query_index(poems, args.title, args.poem_id)
    q = emb[q_idx]
    scores = emb @ q
    if args.exclude_self:
        scores[q_idx] = -np.inf
    order = np.argsort(scores)[::-1][: args.topk]

    q_row = poems.iloc[q_idx]
    print(f"Query poem: [{q_idx}] {q_row.get('title', '')} — {q_row.get('poet', '')}")
    print(str(q_row.get("text", ""))[:300].replace("\n", " ") + "...\n")

    for rank, idx in enumerate(order, start=1):
        row = poems.iloc[int(idx)]
        text = str(row.get("text", ""))
        excerpt = text[:300].replace("\n", " ")
        print(f"#{rank} score={scores[idx]:.4f}")
        print(f"[{idx}] {row.get('title', '')} — {row.get('poet', '')}")
        print(excerpt + ("..." if len(text) > 300 else ""))
        print()


if __name__ == "__main__":
    main()
