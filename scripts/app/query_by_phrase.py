from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--poet", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poems = pd.read_parquet(args.poems)
    emb = np.load(args.embeddings)
    if len(poems) != emb.shape[0]:
        raise ValueError("poem and embedding row counts do not match")
    if args.poet is not None and "poet" in poems.columns:
        mask = poems["poet"].astype(str).str.contains(args.poet, case=False, regex=False)
        poems = poems.loc[mask].reset_index(drop=True)
        emb = emb[mask.to_numpy()]
    if len(poems) == 0:
        raise ValueError("No poems left after filtering")

    model = SentenceTransformer(args.model)
    q = model.encode([args.text], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = emb @ q
    order = np.argsort(scores)[::-1][: args.topk]

    for rank, idx in enumerate(order, start=1):
        row = poems.iloc[int(idx)]
        title = row["title"] if "title" in poems.columns else ""
        poet = row["poet"] if "poet" in poems.columns else ""
        text = str(row["text"] if "text" in poems.columns else "")
        excerpt = text[:300].replace("\n", " ")
        print(f"\n#{rank} score={scores[idx]:.4f}")
        print(f"{title} — {poet}")
        print(excerpt + ("..." if len(text) > 300 else ""))


if __name__ == "__main__":
    main()
