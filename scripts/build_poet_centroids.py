from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--output-parquet", type=Path, default=Path("data/poet_centroids.parquet"))
    parser.add_argument("--output-npy", type=Path, default=Path("data/poet_centroids.npy"))
    parser.add_argument("--min-poems", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poems = pd.read_parquet(args.poems)
    emb = np.load(args.embeddings)
    if len(poems) != emb.shape[0]:
        raise ValueError("poem and embedding row counts do not match")
    if "poet" not in poems.columns:
        raise ValueError("Expected canonical poems.parquet with a 'poet' column")

    work = poems.copy()
    work["row_index"] = np.arange(len(work))
    counts = work.groupby("poet", dropna=False).size().rename("n_poems").reset_index()
    counts = counts[(counts["poet"].astype(str).str.len() > 0) & (counts["n_poems"] >= args.min_poems)].reset_index(drop=True)

    rows = []
    centroid_list = []
    for poet, n_poems in counts[["poet", "n_poems"]].itertuples(index=False):
        idx = work.loc[work["poet"] == poet, "row_index"].to_numpy()
        centroid = emb[idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroid_list.append(centroid)
        rows.append({"poet": poet, "n_poems": int(n_poems)})

    out_df = pd.DataFrame(rows)
    out_emb = np.vstack(centroid_list) if centroid_list else np.zeros((0, emb.shape[1]), dtype=emb.dtype)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_parquet, index=False)
    np.save(args.output_npy, out_emb)
    print(f"wrote {len(out_df)} poet centroids to {args.output_parquet} and {args.output_npy}")


if __name__ == "__main__":
    main()
