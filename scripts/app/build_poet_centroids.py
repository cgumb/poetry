from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BAD_POET_NAMES = {
    "anonymous",
    "anon",
    "unknown",
    "unknown poet",
    "[unknown poet]",
    "traditional",
    "various",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--output-parquet", type=Path, default=Path("data/poet_centroids.parquet"))
    parser.add_argument("--output-npy", type=Path, default=Path("data/poet_centroids.npy"))
    parser.add_argument("--min-poems", type=int, default=3)
    return parser.parse_args()


def is_valid_poet_name(value: object) -> bool:
    name = " ".join(str(value).strip().lower().split())
    if not name:
        return False
    if name in BAD_POET_NAMES:
        return False
    if name.startswith("anonymous ") or name.startswith("unknown "):
        return False
    return True


def main() -> None:
    args = parse_args()
    poems = pd.read_parquet(args.poems, columns=["poet"])
    emb = np.load(args.embeddings, mmap_mode="r")
    if len(poems) != emb.shape[0]:
        raise ValueError("poem and embedding row counts do not match")
    if "poet" not in poems.columns:
        raise ValueError("Expected canonical poems.parquet with a 'poet' column")

    work = poems.copy()
    work["row_index"] = np.arange(len(work), dtype=np.int64)
    counts = work.groupby("poet", dropna=False).size().rename("n_poems").reset_index()
    counts = counts[counts["poet"].map(is_valid_poet_name)].reset_index(drop=True)
    counts = counts[counts["n_poems"] >= args.min_poems].reset_index(drop=True)

    poet_to_rows: dict[object, list[int]] = {}
    for poet, row_index in work[["poet", "row_index"]].itertuples(index=False):
        poet_to_rows.setdefault(poet, []).append(int(row_index))

    rows = []
    centroid_list = []
    for poet, n_poems in counts[["poet", "n_poems"]].itertuples(index=False):
        idx = np.asarray(poet_to_rows[poet], dtype=np.int64)
        centroid = np.asarray(emb[idx], dtype=np.float32).mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroid_list.append(centroid.astype(np.float32, copy=False))
        rows.append({"poet": poet, "n_poems": int(n_poems)})

    out_df = pd.DataFrame(rows)
    out_emb = np.vstack(centroid_list) if centroid_list else np.zeros((0, emb.shape[1]), dtype=np.float32)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_parquet, index=False)
    np.save(args.output_npy, out_emb)
    print(f"wrote {len(out_df)} poet centroids to {args.output_parquet} and {args.output_npy}")


if __name__ == "__main__":
    main()
