from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poets", type=Path, default=Path("data/poet_centroids.parquet"))
    parser.add_argument("--coords", type=Path, default=Path("data/poet_centroids_2d.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/poet_map.png"))
    parser.add_argument("--topn", type=int, default=80)
    parser.add_argument("--label-topn", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poets = pd.read_parquet(args.poets)
    coords = np.load(args.coords)
    if len(poets) != coords.shape[0]:
        raise ValueError("poet metadata and coordinate row counts do not match")
    if len(poets) == 0:
        raise ValueError("No poet centroids found")

    poets = poets.sort_values("n_poems", ascending=False).head(args.topn).reset_index(drop=True)
    coords = coords[: len(poets)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sizes = 20 + 8 * np.sqrt(poets["n_poems"].to_numpy())
    ax.scatter(coords[:, 0], coords[:, 1], s=sizes, alpha=0.5)
    label_n = min(args.label_topn, len(poets))
    for i in range(label_n):
        ax.text(coords[i, 0], coords[i, 1], str(poets.iloc[i]["poet"]), fontsize=8)
    ax.set_title("Poet centroid map (PCA of centroid embeddings)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
