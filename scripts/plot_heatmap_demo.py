from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from poetry_gp.heatmap import smooth_scalar_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("heatmap_demo.png"))
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    coords = rng.normal(size=(args.n, 2))
    values = np.sin(coords[:, 0]) + 0.5 * np.cos(2 * coords[:, 1])
    hm = smooth_scalar_field(coords, values)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(
        hm["zz"],
        extent=[hm["xs"][0], hm["xs"][-1], hm["ys"][0], hm["ys"][-1]],
        origin="lower",
        aspect="auto",
        alpha=0.75,
    )
    ax.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.3)
    ax.set_title("Smoothed scalar heatmap over 2D coordinates")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
