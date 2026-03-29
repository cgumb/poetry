from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from poetry_gp.reducer_2d import load_reducer, transform_with_reducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/poet_centroids.npy"))
    parser.add_argument("--output", type=Path, default=Path("data/poet_centroids_2d.npy"))
    parser.add_argument("--reducer", type=Path, default=Path("data/proj2d_reducer.pkl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = np.load(args.input)
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 0:
        z = np.zeros((0, 2), dtype=np.float64)
    else:
        reducer = load_reducer(args.reducer)
        z = transform_with_reducer(reducer, x)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, z)
    print(f"wrote 2D poet projection with shape {z.shape} to {args.output}")


if __name__ == "__main__":
    main()
