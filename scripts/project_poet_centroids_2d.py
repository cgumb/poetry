from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import svd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/poet_centroids.npy"))
    parser.add_argument("--output", type=Path, default=Path("data/poet_centroids_2d.npy"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = np.load(args.input)
    x = x.astype(np.float64, copy=False)
    if x.shape[0] == 0:
        z = np.zeros((0, 2), dtype=np.float64)
    else:
        x = x - x.mean(axis=0, keepdims=True)
        u, s, _vt = svd(x, full_matrices=False, check_finite=False)
        if x.shape[0] == 1:
            z = np.zeros((1, 2), dtype=np.float64)
        else:
            z = u[:, :2] * s[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, z)
    print(f"wrote 2D poet projection with shape {z.shape} to {args.output}")


if __name__ == "__main__":
    main()
