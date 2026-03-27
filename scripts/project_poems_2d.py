from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import svd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--output", type=Path, default=Path("data/proj2d.npy"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = np.load(args.input)
    if args.limit is not None:
        x = x[: args.limit]
    x = x.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = svd(x, full_matrices=False, check_finite=False)
    z = u[:, :2] * s[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, z)
    print(f"wrote 2D projection with shape {z.shape} to {args.output}")


if __name__ == "__main__":
    main()
