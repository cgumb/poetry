from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from poetry_gp.backends.scalapack_fit import prepare_scalapack_fit_workdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--launcher", choices=["srun", "mpirun"], default="srun")
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--executable", default="native/build/scalapack_gp_fit")
    parser.add_argument("--workdir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n, 8))
    K = X @ X.T + 1e-2 * np.eye(args.n)
    y = rng.normal(size=args.n)
    prepared = prepare_scalapack_fit_workdir(
        K,
        y,
        launcher=args.launcher,
        nprocs=args.nprocs,
        executable=args.executable,
        block_size=args.block_size,
        workdir=args.workdir,
    )
    print(f"Prepared workdir: {prepared.workdir}")
    print("Run the scaffold with:")
    print(" ".join(prepared.command))
    print(f"Input metadata: {prepared.input_meta_path}")


if __name__ == "__main__":
    main()
