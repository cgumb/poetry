from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DEFAULT_DATASET = "DanFosing/public-domain-poetry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/poems.parquet"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    df = ds.to_pandas()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")
    print("columns:", list(df.columns))


if __name__ == "__main__":
    main()
