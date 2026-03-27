from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from poetry_gp.data_utils import canonicalize_poems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw_poems.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--min-chars", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    canon, detected = canonicalize_poems(df)
    canon = canon.drop_duplicates(subset=["title", "poet", "text"]).reset_index(drop=True)
    canon = canon[canon["text"].str.len() >= args.min_chars].reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canon.to_parquet(args.output, index=False)
    print(f"detected columns: {detected}")
    print(f"wrote {len(canon)} canonical poems to {args.output}")


if __name__ == "__main__":
    main()
