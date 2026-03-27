from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from poetry_gp.data_utils import canonicalize_poems


DEFAULT_DATASET = "DanFosing/public-domain-poetry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("data/poems.parquet"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    df = ds.to_pandas()
    canon, detected = canonicalize_poems(df)
    canon = canon.drop_duplicates(subset=["title", "poet", "text"]).reset_index(drop=True)
    canon = canon[canon["text"].str.len() >= args.min_chars].reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canon.to_parquet(args.output, index=False)
    print(f"detected source columns: {detected}")
    print(f"wrote {len(canon)} canonical poems to {args.output}")
    print("canonical columns:", list(canon.columns))
    print("non-empty poet rows:", int((canon["poet"].str.len() > 0).sum()))


if __name__ == "__main__":
    main()
