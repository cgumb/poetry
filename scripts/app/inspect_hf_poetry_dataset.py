from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET = "DanFosing/public-domain-poetry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset, split=args.split)
    lines = []
    lines.append(f"dataset={args.dataset}")
    lines.append(f"split={args.split}")
    lines.append(f"rows={len(ds)}")
    lines.append(f"columns={list(ds.column_names)}")
    sample_n = min(args.limit, len(ds))
    for i in range(sample_n):
        row = ds[i]
        lines.append(f"\n--- sample {i} ---")
        for k, v in row.items():
            text = str(v)
            if len(text) > 300:
                text = text[:300] + "..."
            lines.append(f"{k}: {text}")
    text = "\n".join(lines)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote inspection summary to {args.output}")


if __name__ == "__main__":
    main()
