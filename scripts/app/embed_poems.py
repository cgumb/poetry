from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_CANDIDATES = ["text", "poem", "content", "body"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def pick_text_column(df: pd.DataFrame) -> str:
    for col in TEXT_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a text column. Tried {TEXT_CANDIDATES}; found {list(df.columns)}")


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    text_col = pick_text_column(df)
    texts = df[text_col].fillna("").astype(str).tolist()

    model = SentenceTransformer(args.model)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, emb)
    print(f"wrote embeddings with shape {emb.shape} to {args.output}")


if __name__ == "__main__":
    main()
