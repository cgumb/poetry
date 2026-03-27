from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TEXT_CANDIDATES = ["text", "poem", "content", "body"]
TITLE_CANDIDATES = ["title", "poem_title", "name"]
POET_CANDIDATES = ["poet", "author", "poet_name"]
ID_CANDIDATES = ["id", "poem_id"]


@dataclass(frozen=True)
class CanonicalColumns:
    id_col: str
    title_col: str
    poet_col: str
    text_col: str


def _pick_column(df: pd.DataFrame, candidates: list[str], fallback: str | None = None) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not find any of {candidates}. Available columns: {list(df.columns)}")


def detect_columns(df: pd.DataFrame) -> CanonicalColumns:
    text_col = _pick_column(df, TEXT_CANDIDATES)
    title_col = _pick_column(df, TITLE_CANDIDATES, fallback="_missing_title")
    poet_col = _pick_column(df, POET_CANDIDATES, fallback="_missing_poet")
    id_col = _pick_column(df, ID_CANDIDATES, fallback="_row_id")
    return CanonicalColumns(id_col=id_col, title_col=title_col, poet_col=poet_col, text_col=text_col)


def canonicalize_poems(df: pd.DataFrame) -> tuple[pd.DataFrame, CanonicalColumns]:
    df = df.copy()
    cols = detect_columns(df)

    if cols.id_col == "_row_id":
        df["_row_id"] = range(len(df))
    if cols.title_col == "_missing_title":
        df["_missing_title"] = ""
    if cols.poet_col == "_missing_poet":
        df["_missing_poet"] = ""

    cols = CanonicalColumns(
        id_col="_row_id" if cols.id_col == "_row_id" else cols.id_col,
        title_col="_missing_title" if cols.title_col == "_missing_title" else cols.title_col,
        poet_col="_missing_poet" if cols.poet_col == "_missing_poet" else cols.poet_col,
        text_col=cols.text_col,
    )

    out = pd.DataFrame(
        {
            "poem_id": df[cols.id_col],
            "title": df[cols.title_col].fillna("").astype(str),
            "poet": df[cols.poet_col].fillna("").astype(str),
            "text": df[cols.text_col].fillna("").astype(str),
        }
    )
    return out, cols
