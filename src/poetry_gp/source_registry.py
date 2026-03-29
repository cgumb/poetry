from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from poetry_gp.data_utils import canonicalize_poems


WHITESPACE_RE = re.compile(r"\s+")
BLANKLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class PoetrySourceSpec:
    name: str
    kind: str
    location: str
    split: str = "train"
    enabled: bool = True
    priority: int = 100
    language: str = "en"
    license_family: str = ""
    extra: dict[str, Any] | None = None


def _normalize_inline_text(text: str) -> str:
    out = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def normalize_poem_text(text: object) -> str:
    raw = _normalize_inline_text(str(text))
    lines = []
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = WHITESPACE_RE.sub(" ", line).strip()
        lines.append(stripped)
    normalized = "\n".join(lines).strip()
    normalized = BLANKLINE_RE.sub("\n\n", normalized)
    return normalized


def normalize_loose_text(text: object) -> str:
    normalized = normalize_poem_text(text).lower()
    normalized = normalized.replace("\n", " ")
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def normalize_metadata_text(text: object) -> str:
    normalized = _normalize_inline_text(str(text)).lower().strip()
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized


def stable_hash(text: object) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def load_source_manifest(path: Path) -> list[PoetrySourceSpec]:
    payload = json.loads(path.read_text())
    specs: list[PoetrySourceSpec] = []
    for item in payload:
        specs.append(
            PoetrySourceSpec(
                name=str(item["name"]),
                kind=str(item["kind"]),
                location=str(item["location"]),
                split=str(item.get("split", "train")),
                enabled=bool(item.get("enabled", True)),
                priority=int(item.get("priority", 100)),
                language=str(item.get("language", "en")),
                license_family=str(item.get("license_family", "")),
                extra=dict(item.get("extra", {})),
            )
        )
    return specs


def _load_huggingface_source(spec: PoetrySourceSpec, limit: int | None = None) -> pd.DataFrame:
    ds = load_dataset(spec.location, split=spec.split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds.to_pandas()


def _load_file_source(spec: PoetrySourceSpec, limit: int | None = None) -> pd.DataFrame:
    path = Path(spec.location)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file source suffix for {path}: {suffix}")
    if limit is not None:
        df = df.head(limit).copy()
    return df


def load_source_dataframe(spec: PoetrySourceSpec, limit: int | None = None) -> pd.DataFrame:
    if spec.kind == "huggingface":
        return _load_huggingface_source(spec, limit=limit)
    if spec.kind in {"parquet", "csv", "json", "jsonl", "ndjson"}:
        return _load_file_source(spec, limit=limit)
    raise ValueError(f"Unsupported source kind {spec.kind!r} for source {spec.name!r}")


def canonicalize_source_dataframe(df: pd.DataFrame, spec: PoetrySourceSpec) -> pd.DataFrame:
    canon, detected = canonicalize_poems(df)
    canon = canon.copy()
    canon["title"] = canon["title"].map(normalize_poem_text)
    canon["poet"] = canon["poet"].map(normalize_poem_text)
    canon["text"] = canon["text"].map(normalize_poem_text)

    raw_ids = canon["poem_id"].astype(str)
    canon["source_name"] = spec.name
    canon["source_kind"] = spec.kind
    canon["source_location"] = spec.location
    canon["source_split"] = spec.split
    canon["source_language"] = spec.language
    canon["license_family"] = spec.license_family
    canon["source_priority"] = spec.priority
    canon["source_row_id"] = raw_ids
    canon["poem_id"] = [f"{spec.name}:{rid}" for rid in raw_ids]

    canon["normalized_title"] = canon["title"].map(normalize_metadata_text)
    canon["normalized_poet"] = canon["poet"].map(normalize_metadata_text)
    canon["normalized_text"] = canon["text"].map(normalize_poem_text)
    canon["normalized_text_loose"] = canon["text"].map(normalize_loose_text)
    canon["text_hash"] = canon["normalized_text"].map(stable_hash)
    canon["text_hash_loose"] = canon["normalized_text_loose"].map(stable_hash)
    canon["title_poet_key"] = [
        stable_hash(f"{title}\n{poet}")
        for title, poet in zip(canon["normalized_title"], canon["normalized_poet"])
    ]
    canon["title_poet_text_key"] = [
        stable_hash(f"{title}\n{poet}\n{text_hash}")
        for title, poet, text_hash in zip(
            canon["normalized_title"], canon["normalized_poet"], canon["text_hash_loose"]
        )
    ]
    canon["text_len"] = canon["text"].str.len()
    canon["metadata_score"] = (
        canon["title"].str.len().gt(0).astype(int)
        + canon["poet"].str.len().gt(0).astype(int)
        + canon["text"].str.len().gt(0).astype(int)
    )
    canon.attrs["detected_columns"] = detected
    return canon


def combine_sources(
    specs: list[PoetrySourceSpec],
    *,
    selected_names: set[str] | None = None,
    per_source_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.enabled:
            continue
        if selected_names is not None and spec.name not in selected_names:
            continue
        raw = load_source_dataframe(spec, limit=per_source_limit)
        canon = canonicalize_source_dataframe(raw, spec)
        detected = canon.attrs.get("detected_columns")
        frames.append(canon)
        audit_rows.append(
            {
                "source_name": spec.name,
                "source_kind": spec.kind,
                "source_location": spec.location,
                "rows_loaded": int(len(raw)),
                "rows_canonical": int(len(canon)),
                "detected_columns": str(detected),
            }
        )
    if not frames:
        raise ValueError("No enabled sources were loaded. Check your manifest and --sources selection.")
    combined = pd.concat(frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    return combined, audit


def dedupe_canonical_poems(df: pd.DataFrame, *, min_chars: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    before_len = len(work)
    work = work[work["text"].str.len() >= min_chars].copy()
    work["passes_min_chars"] = True

    work = work.sort_values(
        by=["text_hash_loose", "source_priority", "metadata_score", "text_len", "poem_id"],
        ascending=[True, True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    work["dedupe_rank"] = work.groupby("text_hash_loose").cumcount()
    dedupe_groups = work.groupby("text_hash_loose")["poem_id"].transform("size")
    work["duplicate_group_size"] = dedupe_groups
    deduped = work[work["dedupe_rank"] == 0].copy().reset_index(drop=True)
    duplicates = work[work["dedupe_rank"] > 0].copy().reset_index(drop=True)

    summary_rows = [
        {"stage": "input_rows", "count": int(before_len)},
        {"stage": "after_min_chars", "count": int(len(work))},
        {"stage": "deduped_rows", "count": int(len(deduped))},
        {"stage": "removed_as_duplicates", "count": int(len(duplicates))},
    ]
    deduped.attrs["summary_rows"] = summary_rows
    return deduped, duplicates
