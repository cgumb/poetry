from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from poetry_gp.source_registry import (
    PoetrySourceSpec,
    combine_sources,
    dedupe_canonical_poems,
    load_source_manifest,
)


@dataclass(frozen=True)
class BuildCorpusResult:
    deduped: pd.DataFrame
    duplicates: pd.DataFrame
    duplicate_groups: pd.DataFrame
    audit: pd.DataFrame
    summary: pd.DataFrame


def parse_selected_names(spec: str) -> set[str] | None:
    cleaned = spec.strip()
    if cleaned.lower() == "all":
        return None
    names = {part.strip() for part in cleaned.split(",") if part.strip()}
    if not names:
        raise ValueError("--sources was provided but no source names were parsed.")
    return names


def build_duplicate_group_report(deduped: pd.DataFrame, duplicates: pd.DataFrame) -> pd.DataFrame:
    kept = deduped.copy()
    kept["kept_representative"] = True
    dropped = duplicates.copy()
    dropped["kept_representative"] = False
    combined = pd.concat([kept, dropped], ignore_index=True, sort=False)
    columns = [
        "text_hash_loose",
        "kept_representative",
        "dedupe_rank",
        "duplicate_group_size",
        "source_priority",
        "source_name",
        "source_kind",
        "source_location",
        "poem_id",
        "source_row_id",
        "title",
        "poet",
        "text_len",
        "license_family",
        "normalized_title",
        "normalized_poet",
        "text_hash",
        "title_poet_key",
        "title_poet_text_key",
    ]
    existing = list(dict.fromkeys(col for col in columns if col in combined.columns))
    report = combined[existing].sort_values(
        by=["text_hash_loose", "kept_representative", "source_priority", "poem_id"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    return report.reset_index(drop=True)


def build_poetry_corpus_from_specs(
    specs: list[PoetrySourceSpec],
    *,
    selected_names: set[str] | None = None,
    per_source_limit: int | None = None,
    min_chars: int = 80,
) -> BuildCorpusResult:
    combined, audit = combine_sources(
        specs,
        selected_names=selected_names,
        per_source_limit=per_source_limit,
    )
    deduped, duplicates = dedupe_canonical_poems(combined, min_chars=min_chars)
    duplicate_groups = build_duplicate_group_report(deduped, duplicates)
    summary = deduped.attrs.get("summary", pd.DataFrame())
    return BuildCorpusResult(
        deduped=deduped,
        duplicates=duplicates,
        duplicate_groups=duplicate_groups,
        audit=audit,
        summary=summary,
    )


def build_poetry_corpus_from_manifest(
    manifest: Path,
    *,
    selected_names: set[str] | None = None,
    per_source_limit: int | None = None,
    min_chars: int = 80,
) -> BuildCorpusResult:
    specs = load_source_manifest(manifest)
    return build_poetry_corpus_from_specs(
        specs,
        selected_names=selected_names,
        per_source_limit=per_source_limit,
        min_chars=min_chars,
    )
