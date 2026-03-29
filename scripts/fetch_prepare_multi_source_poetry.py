from __future__ import annotations

import argparse
from pathlib import Path

from poetry_gp.source_registry import combine_sources, dedupe_canonical_poems, load_source_manifest


DEFAULT_MANIFEST = Path("configs/poetry_sources.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated list of source names from the manifest. Use 'all' to include every enabled source.",
    )
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to ingest from each source before canonicalization.",
    )
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/source_audit.parquet"),
        help="Where to write per-source load/canonicalization counts.",
    )
    parser.add_argument(
        "--duplicates-output",
        type=Path,
        default=Path("data/duplicate_poems.parquet"),
        help="Where to write the dropped duplicate rows for inspection.",
    )
    return parser.parse_args()


def parse_selected_names(spec: str) -> set[str] | None:
    cleaned = spec.strip()
    if cleaned.lower() == "all":
        return None
    names = {part.strip() for part in cleaned.split(",") if part.strip()}
    if not names:
        raise ValueError("--sources was provided but no source names were parsed.")
    return names


def main() -> None:
    args = parse_args()
    specs = load_source_manifest(args.manifest)
    selected_names = parse_selected_names(args.sources)

    combined, audit = combine_sources(
        specs,
        selected_names=selected_names,
        per_source_limit=args.per_source_limit,
    )
    deduped, duplicates = dedupe_canonical_poems(combined, min_chars=args.min_chars)
    summary = deduped.attrs.get("summary")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.duplicates_output.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_parquet(args.output, index=False)
    audit.to_parquet(args.audit_output, index=False)
    duplicates.to_parquet(args.duplicates_output, index=False)

    print(f"manifest={args.manifest}")
    print(f"sources={'all enabled sources' if selected_names is None else sorted(selected_names)}")
    print(f"wrote deduped poems to {args.output}")
    print(f"wrote source audit to {args.audit_output}")
    print(f"wrote dropped duplicates to {args.duplicates_output}")
    if summary is not None:
        for row in summary.to_dict(orient="records"):
            print(f"{row['stage']}: {row['count']}")
    print("rows per source after canonicalization:")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
