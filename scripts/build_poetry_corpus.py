from __future__ import annotations

import argparse
from pathlib import Path

from poetry_gp.corpus_builder import build_poetry_corpus_from_manifest, parse_selected_names


DEFAULT_MANIFEST = Path("configs/poetry_sources.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated list of source names from the manifest. Use 'all' to include every enabled source.",
    )
    parser.add_argument("--per-source-limit", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/source_audit.parquet"))
    parser.add_argument("--duplicates-output", type=Path, default=Path("data/duplicate_poems.parquet"))
    parser.add_argument(
        "--duplicate-groups-output",
        type=Path,
        default=Path("data/duplicate_groups.parquet"),
        help="Detailed audit table containing both kept and dropped rows for each duplicate group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_names = parse_selected_names(args.sources)
    result = build_poetry_corpus_from_manifest(
        args.manifest,
        selected_names=selected_names,
        per_source_limit=args.per_source_limit,
        min_chars=args.min_chars,
    )

    for path in [args.output, args.audit_output, args.duplicates_output, args.duplicate_groups_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    result.deduped.to_parquet(args.output, index=False)
    result.audit.to_parquet(args.audit_output, index=False)
    result.duplicates.to_parquet(args.duplicates_output, index=False)
    result.duplicate_groups.to_parquet(args.duplicate_groups_output, index=False)

    print(f"manifest={args.manifest}")
    print(f"sources={'all enabled sources' if selected_names is None else sorted(selected_names)}")
    print(f"wrote deduped poems to {args.output}")
    print(f"wrote source audit to {args.audit_output}")
    print(f"wrote dropped duplicates to {args.duplicates_output}")
    print(f"wrote duplicate-group audit to {args.duplicate_groups_output}")
    for row in result.summary.to_dict(orient="records"):
        print(f"{row['stage']}: {row['count']}")
    print("rows per source after canonicalization:")
    print(result.audit.to_string(index=False))


if __name__ == "__main__":
    main()
