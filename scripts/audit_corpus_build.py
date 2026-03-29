from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


DEFAULT_AUDIT = Path("data/source_audit.parquet")
DEFAULT_DUPLICATE_GROUPS = Path("data/duplicate_groups.parquet")
DEFAULT_DEDUPED = Path("data/poems.parquet")
DEFAULT_SOURCE_SUMMARY = Path("data/source_contribution_audit.parquet")
DEFAULT_OVERLAP = Path("data/source_overlap_audit.parquet")
DEFAULT_WINLOSS = Path("data/source_winloss_audit.parquet")
DEFAULT_MARKDOWN = Path("data/corpus_audit_summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-input", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--duplicate-groups-input", type=Path, default=DEFAULT_DUPLICATE_GROUPS)
    parser.add_argument("--deduped-input", type=Path, default=DEFAULT_DEDUPED)
    parser.add_argument("--source-summary-output", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--overlap-output", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--winloss-output", type=Path, default=DEFAULT_WINLOSS)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace({0: pd.NA})
    return (num / denom).fillna(0.0)


def build_source_summary(audit: pd.DataFrame, duplicate_groups: pd.DataFrame) -> pd.DataFrame:
    kept = duplicate_groups[duplicate_groups["kept_representative"]].copy()
    dropped = duplicate_groups[~duplicate_groups["kept_representative"]].copy()

    kept_counts = kept.groupby("source_name").size().rename("kept_rows")
    dropped_counts = dropped.groupby("source_name").size().rename("dropped_rows")
    kept_groups = kept.groupby("source_name")["text_hash_loose"].nunique().rename("groups_won")
    dropped_groups = dropped.groupby("source_name")["text_hash_loose"].nunique().rename("groups_lost")

    source_group_counts = duplicate_groups.groupby("source_name")["text_hash_loose"].nunique().rename("groups_touched")
    conflict_mask = duplicate_groups.groupby("text_hash_loose")["source_name"].transform("nunique") > 1
    conflicts = duplicate_groups[conflict_mask].copy()
    conflict_groups = conflicts.groupby("source_name")["text_hash_loose"].nunique().rename("conflict_groups_touched")

    summary = audit.copy()
    for series in [kept_counts, dropped_counts, kept_groups, dropped_groups, source_group_counts, conflict_groups]:
        summary = summary.merge(series, how="left", left_on="source_name", right_index=True)

    fill_zero_cols = [
        "kept_rows",
        "dropped_rows",
        "groups_won",
        "groups_lost",
        "groups_touched",
        "conflict_groups_touched",
    ]
    summary[fill_zero_cols] = summary[fill_zero_cols].fillna(0).astype(int)
    summary["retention_rate_from_canonical"] = _safe_divide(summary["kept_rows"], summary["rows_canonical"])
    summary["drop_rate_from_canonical"] = _safe_divide(summary["dropped_rows"], summary["rows_canonical"])
    summary["conflict_group_rate"] = _safe_divide(summary["conflict_groups_touched"], summary["groups_touched"])
    summary = summary.sort_values(
        by=["kept_rows", "rows_canonical", "source_name"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return summary


def build_pairwise_overlap(duplicate_groups: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = duplicate_groups.groupby("text_hash_loose")
    for text_hash, group in grouped:
        sources = sorted(set(group["source_name"].astype(str)))
        if len(sources) < 2:
            continue
        for source_a, source_b in combinations(sources, 2):
            records.append(
                {
                    "text_hash_loose": text_hash,
                    "source_a": source_a,
                    "source_b": source_b,
                }
            )
    if not records:
        return pd.DataFrame(columns=["source_a", "source_b", "shared_duplicate_groups"])
    pairs = pd.DataFrame(records)
    overlap = (
        pairs.groupby(["source_a", "source_b"])
        .size()
        .rename("shared_duplicate_groups")
        .reset_index()
        .sort_values(by=["shared_duplicate_groups", "source_a", "source_b"], ascending=[False, True, True], kind="stable")
        .reset_index(drop=True)
    )
    return overlap


def build_winloss_table(duplicate_groups: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = duplicate_groups.groupby("text_hash_loose")
    for _, group in grouped:
        if group["source_name"].nunique() < 2:
            continue
        kept_rows = group[group["kept_representative"]]
        if kept_rows.empty:
            continue
        winner = str(kept_rows.iloc[0]["source_name"])
        losers = sorted(set(group.loc[~group["kept_representative"], "source_name"].astype(str)))
        for loser in losers:
            records.append({"winner_source": winner, "loser_source": loser})
    if not records:
        return pd.DataFrame(columns=["winner_source", "loser_source", "duplicate_groups_won"])
    wins = (
        pd.DataFrame(records)
        .groupby(["winner_source", "loser_source"])
        .size()
        .rename("duplicate_groups_won")
        .reset_index()
        .sort_values(by=["duplicate_groups_won", "winner_source", "loser_source"], ascending=[False, True, True], kind="stable")
        .reset_index(drop=True)
    )
    return wins


def build_markdown_report(
    source_summary: pd.DataFrame,
    overlap: pd.DataFrame,
    winloss: pd.DataFrame,
    deduped_count: int,
) -> str:
    lines: list[str] = []
    lines.append("# Corpus audit summary")
    lines.append("")
    lines.append(f"Final deduped corpus size: **{deduped_count:,}** poems")
    lines.append("")
    lines.append("## Source contribution summary")
    lines.append("")
    for _, row in source_summary.iterrows():
        lines.append(
            "- "
            f"`{row['source_name']}`: loaded {int(row['rows_loaded']):,}, canonical {int(row['rows_canonical']):,}, "
            f"kept {int(row['kept_rows']):,}, dropped {int(row['dropped_rows']):,}, "
            f"retention {row['retention_rate_from_canonical']:.1%}"
        )
    lines.append("")
    lines.append("## Strongest pairwise overlaps")
    lines.append("")
    if overlap.empty:
        lines.append("- No cross-source duplicate groups detected.")
    else:
        for _, row in overlap.head(10).iterrows():
            lines.append(
                f"- `{row['source_a']}` vs `{row['source_b']}`: {int(row['shared_duplicate_groups']):,} shared duplicate groups"
            )
    lines.append("")
    lines.append("## Duplicate-group winner/loser summary")
    lines.append("")
    if winloss.empty:
        lines.append("- No cross-source duplicate conflicts detected.")
    else:
        for _, row in winloss.head(10).iterrows():
            lines.append(
                f"- `{row['winner_source']}` beat `{row['loser_source']}` in {int(row['duplicate_groups_won']):,} duplicate groups"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    audit = pd.read_parquet(args.audit_input)
    duplicate_groups = pd.read_parquet(args.duplicate_groups_input)
    deduped = pd.read_parquet(args.deduped_input)

    source_summary = build_source_summary(audit, duplicate_groups)
    overlap = build_pairwise_overlap(duplicate_groups)
    winloss = build_winloss_table(duplicate_groups)
    markdown = build_markdown_report(source_summary, overlap, winloss, deduped_count=len(deduped))

    for path in [args.source_summary_output, args.overlap_output, args.winloss_output, args.markdown_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    source_summary.to_parquet(args.source_summary_output, index=False)
    overlap.to_parquet(args.overlap_output, index=False)
    winloss.to_parquet(args.winloss_output, index=False)
    args.markdown_output.write_text(markdown)

    print(f"wrote source summary to {args.source_summary_output}")
    print(f"wrote pairwise overlap audit to {args.overlap_output}")
    print(f"wrote win/loss audit to {args.winloss_output}")
    print(f"wrote markdown summary to {args.markdown_output}")
    print(markdown)


if __name__ == "__main__":
    main()
