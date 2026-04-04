"""
Apply LLM batch API imputation results to poems parquet file.

This script parses the JSONL output from Claude Batch API and applies
the poet/title imputations to the poems dataframe.

Usage:
  python scripts/app/apply_llm_imputation_results.py \\
    --poems data/poems_imputed.parquet \\
    --results data/llm_batch_results.jsonl \\
    --output data/poems_imputed.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, required=True,
                       help="Input poems parquet file")
    parser.add_argument("--results", type=Path, required=True,
                       help="LLM batch results JSONL file from Claude API")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output poems parquet file with applied imputations")
    parser.add_argument("--min-confidence", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Minimum confidence level to apply (default: medium)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be applied without writing output")
    return parser.parse_args()


def parse_llm_result(result_line: str) -> tuple[int | None, dict]:
    """
    Parse a single line from LLM batch results JSONL.

    Returns: (poem_index, parsed_data_dict)
    """
    try:
        result = json.loads(result_line)
    except json.JSONDecodeError:
        return None, {}

    # Extract custom_id to get poem index
    custom_id = result.get("custom_id", "")
    if not custom_id.startswith("poem_"):
        return None, {}

    try:
        poem_idx = int(custom_id.split("_")[1])
    except (IndexError, ValueError):
        return None, {}

    # Extract the response content
    response = result.get("result", {})
    if response.get("type") != "succeeded":
        return poem_idx, {"error": f"API error: {response.get('type')}"}

    message = response.get("message", {})
    content_blocks = message.get("content", [])

    if not content_blocks:
        return poem_idx, {"error": "No content in response"}

    # Parse the JSON from the text content
    text_content = content_blocks[0].get("text", "")
    try:
        parsed = json.loads(text_content)
        return poem_idx, parsed
    except json.JSONDecodeError:
        return poem_idx, {"error": f"Invalid JSON in response: {text_content}"}


def confidence_rank(conf_str: str) -> int:
    """Convert confidence string to numeric rank for comparison."""
    mapping = {"low": 0, "medium": 1, "high": 2}
    return mapping.get(str(conf_str).lower(), 0)


def apply_llm_imputations(
    poems: pd.DataFrame,
    results_path: Path,
    min_confidence: str,
    dry_run: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Apply LLM imputation results to poems dataframe.

    Returns: (updated_dataframe, statistics_dict)
    """
    # Initialize tracking columns if not present
    if "poet_imputed" not in poems.columns:
        poems["poet_imputed"] = False
    if "title_imputed" not in poems.columns:
        poems["title_imputed"] = False
    if "poet_imputation_confidence" not in poems.columns:
        poems["poet_imputation_confidence"] = 0.0
    if "title_imputation_confidence" not in poems.columns:
        poems["title_imputation_confidence"] = 0.0

    output_poems = poems.copy()

    min_conf_rank = confidence_rank(min_confidence)

    stats = {
        "total_results": 0,
        "parse_errors": 0,
        "skipped_already_imputed": 0,
        "skipped_low_confidence": 0,
        "applied_poet": 0,
        "applied_title": 0,
        "applied_both": 0,
    }

    # Parse results file
    with results_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            stats["total_results"] += 1
            poem_idx, parsed = parse_llm_result(line)

            if poem_idx is None:
                stats["parse_errors"] += 1
                continue

            if "error" in parsed:
                print(f"Warning: Poem {poem_idx}: {parsed['error']}")
                stats["parse_errors"] += 1
                continue

            # Check if row exists
            if poem_idx not in output_poems.index:
                print(f"Warning: Poem index {poem_idx} not found in dataframe")
                continue

            # Extract poet, title, confidence
            poet = parsed.get("poet")
            title = parsed.get("title")
            confidence = parsed.get("confidence", "low")

            # Check confidence threshold
            if confidence_rank(confidence) < min_conf_rank:
                stats["skipped_low_confidence"] += 1
                continue

            # Apply poet if provided and not already imputed
            poet_applied = False
            if poet and not pd.isna(poet) and poet.lower() not in {"null", "none", "unknown"}:
                if not output_poems.loc[poem_idx, "poet_imputed"]:
                    if not dry_run:
                        output_poems.loc[poem_idx, "poet"] = poet
                        output_poems.loc[poem_idx, "poet_imputed"] = True
                        output_poems.loc[poem_idx, "poet_imputation_confidence"] = confidence_rank(confidence) / 2.0
                    poet_applied = True
                    stats["applied_poet"] += 1
                else:
                    stats["skipped_already_imputed"] += 1

            # Apply title if provided and not already imputed
            title_applied = False
            if title and not pd.isna(title) and title.lower() not in {"null", "none", "untitled"}:
                if not output_poems.loc[poem_idx, "title_imputed"]:
                    if not dry_run:
                        output_poems.loc[poem_idx, "title"] = title
                        output_poems.loc[poem_idx, "title_imputed"] = True
                        output_poems.loc[poem_idx, "title_imputation_confidence"] = confidence_rank(confidence) / 2.0
                    title_applied = True
                    stats["applied_title"] += 1
                else:
                    stats["skipped_already_imputed"] += 1

            if poet_applied and title_applied:
                stats["applied_both"] += 1

    return output_poems, stats


def main() -> None:
    args = parse_args()

    if not args.poems.exists():
        raise FileNotFoundError(f"Poems file not found: {args.poems}")
    if not args.results.exists():
        raise FileNotFoundError(f"Results file not found: {args.results}")

    print(f"Loading poems from {args.poems}")
    poems = pd.read_parquet(args.poems)
    print(f"  Total poems: {len(poems):,}")

    print(f"\nApplying LLM imputation results from {args.results}")
    print(f"  Minimum confidence: {args.min_confidence}")

    output_poems, stats = apply_llm_imputations(
        poems, args.results, args.min_confidence, args.dry_run
    )

    # Print statistics
    print(f"\n=== Results ===")
    print(f"Total LLM results processed: {stats['total_results']}")
    print(f"Parse errors: {stats['parse_errors']}")
    print(f"Skipped (already imputed): {stats['skipped_already_imputed']}")
    print(f"Skipped (low confidence): {stats['skipped_low_confidence']}")
    print(f"\nApplied imputations:")
    print(f"  Poet only: {stats['applied_poet'] - stats['applied_both']}")
    print(f"  Title only: {stats['applied_title'] - stats['applied_both']}")
    print(f"  Both poet and title: {stats['applied_both']}")
    print(f"  Total poet imputations: {stats['applied_poet']}")
    print(f"  Total title imputations: {stats['applied_title']}")

    if not args.dry_run:
        print(f"\n✓ Writing updated poems to {args.output}")
        output_poems.to_parquet(args.output, index=False)

        # Final counts
        total_poet_imputed = output_poems["poet_imputed"].sum()
        total_title_imputed = output_poems["title_imputed"].sum()
        print(f"  Total rows with poet_imputed=True: {total_poet_imputed}")
        print(f"  Total rows with title_imputed=True: {total_title_imputed}")
    else:
        print("\n[DRY RUN] - No output written")


if __name__ == "__main__":
    main()
