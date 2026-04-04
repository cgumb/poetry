"""
Impute missing poet names and titles using multiple strategies:
1. Embedding similarity (cheapest, works for poems similar to known works)
2. First-line matching (for well-known poems)
3. LLM batch API (for uncertain cases, more expensive)

Usage:
  python scripts/app/impute_missing_metadata.py --poems data/poems.parquet --embeddings data/embeddings.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--output", type=Path, default=Path("data/poems_imputed.parquet"))
    parser.add_argument("--similarity-threshold", type=float, default=0.95,
                       help="Cosine similarity threshold for confident poet inference")
    parser.add_argument("--min-known-poems", type=int, default=5,
                       help="Minimum known poems by a poet to use for inference")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be imputed without writing output")
    return parser.parse_args()


def is_missing_poet(value: object) -> bool:
    """Check if poet field is missing or placeholder."""
    if pd.isna(value):
        return True
    normalized = " ".join(str(value).strip().lower().split())
    bad_values = {"", "anonymous", "anon", "unknown", "unknown poet", "[unknown poet]",
                  "traditional", "various", "none", "n/a", "null"}
    return normalized in bad_values


def is_missing_title(value: object) -> bool:
    """Check if title field is missing or placeholder."""
    if pd.isna(value):
        return True
    normalized = " ".join(str(value).strip().lower().split())
    bad_values = {"", "untitled", "[untitled]", "none", "n/a", "null", "poem"}
    return normalized in bad_values


def compute_poet_centroids(
    poems: pd.DataFrame,
    embeddings: np.ndarray,
    min_poems: int = 5
) -> tuple[list[str], np.ndarray]:
    """Compute centroids for poets with sufficient known poems."""
    valid_mask = ~poems["poet"].map(is_missing_poet)
    valid_poems = poems[valid_mask].copy()
    valid_poems["row_index"] = poems.index[valid_mask]

    poet_counts = valid_poems.groupby("poet").size()
    qualified_poets = poet_counts[poet_counts >= min_poems].index.tolist()

    centroids = []
    poet_names = []

    for poet in qualified_poets:
        poet_mask = valid_poems["poet"] == poet
        poet_indices = valid_poems.loc[poet_mask, "row_index"].to_numpy()
        poet_embs = embeddings[poet_indices]
        centroid = poet_embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)
        poet_names.append(poet)

    centroid_matrix = np.vstack(centroids) if centroids else np.zeros((0, embeddings.shape[1]))
    return poet_names, centroid_matrix


def impute_poets_by_similarity(
    poems: pd.DataFrame,
    embeddings: np.ndarray,
    poet_names: list[str],
    centroids: np.ndarray,
    threshold: float = 0.95
) -> tuple[pd.Series, pd.Series]:
    """Impute missing poets using embedding similarity to poet centroids."""
    missing_mask = poems["poet"].map(is_missing_poet)
    missing_indices = poems.index[missing_mask].to_numpy()

    imputed_poets = poems["poet"].copy()
    confidence_scores = pd.Series(0.0, index=poems.index)

    if len(missing_indices) == 0 or len(centroids) == 0:
        return imputed_poets, confidence_scores

    missing_embs = embeddings[missing_indices]

    # Normalize embeddings
    norms = np.linalg.norm(missing_embs, axis=1, keepdims=True)
    missing_embs_normed = np.divide(missing_embs, norms, where=norms > 0,
                                    out=np.zeros_like(missing_embs))

    # Compute cosine similarity to all poet centroids
    similarities = missing_embs_normed @ centroids.T  # (n_missing, n_poets)
    best_poet_idx = similarities.argmax(axis=1)
    best_similarities = similarities[np.arange(len(similarities)), best_poet_idx]

    # Only impute if above threshold
    confident_mask = best_similarities >= threshold

    for i, idx in enumerate(missing_indices):
        if confident_mask[i]:
            imputed_poets.iloc[idx] = poet_names[best_poet_idx[i]]
            confidence_scores.iloc[idx] = best_similarities[i]

    return imputed_poets, confidence_scores


def find_first_line_duplicates(poems: pd.DataFrame) -> dict[str, list[int]]:
    """Find poems with identical first lines (may be the same poem)."""
    poems_with_text = poems.dropna(subset=["text"]).copy()
    poems_with_text["first_line"] = poems_with_text["text"].str.strip().str.split("\n").str[0]
    poems_with_text["first_line_norm"] = poems_with_text["first_line"].str.strip().str.lower()

    # Group by normalized first line
    first_line_groups = poems_with_text.groupby("first_line_norm").groups

    # Keep only groups with multiple poems
    duplicates = {line: list(indices) for line, indices in first_line_groups.items()
                  if len(indices) > 1}

    return duplicates


def impute_from_first_line_matches(
    poems: pd.DataFrame,
    first_line_groups: dict[str, list[int]]
) -> pd.Series:
    """If multiple poems have same first line, propagate known poet to unknowns."""
    imputed_poets = poems["poet"].copy()

    for indices in first_line_groups.values():
        group_poets = poems.loc[indices, "poet"]
        known_poets = group_poets[~group_poets.map(is_missing_poet)]

        if len(known_poets) > 0:
            # Use most common known poet
            most_common_poet = known_poets.mode()
            if len(most_common_poet) > 0:
                inferred_poet = most_common_poet.iloc[0]
                # Apply to all missing in this group
                for idx in indices:
                    if is_missing_poet(poems.loc[idx, "poet"]):
                        imputed_poets.iloc[idx] = inferred_poet

    return imputed_poets


def generate_llm_batch_input(
    poems: pd.DataFrame,
    output_path: Path,
    max_poems: int | None = None
) -> None:
    """
    Generate JSONL file for Claude batch API to identify poets and titles.

    Format: https://docs.anthropic.com/en/docs/build-with-claude/message-batching
    """
    missing_mask = poems["poet"].map(is_missing_poet) | poems["title"].map(is_missing_title)
    candidates = poems[missing_mask].copy()

    if max_poems is not None:
        candidates = candidates.head(max_poems)

    requests = []
    for idx, row in candidates.iterrows():
        text_preview = str(row["text"])[:1000] if "text" in row else ""

        prompt = f"""Given this poem excerpt, identify the poet and title if recognizable.

Poem excerpt:
{text_preview}

If you can confidently identify this poem, respond with JSON:
{{"poet": "Poet Name", "title": "Poem Title", "confidence": "high|medium|low"}}

If unrecognizable or anonymous/traditional, respond:
{{"poet": null, "title": null, "confidence": "low"}}

Respond only with valid JSON, no other text."""

        request = {
            "custom_id": f"poem_{idx}",
            "params": {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            }
        }
        requests.append(request)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"Generated {len(requests)} batch API requests in {output_path}")
    print(f"Estimated cost: ~${len(requests) * 0.00025:.2f} (at $0.25 per 1M input tokens)")


def main() -> None:
    args = parse_args()

    poems = pd.read_parquet(args.poems)
    embeddings = np.load(args.embeddings, mmap_mode="r")

    if len(poems) != embeddings.shape[0]:
        raise ValueError("Poem and embedding row counts do not match")

    # Ensure required columns exist
    if "text" not in poems.columns:
        raise ValueError("poems.parquet must have 'text' column")
    if "poet" not in poems.columns:
        poems["poet"] = None
    if "title" not in poems.columns:
        poems["title"] = None

    print(f"Total poems: {len(poems)}")
    missing_poets = poems["poet"].map(is_missing_poet).sum()
    missing_titles = poems["title"].map(is_missing_title).sum()
    print(f"Missing poets: {missing_poets} ({100*missing_poets/len(poems):.1f}%)")
    print(f"Missing titles: {missing_titles} ({100*missing_titles/len(poems):.1f}%)")

    # Strategy 1: First-line matching
    print("\n=== Strategy 1: First-line matching ===")
    first_line_groups = find_first_line_duplicates(poems)
    print(f"Found {len(first_line_groups)} duplicate first-line groups")
    imputed_by_first_line = impute_from_first_line_matches(poems, first_line_groups)
    n_first_line = (imputed_by_first_line != poems["poet"]).sum()
    print(f"Imputed {n_first_line} poets via first-line matching")

    # Strategy 2: Embedding similarity
    print("\n=== Strategy 2: Embedding similarity ===")
    poet_names, centroids = compute_poet_centroids(poems, embeddings, args.min_known_poems)
    print(f"Computed centroids for {len(poet_names)} poets with ≥{args.min_known_poems} poems")

    poems_with_first_line = poems.copy()
    poems_with_first_line["poet"] = imputed_by_first_line

    imputed_poets, confidence = impute_poets_by_similarity(
        poems_with_first_line, embeddings, poet_names, centroids, args.similarity_threshold
    )
    n_similarity = ((imputed_poets != poems_with_first_line["poet"]) & (confidence > 0)).sum()
    print(f"Imputed {n_similarity} poets via embedding similarity (threshold={args.similarity_threshold})")

    # Statistics
    still_missing = imputed_poets.map(is_missing_poet).sum()
    total_imputed = missing_poets - still_missing
    print(f"\n=== Summary ===")
    print(f"Originally missing: {missing_poets}")
    print(f"Imputed: {total_imputed} ({100*total_imputed/missing_poets:.1f}% of missing)")
    print(f"Still missing: {still_missing} ({100*still_missing/len(poems):.1f}% of total)")

    if still_missing > 0:
        print(f"\n=== Strategy 3: LLM Batch API (optional) ===")
        batch_file = args.output.parent / "llm_batch_requests.jsonl"
        print(f"For remaining {still_missing} poems, consider LLM batch processing.")
        print(f"Generate batch file with: --generate-llm-batch {batch_file}")

    if not args.dry_run:
        output_poems = poems.copy()

        # Track which rows were imputed
        poet_was_missing = poems["poet"].map(is_missing_poet)
        poet_is_imputed = (imputed_poets != poems["poet"]) & poet_was_missing

        output_poems["poet"] = imputed_poets
        output_poems["poet_imputed"] = poet_is_imputed
        output_poems["poet_imputation_confidence"] = confidence

        # Also add title_imputed column for future use (not yet implemented)
        output_poems["title_imputed"] = False

        output_poems.to_parquet(args.output, index=False)
        print(f"\nWrote imputed poems to {args.output}")
        print(f"  poet_imputed=True: {poet_is_imputed.sum()} rows")
        print(f"  poet_imputation_confidence > 0: {(confidence > 0).sum()} rows")
    else:
        print("\nDry run - no output written")


if __name__ == "__main__":
    main()
