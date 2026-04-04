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
    parser.add_argument("--use-similarity", action="store_true",
                       help="Enable embedding similarity imputation (disabled by default)")
    parser.add_argument("--similarity-threshold", type=float, default=0.95,
                       help="Cosine similarity threshold for confident poet inference")
    parser.add_argument("--min-known-poems", type=int, default=5,
                       help="Minimum known poems by a poet to use for inference")
    parser.add_argument("--generate-llm-batch", type=Path, default=None,
                       help="Generate LLM batch API request file at this path")
    parser.add_argument("--max-llm-requests", type=int, default=None,
                       help="Maximum number of LLM requests to generate (for cost control)")
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


def was_already_imputed(df: pd.DataFrame, idx: int, field: str) -> bool:
    """Check if a specific field was already imputed in a previous run."""
    imputed_col = f"{field}_imputed"
    if imputed_col not in df.columns:
        return False
    return bool(df.loc[idx, imputed_col])


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
) -> int:
    """
    Generate JSONL file for Claude batch API to identify poets and titles.
    Only includes rows that (1) need imputation and (2) haven't been imputed yet.

    Format: https://docs.anthropic.com/en/docs/build-with-claude/message-batching

    Returns: Number of requests generated
    """
    # Check which rows need imputation
    needs_poet = poems["poet"].map(is_missing_poet)
    needs_title = poems["title"].map(is_missing_title)

    # Exclude rows that were already imputed in a previous run
    if "poet_imputed" in poems.columns:
        already_imputed_poet = poems["poet_imputed"].fillna(False)
        needs_poet = needs_poet & ~already_imputed_poet

    if "title_imputed" in poems.columns:
        already_imputed_title = poems["title_imputed"].fillna(False)
        needs_title = needs_title & ~already_imputed_title

    # Only include rows where at least one field needs imputation
    needs_imputation = needs_poet | needs_title
    candidates = poems[needs_imputation].copy()

    if len(candidates) == 0:
        print("No poems need LLM imputation (all already imputed or complete).")
        return 0

    if max_poems is not None:
        candidates = candidates.head(max_poems)
        if len(candidates) < len(poems[needs_imputation]):
            print(f"Limiting to first {max_poems} of {len(poems[needs_imputation])} candidates (cost control)")

    requests = []
    for idx, row in candidates.iterrows():
        text_preview = str(row["text"])[:1000] if "text" in row else ""
        needs_poet_flag = needs_poet.loc[idx]
        needs_title_flag = needs_title.loc[idx]

        # Customize prompt based on what's needed
        if needs_poet_flag and needs_title_flag:
            task = "identify the poet and title"
            response_example = '{"poet": "Poet Name", "title": "Poem Title", "confidence": "high|medium|low"}'
        elif needs_poet_flag:
            task = "identify the poet"
            response_example = '{"poet": "Poet Name", "confidence": "high|medium|low"}'
        else:
            task = "identify the title"
            response_example = '{"title": "Poem Title", "confidence": "high|medium|low"}'

        prompt = f"""Given this poem excerpt, {task} if recognizable.

Poem excerpt:
{text_preview}

If you can confidently identify this poem, respond with JSON:
{response_example}

If unrecognizable or anonymous/traditional, respond with null values and low confidence.

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

    # More accurate cost estimate (tokens vary by poem length, ~150-200 per request)
    estimated_input_tokens = len(requests) * 175
    estimated_cost = (estimated_input_tokens / 1_000_000) * 0.25  # $0.25 per 1M input tokens
    estimated_output_tokens = len(requests) * 50
    estimated_cost += (estimated_output_tokens / 1_000_000) * 1.25  # $1.25 per 1M output tokens

    print(f"\n✓ Generated {len(requests)} batch API requests in {output_path}")
    print(f"  Estimated cost: ~${estimated_cost:.3f}")
    print(f"  (Based on ~{estimated_input_tokens:,} input + ~{estimated_output_tokens:,} output tokens)")
    print(f"\n  Submit with: anthropic messages batches create --input-file {output_path}")
    print(f"  See: https://docs.anthropic.com/en/docs/build-with-claude/message-batching")

    return len(requests)


def main() -> None:
    args = parse_args()

    poems = pd.read_parquet(args.poems)

    # Only load embeddings if similarity is requested
    if args.use_similarity:
        embeddings = np.load(args.embeddings, mmap_mode="r")
        if len(poems) != embeddings.shape[0]:
            raise ValueError("Poem and embedding row counts do not match")
    else:
        embeddings = None

    # Ensure required columns exist
    if "text" not in poems.columns:
        raise ValueError("poems.parquet must have 'text' column")
    if "poet" not in poems.columns:
        poems["poet"] = None
    if "title" not in poems.columns:
        poems["title"] = None

    # Initialize imputation tracking columns if they don't exist
    if "poet_imputed" not in poems.columns:
        poems["poet_imputed"] = False
    if "title_imputed" not in poems.columns:
        poems["title_imputed"] = False
    if "poet_imputation_confidence" not in poems.columns:
        poems["poet_imputation_confidence"] = 0.0

    print(f"Total poems: {len(poems)}")

    # Calculate missing counts excluding already-imputed rows
    missing_poets = poems["poet"].map(is_missing_poet) & ~poems["poet_imputed"]
    missing_titles = poems["title"].map(is_missing_title) & ~poems["title_imputed"]
    already_imputed_poets = poems["poet_imputed"].sum()
    already_imputed_titles = poems["title_imputed"].sum()

    print(f"Missing poets: {missing_poets.sum()} ({100*missing_poets.sum()/len(poems):.1f}%)")
    if already_imputed_poets > 0:
        print(f"  (Already imputed: {already_imputed_poets})")
    print(f"Missing titles: {missing_titles.sum()} ({100*missing_titles.sum()/len(poems):.1f}%)")
    if already_imputed_titles > 0:
        print(f"  (Already imputed: {already_imputed_titles})")

    # If user just wants to generate LLM batch file, do that and exit
    if args.generate_llm_batch is not None:
        print("\n=== Generating LLM Batch API Request File ===")
        n_requests = generate_llm_batch_input(poems, args.generate_llm_batch, args.max_llm_requests)
        if n_requests > 0:
            print(f"\n✓ Ready to submit {n_requests} requests to Claude Batch API")
            print(f"  This will only impute rows that haven't been imputed yet.")
        return

    # Strategy 1: First-line matching
    print("\n=== Strategy 1: First-line matching ===")
    first_line_groups = find_first_line_duplicates(poems)
    print(f"Found {len(first_line_groups)} duplicate first-line groups")

    # Only impute if not already imputed
    imputed_by_first_line = impute_from_first_line_matches(poems, first_line_groups)
    newly_imputed_first_line = 0
    for idx in poems.index:
        if not poems.loc[idx, "poet_imputed"] and imputed_by_first_line.iloc[idx] != poems.loc[idx, "poet"]:
            newly_imputed_first_line += 1
    print(f"Imputed {newly_imputed_first_line} poets via first-line matching")

    # Strategy 2: Embedding similarity (optional)
    imputed_poets = imputed_by_first_line.copy()
    confidence = poems["poet_imputation_confidence"].copy()

    if args.use_similarity:
        if embeddings is None:
            raise ValueError("--use-similarity requires embeddings to be loaded")

        print("\n=== Strategy 2: Embedding similarity ===")
        poet_names, centroids = compute_poet_centroids(poems, embeddings, args.min_known_poems)
        print(f"Computed centroids for {len(poet_names)} poets with ≥{args.min_known_poems} poems")

        poems_with_first_line = poems.copy()
        poems_with_first_line["poet"] = imputed_by_first_line

        imputed_poets, confidence = impute_poets_by_similarity(
            poems_with_first_line, embeddings, poet_names, centroids, args.similarity_threshold
        )

        # Only count new imputations (not already imputed)
        newly_imputed_similarity = 0
        for idx in poems.index:
            if not poems.loc[idx, "poet_imputed"] and imputed_poets.iloc[idx] != poems_with_first_line.loc[idx, "poet"] and confidence.iloc[idx] > 0:
                newly_imputed_similarity += 1
        print(f"Imputed {newly_imputed_similarity} poets via embedding similarity (threshold={args.similarity_threshold})")
    else:
        print("\n=== Strategy 2: Embedding similarity (SKIPPED) ===")
        print("Use --use-similarity to enable this strategy (not recommended)")

    # Statistics
    still_missing_mask = imputed_poets.map(is_missing_poet) & ~poems["poet_imputed"]
    still_missing = still_missing_mask.sum()
    total_imputed_this_run = missing_poets.sum() - still_missing

    print(f"\n=== Summary ===")
    print(f"Originally missing (not already imputed): {missing_poets.sum()}")
    print(f"Newly imputed this run: {total_imputed_this_run} ({100*total_imputed_this_run/max(missing_poets.sum(), 1):.1f}% of missing)")
    print(f"Still missing: {still_missing} ({100*still_missing/len(poems):.1f}% of total)")

    if still_missing > 0:
        print(f"\n=== Strategy 3: LLM Batch API (optional) ===")
        batch_file = args.output.parent / "llm_batch_requests.jsonl"
        print(f"For remaining {still_missing} poems, consider LLM batch processing.")
        print(f"Generate batch file with:")
        print(f"  python {Path(__file__).name} --poems {args.poems} --generate-llm-batch {batch_file}")
        if args.max_llm_requests:
            print(f"  (Use --max-llm-requests {args.max_llm_requests} to limit cost)")

    if not args.dry_run:
        output_poems = poems.copy()

        # Track which rows were newly imputed this run
        poet_was_missing = poems["poet"].map(is_missing_poet)
        poet_newly_imputed = (imputed_poets != poems["poet"]) & poet_was_missing & ~poems["poet_imputed"]

        # Update poet values only for newly imputed rows (preserve existing)
        for idx in output_poems.index:
            if poet_newly_imputed.iloc[idx]:
                output_poems.loc[idx, "poet"] = imputed_poets.iloc[idx]
                output_poems.loc[idx, "poet_imputed"] = True
                output_poems.loc[idx, "poet_imputation_confidence"] = confidence.iloc[idx]

        # Ensure columns exist (preserve existing if already present)
        if "poet_imputed" not in output_poems.columns:
            output_poems["poet_imputed"] = False
        if "title_imputed" not in output_poems.columns:
            output_poems["title_imputed"] = False
        if "poet_imputation_confidence" not in output_poems.columns:
            output_poems["poet_imputation_confidence"] = 0.0

        output_poems.to_parquet(args.output, index=False)
        print(f"\n✓ Wrote imputed poems to {args.output}")
        print(f"  poet_imputed=True (total): {output_poems['poet_imputed'].sum()} rows")
        print(f"  Newly imputed this run: {poet_newly_imputed.sum()} rows")
        if (confidence > 0).sum() > 0:
            print(f"  poet_imputation_confidence > 0: {(confidence > 0).sum()} rows")
    else:
        print("\n[DRY RUN] - No output written")
        print(f"Would newly impute {poet_newly_imputed.sum()} poets")


if __name__ == "__main__":
    main()
