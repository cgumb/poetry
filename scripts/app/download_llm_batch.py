"""
Download results from a completed Claude Batch API job.

Usage:
  python scripts/app/download_llm_batch.py \\
    --batch-id batch_abc123... \\
    --output data/llm_batch_results.jsonl

Requires: anthropic>=0.34 and ANTHROPIC_API_KEY environment variable
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic package not found. Install with:")
    print("  pip install -r requirements-llm.txt")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", type=str, default=None,
                       help="Batch ID to download (or will look for *_batch_id.txt in data/)")
    parser.add_argument("--output", type=Path, default=Path("data/llm_batch_results.jsonl"),
                       help="Output JSONL file for results")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    return parser.parse_args()


def get_api_key(args_api_key: str | None) -> str:
    """Get API key from args or environment, prompting if necessary."""
    api_key = args_api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("\nNo API key found in environment variable ANTHROPIC_API_KEY")
        api_key = input("Enter your Anthropic API key: ").strip()

        if not api_key:
            print("Error: API key is required")
            sys.exit(1)

    return api_key


def find_batch_id_file() -> Path | None:
    """Look for a batch ID file in data/ directory."""
    data_dir = Path("data")
    if not data_dir.exists():
        return None

    batch_id_files = list(data_dir.glob("*_batch_id.txt"))
    if not batch_id_files:
        return None

    # Return most recent
    return max(batch_id_files, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()

    batch_id = args.batch_id

    # If no batch ID provided, try to find it
    if not batch_id:
        batch_id_file = find_batch_id_file()
        if batch_id_file:
            batch_id = batch_id_file.read_text().strip()
            print(f"Using batch ID from {batch_id_file}")
        else:
            print("Error: No batch ID provided and no *_batch_id.txt file found in data/")
            print("Usage: python scripts/app/download_llm_batch.py --batch-id batch_abc123...")
            sys.exit(1)

    # Get API key
    api_key = get_api_key(args.api_key)

    # Initialize client
    client = Anthropic(api_key=api_key)

    print(f"Downloading results for batch: {batch_id}")

    try:
        # Check batch status first
        batch = client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            print(f"\nWarning: Batch status is '{batch.processing_status}', not 'ended'")
            print(f"Results may be incomplete.")

            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != "y":
                print("Aborted.")
                sys.exit(0)

        print(f"Batch status: {batch.processing_status}")
        print(f"Succeeded: {batch.request_counts.succeeded}")
        print(f"Errored: {batch.request_counts.errored}")

        # Download results
        print(f"\nDownloading results...")

        # Get results URL and download
        # The Anthropic SDK provides results as an iterator
        results = client.messages.batches.results(batch_id)

        # Write to output file
        args.output.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with args.output.open("w") as f:
            for result in results:
                # The result is already in the correct format
                import json
                f.write(json.dumps(result.model_dump()) + "\n")
                count += 1

        print(f"\n✓ Downloaded {count} results to {args.output}")

        print(f"\nNext step: Apply results to dataframe with:")
        print(f"  python scripts/app/apply_llm_imputation_results.py \\")
        print(f"    --poems data/poems_imputed.parquet \\")
        print(f"    --results {args.output} \\")
        print(f"    --output data/poems_imputed.parquet")

    except Exception as e:
        print(f"\nError downloading batch results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
