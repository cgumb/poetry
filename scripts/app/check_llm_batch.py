"""
Check status of a Claude Batch API job.

Usage:
  python scripts/app/check_llm_batch.py --batch-id batch_abc123...

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
                       help="Batch ID to check (or will look for *_batch_id.txt in data/)")
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
            print("Usage: python scripts/app/check_llm_batch.py --batch-id batch_abc123...")
            sys.exit(1)

    # Get API key
    api_key = get_api_key(args.api_key)

    # Initialize client
    client = Anthropic(api_key=api_key)

    print(f"Checking status of batch: {batch_id}")

    try:
        batch = client.messages.batches.retrieve(batch_id)

        print(f"\n{'='*60}")
        print(f"Batch ID: {batch.id}")
        print(f"Status: {batch.processing_status}")
        print(f"Created: {batch.created_at}")
        print(f"Expires: {batch.expires_at}")
        print(f"{'='*60}")

        print(f"\nRequest counts:")
        print(f"  Processing: {batch.request_counts.processing}")
        print(f"  Succeeded: {batch.request_counts.succeeded}")
        print(f"  Errored: {batch.request_counts.errored}")
        print(f"  Canceled: {batch.request_counts.canceled}")
        print(f"  Expired: {batch.request_counts.expired}")

        if batch.processing_status == "ended":
            print(f"\n✓ Batch complete!")
            print(f"\nDownload results with:")
            print(f"  python scripts/app/download_llm_batch.py --batch-id {batch_id}")
        elif batch.processing_status == "in_progress":
            total = sum([
                batch.request_counts.processing,
                batch.request_counts.succeeded,
                batch.request_counts.errored,
                batch.request_counts.canceled,
                batch.request_counts.expired
            ])
            completed = batch.request_counts.succeeded + batch.request_counts.errored
            if total > 0:
                progress = (completed / total) * 100
                print(f"\nProgress: {completed}/{total} ({progress:.1f}%)")
            print(f"\nBatch is still processing. Check again later.")
        else:
            print(f"\nBatch status: {batch.processing_status}")

    except Exception as e:
        print(f"\nError checking batch: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
