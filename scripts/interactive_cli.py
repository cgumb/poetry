from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from poetry_gp.backends.blocked import run_blocked_step


TEXT_CANDIDATES = ["text", "poem", "content", "body"]
TITLE_CANDIDATES = ["title", "poem_title", "name"]
POET_CANDIDATES = ["poet", "author", "poet_name"]


def pick_column(columns: list[str], candidates: list[str], fallback: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--session-file", type=Path, default=Path("data/ratings_session.json"))
    return parser.parse_args()


def display_text(value: object, fallback: str) -> str:
    text = str(value).strip()
    return text if text else fallback


def show_poem(df: pd.DataFrame, idx: int, title_col: str, poet_col: str, text_col: str) -> None:
    row = df.iloc[idx]
    title = display_text(row[title_col], "[untitled poem]")
    poet = display_text(row[poet_col], "[unknown poet]")
    print("\n" + "=" * 80)
    print(f"[{idx}] {title}")
    print(f"by {poet}")
    print("-" * 80)
    print(str(row[text_col])[:4000])
    print("=" * 80)


def save_session(path: Path, current_idx: int | None, rated_indices: list[int], ratings: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_index": current_idx,
        "rated_indices": rated_indices,
        "ratings": ratings,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_session(path: Path) -> tuple[int | None, list[int], list[float]]:
    if not path.exists():
        return None, [], []
    payload = json.loads(path.read_text())
    current_index = payload.get("current_index")
    rated_indices = [int(x) for x in payload.get("rated_indices", [])]
    ratings = [float(x) for x in payload.get("ratings", [])]
    if len(rated_indices) != len(ratings):
        raise ValueError(f"Session file {path} has mismatched rated_indices and ratings lengths")
    return current_index, rated_indices, ratings


def print_rated_summary(df: pd.DataFrame, rated_indices: list[int], ratings: list[float], title_col: str, poet_col: str) -> None:
    if not rated_indices:
        print("No poems rated yet.")
        return
    print("\nRated poems:")
    for idx, rating in zip(rated_indices, ratings):
        row = df.iloc[idx]
        title = display_text(row[title_col], "[untitled poem]")
        poet = display_text(row[poet_col], "[unknown poet]")
        print(f"  [{idx}] rating={rating:+.1f}  {title} — {poet}")


def search_poems(df: pd.DataFrame, query: str, title_col: str, poet_col: str, text_col: str, limit: int = 10) -> list[int]:
    q = query.strip()
    if not q:
        return []
    title_mask = df[title_col].astype(str).str.contains(q, case=False, regex=False)
    poet_mask = df[poet_col].astype(str).str.contains(q, case=False, regex=False)
    text_mask = df[text_col].astype(str).str.contains(q, case=False, regex=False)
    mask = title_mask | poet_mask | text_mask
    matches = df.index[mask].tolist()[:limit]
    return [int(i) for i in matches]


def prompt_search(df: pd.DataFrame, title_col: str, poet_col: str, text_col: str) -> int | None:
    query = input("Search title/poet/text: ").strip()
    matches = search_poems(df, query, title_col, poet_col, text_col)
    if not matches:
        print("No matches found.")
        return None
    print("\nMatches:")
    for i, idx in enumerate(matches, start=1):
        row = df.iloc[idx]
        title = display_text(row[title_col], "[untitled poem]")
        poet = display_text(row[poet_col], "[unknown poet]")
        print(f"  {i}. [{idx}] {title} — {poet}")
    selection = input("Choose a match number (blank to cancel): ").strip()
    if not selection:
        return None
    try:
        chosen = int(selection)
    except ValueError:
        print("Invalid selection.")
        return None
    if not (1 <= chosen <= len(matches)):
        print("Selection out of range.")
        return None
    return matches[chosen - 1]


def main() -> None:
    args = parse_args()
    poems = pd.read_parquet(args.poems)
    embeddings = np.load(args.embeddings)
    if len(poems) != embeddings.shape[0]:
        raise ValueError("poem and embedding row counts do not match")

    cols = list(poems.columns)
    text_col = pick_column(cols, TEXT_CANDIDATES, cols[-1])
    title_col = pick_column(cols, TITLE_CANDIDATES, cols[0])
    poet_col = pick_column(cols, POET_CANDIDATES, cols[0])

    saved_current, rated_indices, ratings = load_session(args.session_file)
    rng = np.random.default_rng(args.seed)
    if saved_current is not None and 0 <= saved_current < len(poems):
        current_idx = int(saved_current)
        print(f"Loaded session from {args.session_file}")
    else:
        current_idx = int(rng.integers(len(poems)))

    help_text = (
        "Commands: [l]ike [n]eutral [d]islike [e]xploit e[x]plore [s]earch [r]ated [q]uit"
    )

    while True:
        show_poem(poems, current_idx, title_col, poet_col, text_col)
        cmd = input(f"{help_text}\n> ").strip().lower()
        if cmd == "q":
            save_session(args.session_file, current_idx, rated_indices, ratings)
            print(f"Saved session to {args.session_file}")
            break
        if cmd in {"l", "n", "d"}:
            if current_idx not in rated_indices:
                rated_indices.append(current_idx)
                ratings.append({"l": 1.0, "n": 0.0, "d": -1.0}[cmd])
                save_session(args.session_file, current_idx, rated_indices, ratings)
                print(f"Recorded rating {ratings[-1]} for poem {current_idx}")
            else:
                print("Poem already rated.")
            continue
        if cmd == "r":
            print_rated_summary(poems, rated_indices, ratings, title_col, poet_col)
            continue
        if cmd == "s":
            selected = prompt_search(poems, title_col, poet_col, text_col)
            if selected is not None:
                current_idx = selected
                save_session(args.session_file, current_idx, rated_indices, ratings)
            continue
        if cmd not in {"e", "x"}:
            print("Unknown command.")
            continue
        if not rated_indices:
            print("Rate at least one poem first.")
            continue
        result = run_blocked_step(
            embeddings,
            np.array(rated_indices, dtype=np.int64),
            np.array(ratings, dtype=np.float64),
        )
        current_idx = result.exploit_index if cmd == "e" else result.explore_index
        save_session(args.session_file, current_idx, rated_indices, ratings)
        print(
            f"Timing: fit={result.profile.fit_seconds:.4f}s score={result.profile.score_seconds:.4f}s total={result.profile.total_seconds:.4f}s"
        )


if __name__ == "__main__":
    main()
