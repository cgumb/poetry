from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = [
    "backend",
    "fit_seconds",
    "score_seconds",
    "select_seconds",
    "total_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_plot.png"))
    return parser.parse_args()


def load_records(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        obj = json.loads(path.read_text())
        rows.append(obj)
    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return df


def main() -> None:
    args = parse_args()
    df = load_records(args.inputs)
    df = df.sort_values(["backend", "n_poems", "m_rated"], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    grouped = df.groupby("backend", sort=False)
    for backend, sub in grouped:
        axes[0].plot(sub["n_poems"], sub["total_seconds"], marker="o", label=backend)
    axes[0].set_xlabel("Number of poems")
    axes[0].set_ylabel("Total step time (s)")
    axes[0].set_title("Total interaction-step time")
    axes[0].legend()

    latest = df.sort_values("n_poems").groupby("backend", as_index=False).tail(1)
    x = range(len(latest))
    axes[1].bar(x, latest["fit_seconds"], label="fit")
    axes[1].bar(x, latest["score_seconds"], bottom=latest["fit_seconds"], label="score")
    axes[1].bar(
        x,
        latest["select_seconds"],
        bottom=latest["fit_seconds"] + latest["score_seconds"],
        label="select",
    )
    axes[1].set_xticks(list(x), latest["backend"].tolist())
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Stage breakdown for largest run")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
