from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/bench_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/bench_results.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df = df.sort_values(["backend", "n_poems", "m_rated"], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for backend, sub in df.groupby("backend", sort=False):
        grp = sub.groupby("n_poems", as_index=False)["total_seconds"].mean()
        axes[0].plot(grp["n_poems"], grp["total_seconds"], marker="o", label=backend)
    axes[0].set_title("Mean total time vs corpus size")
    axes[0].set_xlabel("Number of poems")
    axes[0].set_ylabel("Seconds")
    axes[0].legend()

    for backend, sub in df.groupby("backend", sort=False):
        grp = sub.groupby("m_rated", as_index=False)["total_seconds"].mean()
        axes[1].plot(grp["m_rated"], grp["total_seconds"], marker="o", label=backend)
    axes[1].set_title("Mean total time vs rated set size")
    axes[1].set_xlabel("Rated poems")
    axes[1].set_ylabel("Seconds")
    axes[1].legend()

    latest = df.sort_values(["n_poems", "m_rated"]).groupby("backend", as_index=False).tail(1)
    x = range(len(latest))
    axes[2].bar(x, latest["fit_seconds"], label="fit")
    axes[2].bar(x, latest["score_seconds"], bottom=latest["fit_seconds"], label="score")
    axes[2].bar(x, latest["select_seconds"], bottom=latest["fit_seconds"] + latest["score_seconds"], label="select")
    axes[2].set_xticks(list(x), latest["backend"].tolist())
    axes[2].set_title("Largest-run stage breakdown")
    axes[2].set_ylabel("Seconds")
    axes[2].legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
