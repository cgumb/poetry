from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-input", type=Path, default=Path("results/bench_results.csv"))
    parser.add_argument("--mpi-input", type=Path, default=Path("results/mpi_bench_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/all_benchmarks.png"))
    return parser.parse_args()


def maybe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    serial_df = maybe_read_csv(args.serial_input)
    mpi_df = maybe_read_csv(args.mpi_input)
    if serial_df is None and mpi_df is None:
        raise ValueError("No benchmark CSV files found")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    if serial_df is not None:
        for backend, sub in serial_df.groupby("backend", sort=False):
            grp = sub.groupby("n_poems", as_index=False)["total_seconds"].mean()
            axes[0].plot(grp["n_poems"], grp["total_seconds"], marker="o", label=backend)
    if mpi_df is not None:
        for ranks, sub in mpi_df.groupby("ranks", sort=False):
            grp = sub.groupby("n_poems", as_index=False)["total_seconds"].mean()
            axes[0].plot(grp["n_poems"], grp["total_seconds"], marker="o", linestyle="--", label=f"mpi-{ranks}r")
    axes[0].set_title("Mean total time vs corpus size")
    axes[0].set_xlabel("Number of poems")
    axes[0].set_ylabel("Seconds")
    axes[0].legend()

    if serial_df is not None:
        for backend, sub in serial_df.groupby("backend", sort=False):
            grp = sub.groupby("m_rated", as_index=False)["total_seconds"].mean()
            axes[1].plot(grp["m_rated"], grp["total_seconds"], marker="o", label=backend)
    if mpi_df is not None:
        for ranks, sub in mpi_df.groupby("ranks", sort=False):
            grp = sub.groupby("m_rated", as_index=False)["total_seconds"].mean()
            axes[1].plot(grp["m_rated"], grp["total_seconds"], marker="o", linestyle="--", label=f"mpi-{ranks}r")
    axes[1].set_title("Mean total time vs rated set size")
    axes[1].set_xlabel("Rated poems")
    axes[1].set_ylabel("Seconds")
    axes[1].legend()

    bars = []
    labels = []
    if serial_df is not None:
        latest_serial = serial_df.sort_values(["n_poems", "m_rated"]).groupby("backend", as_index=False).tail(1)
        for _, row in latest_serial.iterrows():
            bars.append((row["fit_seconds"], row["score_seconds"], row.get("select_seconds", 0.0)))
            labels.append(str(row["backend"]))
    if mpi_df is not None:
        latest_mpi = mpi_df.sort_values(["n_poems", "m_rated", "ranks"]).groupby("ranks", as_index=False).tail(1)
        for _, row in latest_mpi.iterrows():
            bars.append((row["fit_seconds"], row["local_score_seconds"], row["broadcast_seconds"] + row["reduce_seconds"]))
            labels.append(f"mpi-{int(row['ranks'])}r")

    x = range(len(bars))
    fit_vals = [b[0] for b in bars]
    score_vals = [b[1] for b in bars]
    other_vals = [b[2] for b in bars]
    axes[2].bar(x, fit_vals, label="fit")
    axes[2].bar(x, score_vals, bottom=fit_vals, label="score/local-score")
    axes[2].bar(x, other_vals, bottom=[a + b for a, b in zip(fit_vals, score_vals)], label="select or comm")
    axes[2].set_xticks(list(x), labels)
    axes[2].set_title("Largest-run stage breakdown")
    axes[2].set_ylabel("Seconds")
    axes[2].legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
