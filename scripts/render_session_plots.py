from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from poetry_gp.session_viz import load_rating_session, render_session_gp_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--coords", type=Path, default=Path("data/proj2d.npy"))
    parser.add_argument("--session-file", type=Path, default=Path("data/ratings_session.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/session_plots"))
    parser.add_argument("--poets", type=Path, default=Path("data/poet_centroids.parquet"))
    parser.add_argument("--poet-coords", type=Path, default=Path("data/poet_centroids_2d.npy"))
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--grid-size", type=int, default=150)
    parser.add_argument("--bandwidth", type=float, default=0.15)
    parser.add_argument("--poet-topn", type=int, default=80)
    parser.add_argument("--poet-label-topn", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poems = pd.read_parquet(args.poems)
    embeddings = np.load(args.embeddings)
    coords_2d = np.load(args.coords)
    current_index, rated_indices, ratings = load_rating_session(args.session_file)
    if not rated_indices:
        raise ValueError(f"Session file {args.session_file} does not contain any rated poems")

    poets_path = args.poets if args.poets.exists() else None
    poet_coords_path = args.poet_coords if args.poet_coords.exists() else None

    result = render_session_gp_outputs(
        poems,
        embeddings,
        coords_2d,
        rated_indices,
        ratings,
        args.output_dir,
        current_index=current_index,
        poets_path=poets_path,
        poet_coords_path=poet_coords_path,
        length_scale=args.length_scale,
        variance=args.variance,
        noise=args.noise,
        block_size=args.block_size,
        grid_size=args.grid_size,
        bandwidth=args.bandwidth,
        poet_topn=args.poet_topn,
        poet_label_topn=args.poet_label_topn,
    )

    print(f"wrote {result.outputs.step_mean_plot}")
    print(f"wrote {result.outputs.step_variance_plot}")
    print(f"wrote {result.outputs.step_scores}")
    print(f"wrote {result.outputs.step_summary_json}")
    print(
        f"Timing: fit={result.blocked_result.profile.fit_seconds:.4f}s "
        f"score={result.blocked_result.profile.score_seconds:.4f}s "
        f"total={result.blocked_result.profile.total_seconds:.4f}s"
    )


if __name__ == "__main__":
    main()
