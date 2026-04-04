from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.heatmap import smooth_scalar_field

TITLE_CANDIDATES = ["title", "poem_title", "name"]
POET_CANDIDATES = ["poet", "author", "poet_name"]
BAD_POET_NAMES = {"anonymous", "anon", "unknown", "unknown poet", "[unknown poet]", "traditional", "various"}


def _normalize_name(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _is_good_poet_name(value: object) -> bool:
    name = _normalize_name(value)
    if not name:
        return False
    if name in BAD_POET_NAMES or name.startswith("anonymous ") or name.startswith("unknown "):
        return False
    return True


def _available_columns(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def _load_plot_poem_metadata(path: Path) -> pd.DataFrame:
    wanted = ["poem_id", "title", "poem_title", "name", "poet", "author", "poet_name"]
    available = set(_available_columns(path))
    cols = [c for c in wanted if c in available]
    if not cols:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=cols)


def _pick_column(columns: list[str], candidates: list[str], fallback: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    return fallback


def _build_score_frame(
    poems: pd.DataFrame,
    mean: np.ndarray,
    variance: np.ndarray,
    rated_indices: list[int],
    current_index: int | None,
    exploit_index: int,
    explore_index: int,
) -> pd.DataFrame:
    cols = list(poems.columns)
    title_col = _pick_column(cols, TITLE_CANDIDATES, cols[0])
    poet_col = _pick_column(cols, POET_CANDIDATES, cols[0])
    rated_mask = np.zeros(len(poems), dtype=bool)
    if rated_indices:
        rated_mask[np.asarray(rated_indices, dtype=np.int64)] = True
    out = pd.DataFrame(
        {
            "row_index": np.arange(len(poems), dtype=np.int64),
            "poem_id": poems["poem_id"] if "poem_id" in poems.columns else np.arange(len(poems), dtype=np.int64),
            "title": poems[title_col].astype(str),
            "poet": poems[poet_col].astype(str),
            "posterior_mean": mean,
            "posterior_variance": variance,
            "is_rated": rated_mask,
            "is_current": False,
            "is_exploit": False,
            "is_explore": False,
        }
    )
    if current_index is not None and 0 <= current_index < len(out):
        out.loc[int(current_index), "is_current"] = True
    out.loc[int(exploit_index), "is_exploit"] = True
    out.loc[int(explore_index), "is_explore"] = True
    return out


def _load_rating_session(path: Path) -> tuple[int | None, list[int], list[float]]:
    if not path.exists():
        return None, [], []
    payload = json.loads(path.read_text())
    current_index = payload.get("current_index")
    rated_indices = [int(x) for x in payload.get("rated_indices", [])]
    ratings = [float(x) for x in payload.get("ratings", [])]
    if len(rated_indices) != len(ratings):
        raise ValueError(f"Session file {path} has mismatched rated_indices and ratings lengths")
    return current_index, rated_indices, ratings


def _load_poet_overlay(poets_path: Path | None, coords_path: Path | None) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    if poets_path is None or coords_path is None or not poets_path.exists() or not coords_path.exists():
        return None, None
    poets = pd.read_parquet(poets_path)
    coords = np.load(coords_path, mmap_mode="r")
    if len(poets) != coords.shape[0]:
        warnings.warn("poet centroid metadata and coordinate row counts do not match; skipping poet overlay", stacklevel=2)
        return None, None
    if "poet" in poets.columns:
        mask = poets["poet"].map(_is_good_poet_name).to_numpy()
        poets = poets[mask].reset_index(drop=True)
        coords = np.asarray(coords[mask])
    return poets, coords


def _label_indices(coords: np.ndarray, max_labels: int) -> list[int]:
    if len(coords) == 0 or max_labels <= 0:
        return []
    range_x = float(coords[:, 0].max() - coords[:, 0].min())
    range_y = float(coords[:, 1].max() - coords[:, 1].min())
    diag = max((range_x**2 + range_y**2) ** 0.5, 1e-12)
    min_sep = 0.04 * diag
    chosen: list[int] = []
    for i in range(len(coords)):
        if len(chosen) >= max_labels:
            break
        if all(np.linalg.norm(coords[i] - coords[j]) >= min_sep for j in chosen):
            chosen.append(i)
    return chosen


def _draw_poet_overlay(ax: plt.Axes, poets: pd.DataFrame | None, poet_coords: np.ndarray | None, *, topn: int, label_topn: int) -> None:
    if poets is None or poet_coords is None or len(poets) == 0:
        return
    order = np.argsort(-poets["n_poems"].to_numpy(), kind="stable") if "n_poems" in poets.columns else np.arange(len(poets))
    order = order[: min(topn, len(order))]
    poets_sub = poets.iloc[order].reset_index(drop=True)
    coords_sub = np.asarray(poet_coords[order])
    sizes = 15 + 4 * np.sqrt(poets_sub["n_poems"].to_numpy()) if "n_poems" in poets_sub.columns else np.full(len(poets_sub), 25.0)
    ax.scatter(coords_sub[:, 0], coords_sub[:, 1], s=sizes, alpha=0.35, marker="^")
    for i in _label_indices(coords_sub, label_topn):
        ax.text(coords_sub[i, 0], coords_sub[i, 1], str(poets_sub.iloc[i]["poet"]), fontsize=8)


def _annotate_special_points(ax: plt.Axes, coords_2d: np.ndarray, score_df: pd.DataFrame) -> None:
    rated = score_df.loc[score_df["is_rated"], "row_index"].to_numpy(dtype=np.int64)
    current = score_df.loc[score_df["is_current"], "row_index"].to_numpy(dtype=np.int64)
    exploit = score_df.loc[score_df["is_exploit"], "row_index"].to_numpy(dtype=np.int64)
    explore = score_df.loc[score_df["is_explore"], "row_index"].to_numpy(dtype=np.int64)
    if len(rated):
        ax.scatter(coords_2d[rated, 0], coords_2d[rated, 1], s=22, alpha=0.9, label="rated")
    if len(current):
        i = int(current[0])
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], s=90, marker="x", linewidths=1.5, label="current")
    if len(exploit):
        i = int(exploit[0])
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], s=140, marker="*", label="exploit")
    if len(explore):
        i = int(explore[0])
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], s=90, marker="s", label="explore")


def _finalize_plot(fig: plt.Figure, ax: plt.Axes, output_path: Path, *, title: str) -> None:
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _preserve_indices(score_df: pd.DataFrame) -> np.ndarray:
    preserve = np.flatnonzero(
        score_df["is_rated"].to_numpy()
        | score_df["is_current"].to_numpy()
        | score_df["is_exploit"].to_numpy()
        | score_df["is_explore"].to_numpy()
    )
    return preserve.astype(np.int64, copy=False)


def _render_smooth(coords_2d: np.ndarray, values: np.ndarray, score_df: pd.DataFrame, output_path: Path, *, title: str, poets: pd.DataFrame | None, poet_coords: np.ndarray | None, poet_topn: int, poet_label_topn: int, grid_size: int, bandwidth: float, max_smooth_points: int | None, grid_block_size: int) -> None:
    hm = smooth_scalar_field(
        coords_2d,
        values,
        grid_size=grid_size,
        bandwidth=bandwidth,
        max_points=max_smooth_points,
        preserve_indices=_preserve_indices(score_df),
        grid_block_size=grid_block_size,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        hm["zz"],
        extent=[hm["xs"][0], hm["xs"][-1], hm["ys"][0], hm["ys"][-1]],
        origin="lower",
        aspect="auto",
        alpha=0.85,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=3, alpha=0.10)
    _annotate_special_points(ax, coords_2d, score_df)
    _draw_poet_overlay(ax, poets, poet_coords, topn=poet_topn, label_topn=poet_label_topn)
    _finalize_plot(fig, ax, output_path, title=title)


def _render_hexbin(coords_2d: np.ndarray, values: np.ndarray, score_df: pd.DataFrame, output_path: Path, *, title: str, poets: pd.DataFrame | None, poet_coords: np.ndarray | None, poet_topn: int, poet_label_topn: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    hb = ax.hexbin(coords_2d[:, 0], coords_2d[:, 1], C=values, reduce_C_function=np.mean, gridsize=60, mincnt=1)
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    _annotate_special_points(ax, coords_2d, score_df)
    _draw_poet_overlay(ax, poets, poet_coords, topn=poet_topn, label_topn=poet_label_topn)
    _finalize_plot(fig, ax, output_path, title=title)


def _render_scatter(coords_2d: np.ndarray, values: np.ndarray, score_df: pd.DataFrame, output_path: Path, *, title: str, poets: pd.DataFrame | None, poet_coords: np.ndarray | None, poet_topn: int, poet_label_topn: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=values, s=6, alpha=0.45)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _annotate_special_points(ax, coords_2d, score_df)
    _draw_poet_overlay(ax, poets, poet_coords, topn=poet_topn, label_topn=poet_label_topn)
    _finalize_plot(fig, ax, output_path, title=title)


def _render_plot(style: str, coords_2d: np.ndarray, values: np.ndarray, score_df: pd.DataFrame, output_path: Path, *, title: str, poets: pd.DataFrame | None, poet_coords: np.ndarray | None, poet_topn: int, poet_label_topn: int, grid_size: int, bandwidth: float, max_smooth_points: int | None, grid_block_size: int) -> None:
    if style == "smooth":
        _render_smooth(coords_2d, values, score_df, output_path, title=title, poets=poets, poet_coords=poet_coords, poet_topn=poet_topn, poet_label_topn=poet_label_topn, grid_size=grid_size, bandwidth=bandwidth, max_smooth_points=max_smooth_points, grid_block_size=grid_block_size)
    elif style == "hexbin":
        _render_hexbin(coords_2d, values, score_df, output_path, title=title, poets=poets, poet_coords=poet_coords, poet_topn=poet_topn, poet_label_topn=poet_label_topn)
    elif style == "scatter":
        _render_scatter(coords_2d, values, score_df, output_path, title=title, poets=poets, poet_coords=poet_coords, poet_topn=poet_topn, poet_label_topn=poet_label_topn)
    else:
        raise ValueError(f"Unknown plot style: {style}")


def _render_score_frame(score_df: pd.DataFrame, coords_2d: np.ndarray, output_dir: Path, *, step_tag: str, plot_style: str, poets_path: Path | None, poet_coords_path: Path | None, grid_size: int, bandwidth: float, poet_topn: int, poet_label_topn: int, max_smooth_points: int | None, grid_block_size: int) -> tuple[Path, Path]:
    poets, poet_coords = _load_poet_overlay(poets_path, poet_coords_path)
    mean = score_df["posterior_mean"].to_numpy(dtype=np.float32, copy=False)
    variance = score_df["posterior_variance"].to_numpy(dtype=np.float32, copy=False)
    step_mean_plot = output_dir / f"{step_tag}_posterior_mean_{plot_style}.png"
    step_variance_plot = output_dir / f"{step_tag}_posterior_variance_{plot_style}.png"
    latest_mean_plot = output_dir / f"latest_posterior_mean_{plot_style}.png"
    latest_variance_plot = output_dir / f"latest_posterior_variance_{plot_style}.png"
    for target in [step_mean_plot, latest_mean_plot]:
        _render_plot(plot_style, coords_2d, mean, score_df, target, title=f"Posterior mean over projected poem space ({plot_style})", poets=poets, poet_coords=poet_coords, poet_topn=poet_topn, poet_label_topn=poet_label_topn, grid_size=grid_size, bandwidth=bandwidth, max_smooth_points=max_smooth_points, grid_block_size=grid_block_size)
    for target in [step_variance_plot, latest_variance_plot]:
        _render_plot(plot_style, coords_2d, variance, score_df, target, title=f"Posterior variance over projected poem space ({plot_style})", poets=poets, poet_coords=poet_coords, poet_topn=poet_topn, poet_label_topn=poet_label_topn, grid_size=grid_size, bandwidth=bandwidth, max_smooth_points=max_smooth_points, grid_block_size=grid_block_size)
    return step_mean_plot, step_variance_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--coords", type=Path, default=Path("data/proj2d.npy"))
    parser.add_argument("--session-file", type=Path, default=Path("data/ratings_session.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/session_plots"))
    parser.add_argument("--poets", type=Path, default=Path("data/poet_centroids.parquet"))
    parser.add_argument("--poet-coords", type=Path, default=Path("data/poet_centroids_2d.npy"))
    parser.add_argument("--scores-file", type=Path, default=None)
    parser.add_argument("--reuse-saved-scores", action="store_true")
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--bandwidth", type=float, default=0.10)
    parser.add_argument("--poet-topn", type=int, default=40)
    parser.add_argument("--poet-label-topn", type=int, default=12)
    parser.add_argument("--plot-style", choices=["smooth", "hexbin", "scatter"], default="hexbin")
    parser.add_argument("--max-smooth-points", type=int, default=5000)
    parser.add_argument("--grid-block-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coords_2d = np.asarray(np.load(args.coords, mmap_mode="r"))
    current_index, rated_indices, ratings = _load_rating_session(args.session_file)
    if not rated_indices:
        raise ValueError(f"Session file {args.session_file} does not contain any rated poems")

    poets_path = args.poets if args.poets.exists() else None
    poet_coords_path = args.poet_coords if args.poet_coords.exists() else None
    scores_file = args.scores_file
    if scores_file is None and args.reuse_saved_scores:
        candidate = args.output_dir / "latest_scores.parquet"
        scores_file = candidate if candidate.exists() else None

    if scores_file is not None and scores_file.exists():
        score_df = pd.read_parquet(scores_file)
        step_mean_plot, step_variance_plot = _render_score_frame(
            score_df,
            coords_2d,
            args.output_dir,
            step_tag="latest",
            plot_style=args.plot_style,
            poets_path=poets_path,
            poet_coords_path=poet_coords_path,
            grid_size=args.grid_size,
            bandwidth=args.bandwidth,
            poet_topn=args.poet_topn,
            poet_label_topn=args.poet_label_topn,
            max_smooth_points=args.max_smooth_points,
            grid_block_size=args.grid_block_size,
        )
        print(f"wrote {step_mean_plot}")
        print(f"wrote {step_variance_plot}")
        print(f"reused scores from {scores_file}")
        return

    poems = _load_plot_poem_metadata(args.poems)
    embeddings = np.load(args.embeddings, mmap_mode="r")
    result = run_blocked_step(
        embeddings,
        np.asarray(rated_indices, dtype=np.int64),
        np.asarray(ratings, dtype=np.float64),
        length_scale=args.length_scale,
        variance=args.variance,
        noise=args.noise,
        block_size=args.block_size,
    )
    score_df = _build_score_frame(
        poems,
        result.mean,
        result.variance,
        rated_indices,
        current_index,
        result.exploit_index,
        result.explore_index,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_tag = f"step_{len(rated_indices):03d}"
    step_scores = args.output_dir / f"{step_tag}_scores.parquet"
    latest_scores = args.output_dir / "latest_scores.parquet"
    step_summary = args.output_dir / f"{step_tag}_summary.json"
    latest_summary = args.output_dir / "latest_summary.json"
    score_df.to_parquet(step_scores, index=False)
    score_df.to_parquet(latest_scores, index=False)
    summary_payload = {
        "n_rated": len(rated_indices),
        "rated_indices": [int(x) for x in rated_indices],
        "ratings": [float(x) for x in ratings],
        "current_index": None if current_index is None else int(current_index),
        "exploit_index": int(result.exploit_index),
        "explore_index": int(result.explore_index),
        "fit_seconds": float(result.profile.fit_seconds),
        "score_seconds": float(result.profile.score_seconds),
        "select_seconds": float(result.profile.select_seconds),
        "total_seconds": float(result.profile.total_seconds),
        "plot_style": args.plot_style,
        "max_smooth_points": args.max_smooth_points,
    }
    step_summary.write_text(json.dumps(summary_payload, indent=2))
    latest_summary.write_text(json.dumps(summary_payload, indent=2))

    step_mean_plot, step_variance_plot = _render_score_frame(
        score_df,
        coords_2d,
        args.output_dir,
        step_tag=step_tag,
        plot_style=args.plot_style,
        poets_path=poets_path,
        poet_coords_path=poet_coords_path,
        grid_size=args.grid_size,
        bandwidth=args.bandwidth,
        poet_topn=args.poet_topn,
        poet_label_topn=args.poet_label_topn,
        max_smooth_points=args.max_smooth_points,
        grid_block_size=args.grid_block_size,
    )
    print(f"wrote {step_mean_plot}")
    print(f"wrote {step_variance_plot}")
    print(f"wrote {step_scores}")
    print(f"wrote {step_summary}")
    print(
        f"Timing: fit={result.profile.fit_seconds:.4f}s "
        f"score={result.profile.score_seconds:.4f}s "
        f"total={result.profile.total_seconds:.4f}s"
    )


if __name__ == "__main__":
    main()
