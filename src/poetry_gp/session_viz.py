from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backends.blocked import BlockedStepResult, run_blocked_step
from .canonical_poets import is_canonical
from .heatmap import smooth_scalar_field

TEXT_CANDIDATES = ["text", "poem", "content", "body"]
TITLE_CANDIDATES = ["title", "poem_title", "name"]
POET_CANDIDATES = ["poet", "author", "poet_name"]


@dataclass(frozen=True)
class SessionPlotOutputs:
    step_mean_plot: Path
    step_variance_plot: Path
    latest_mean_plot: Path
    latest_variance_plot: Path
    step_scores: Path
    latest_scores: Path
    step_summary_json: Path
    latest_summary_json: Path


@dataclass(frozen=True)
class SessionVizResult:
    blocked_result: BlockedStepResult
    outputs: SessionPlotOutputs


def pick_column(columns: list[str], candidates: list[str], fallback: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    return fallback


def load_rating_session(path: Path) -> tuple[int | None, list[int], list[float]]:
    if not path.exists():
        return None, [], []
    payload = json.loads(path.read_text())
    current_index = payload.get("current_index")
    rated_indices = [int(x) for x in payload.get("rated_indices", [])]
    ratings = [float(x) for x in payload.get("ratings", [])]
    if len(rated_indices) != len(ratings):
        raise ValueError(f"Session file {path} has mismatched rated_indices and ratings lengths")
    return current_index, rated_indices, ratings


def _validate_shapes(poems: pd.DataFrame, embeddings: np.ndarray, coords_2d: np.ndarray) -> None:
    if len(poems) != embeddings.shape[0]:
        raise ValueError("poem and embedding row counts do not match")
    if len(poems) != coords_2d.shape[0]:
        raise ValueError("poem and 2D projection row counts do not match")
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError("coords_2d must have shape (n, 2)")


def _load_poet_overlay(poets_path: Path | None, coords_path: Path | None) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    if poets_path is None or coords_path is None:
        return None, None
    if not poets_path.exists() or not coords_path.exists():
        return None, None
    poets = pd.read_parquet(poets_path)
    coords = np.load(coords_path)
    if len(poets) != coords.shape[0]:
        raise ValueError("poet centroid metadata and coordinate row counts do not match")
    return poets, coords


def build_score_frame(
    poems: pd.DataFrame,
    mean: np.ndarray,
    variance: np.ndarray,
    rated_indices: list[int],
    current_index: int | None,
    exploit_index: int,
    explore_index: int,
) -> pd.DataFrame:
    cols = list(poems.columns)
    title_col = pick_column(cols, TITLE_CANDIDATES, cols[0])
    poet_col = pick_column(cols, POET_CANDIDATES, cols[0])
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


def _draw_poet_overlay(
    ax: plt.Axes,
    poets: pd.DataFrame | None,
    poet_coords: np.ndarray | None,
    *,
    topn: int,
    label_topn: int,
) -> None:
    """Draw poet overlay with hybrid selection: canonical poets prioritized, then by poem count."""
    if poets is None or poet_coords is None or len(poets) == 0:
        return

    # Hybrid selection: prioritize canonical poets, then high-count poets
    poets_work = poets.copy()
    poets_work["is_canonical"] = poets_work["poet"].map(is_canonical)

    # Split into canonical and non-canonical
    canonical_mask = poets_work["is_canonical"].to_numpy()
    canonical_indices = np.flatnonzero(canonical_mask)
    non_canonical_indices = np.flatnonzero(~canonical_mask)

    # Sort both groups by poem count (descending)
    if "n_poems" in poets_work.columns:
        n_poems = poets_work["n_poems"].to_numpy()
        canonical_order = canonical_indices[np.argsort(-n_poems[canonical_indices], kind="stable")]
        non_canonical_order = non_canonical_indices[np.argsort(-n_poems[non_canonical_indices], kind="stable")]
    else:
        canonical_order = canonical_indices
        non_canonical_order = non_canonical_indices

    # Take all canonical poets (up to topn), then fill with high-count non-canonical
    order = np.concatenate([canonical_order, non_canonical_order])
    order = order[: min(topn, len(order))]

    poets_sub = poets.iloc[order].reset_index(drop=True)
    coords_sub = poet_coords[order]

    # Mark which are canonical for sizing
    is_canonical_selected = np.array([is_canonical(p) for p in poets_sub["poet"]])

    if "n_poems" in poets_sub.columns:
        sizes = 20 + 8 * np.sqrt(poets_sub["n_poems"].to_numpy())
        # Boost size for canonical poets
        sizes = np.where(is_canonical_selected, sizes * 1.3, sizes)
    else:
        sizes = np.where(is_canonical_selected, 40.0, 30.0)

    # Less prominent poet markers to reduce clutter
    # Use different colors for canonical vs non-canonical
    colors = np.where(is_canonical_selected, "darkviolet", "purple")
    alphas = np.where(is_canonical_selected, 0.4, 0.2)

    for i, (color, alpha, size) in enumerate(zip(colors, alphas, sizes)):
        ax.scatter(
            coords_sub[i, 0], coords_sub[i, 1],
            s=size, alpha=alpha, marker="^",
            c=color, edgecolors="none"
        )

    # Label canonical poets first, then fill with high-count poets
    n_canonical_to_label = min(sum(is_canonical_selected), label_topn)
    canonical_label_indices = np.flatnonzero(is_canonical_selected)[:n_canonical_to_label]
    remaining_labels = label_topn - len(canonical_label_indices)

    label_indices = set(canonical_label_indices)
    if remaining_labels > 0:
        non_canonical_label_indices = np.flatnonzero(~is_canonical_selected)[:remaining_labels]
        label_indices.update(non_canonical_label_indices)

    for i in sorted(label_indices):
        style_weight = "normal" if is_canonical_selected[i] else "italic"
        ax.text(
            coords_sub[i, 0], coords_sub[i, 1],
            str(poets_sub.iloc[i]["poet"]),
            fontsize=7, alpha=0.7 if is_canonical_selected[i] else 0.5,
            style=style_weight
        )


def render_projection_heatmap(
    coords_2d: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
    rated_indices: list[int],
    current_index: int | None,
    exploit_index: int,
    explore_index: int,
    poets: pd.DataFrame | None,
    poet_coords: np.ndarray | None,
    poet_topn: int,
    poet_label_topn: int,
    grid_size: int,
    bandwidth: float,
) -> None:
    hm = smooth_scalar_field(coords_2d, values, grid_size=grid_size, bandwidth=bandwidth)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Choose colormap: diverging for mean (around 0), sequential for variance
    is_variance_plot = "variance" in title.lower()
    cmap = "YlOrRd" if is_variance_plot else "RdBu_r"

    im = ax.imshow(
        hm["zz"],
        extent=[hm["xs"][0], hm["xs"][-1], hm["ys"][0], hm["ys"][-1]],
        origin="lower",
        aspect="auto",
        alpha=0.7,
        cmap=cmap,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Lighter background scatter to reduce clutter
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=2, alpha=0.08, c="gray")

    # Plot key points with distinct colors and markers
    if rated_indices:
        rated = np.asarray(rated_indices, dtype=np.int64)
        ax.scatter(
            coords_2d[rated, 0], coords_2d[rated, 1],
            s=40, alpha=0.85, c="black", marker="o",
            edgecolors="white", linewidths=0.5, label="rated"
        )
    if current_index is not None and 0 <= current_index < len(coords_2d):
        ax.scatter(
            coords_2d[current_index, 0], coords_2d[current_index, 1],
            s=120, c="cyan", marker="X", edgecolors="black",
            linewidths=1.2, alpha=0.95, label="current"
        )
    ax.scatter(
        coords_2d[exploit_index, 0], coords_2d[exploit_index, 1],
        s=180, c="lime", marker="*", edgecolors="darkgreen",
        linewidths=1.5, alpha=0.95, label="exploit"
    )
    ax.scatter(
        coords_2d[explore_index, 0], coords_2d[explore_index, 1],
        s=120, c="gold", marker="s", edgecolors="darkorange",
        linewidths=1.5, alpha=0.95, label="explore"
    )

    _draw_poet_overlay(ax, poets, poet_coords, topn=poet_topn, label_topn=poet_label_topn)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_session_gp_outputs(
    poems: pd.DataFrame,
    embeddings: np.ndarray,
    coords_2d: np.ndarray,
    rated_indices: list[int],
    ratings: list[float],
    output_dir: Path,
    *,
    current_index: int | None,
    precomputed_result: BlockedStepResult | None = None,
    poets_path: Path | None = None,
    poet_coords_path: Path | None = None,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-3,
    block_size: int = 2048,
    grid_size: int = 150,
    bandwidth: float = 0.15,
    poet_topn: int = 80,
    poet_label_topn: int = 10,
) -> SessionVizResult:
    _validate_shapes(poems, embeddings, coords_2d)
    if not rated_indices:
        raise ValueError("At least one rated poem is required to render GP outputs")

    result = precomputed_result
    if result is None:
        result = run_blocked_step(
            embeddings,
            np.asarray(rated_indices, dtype=np.int64),
            np.asarray(ratings, dtype=np.float64),
            length_scale=length_scale,
            variance=variance,
            noise=noise,
            block_size=block_size,
        )

    poems = poems.reset_index(drop=True)
    poets, poet_coords = _load_poet_overlay(poets_path, poet_coords_path)

    step_tag = f"step_{len(rated_indices):03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    score_df = build_score_frame(
        poems,
        result.mean,
        result.variance,
        rated_indices,
        current_index,
        result.exploit_index,
        result.explore_index,
    )

    step_scores = output_dir / f"{step_tag}_scores.parquet"
    latest_scores = output_dir / "latest_scores.parquet"
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
    }
    step_summary = output_dir / f"{step_tag}_summary.json"
    latest_summary = output_dir / "latest_summary.json"
    step_summary.write_text(json.dumps(summary_payload, indent=2))
    latest_summary.write_text(json.dumps(summary_payload, indent=2))

    step_mean_plot = output_dir / f"{step_tag}_posterior_mean.png"
    latest_mean_plot = output_dir / "latest_posterior_mean.png"
    step_variance_plot = output_dir / f"{step_tag}_posterior_variance.png"
    latest_variance_plot = output_dir / "latest_posterior_variance.png"

    for target in [step_mean_plot, latest_mean_plot]:
        render_projection_heatmap(
            coords_2d,
            result.mean,
            target,
            title="Posterior mean over projected poem space",
            rated_indices=rated_indices,
            current_index=current_index,
            exploit_index=result.exploit_index,
            explore_index=result.explore_index,
            poets=poets,
            poet_coords=poet_coords,
            poet_topn=poet_topn,
            poet_label_topn=poet_label_topn,
            grid_size=grid_size,
            bandwidth=bandwidth,
        )
    for target in [step_variance_plot, latest_variance_plot]:
        render_projection_heatmap(
            coords_2d,
            result.variance,
            target,
            title="Posterior variance over projected poem space",
            rated_indices=rated_indices,
            current_index=current_index,
            exploit_index=result.exploit_index,
            explore_index=result.explore_index,
            poets=poets,
            poet_coords=poet_coords,
            poet_topn=poet_topn,
            poet_label_topn=poet_label_topn,
            grid_size=grid_size,
            bandwidth=bandwidth,
        )

    outputs = SessionPlotOutputs(
        step_mean_plot=step_mean_plot,
        step_variance_plot=step_variance_plot,
        latest_mean_plot=latest_mean_plot,
        latest_variance_plot=latest_variance_plot,
        step_scores=step_scores,
        latest_scores=latest_scores,
        step_summary_json=step_summary,
        latest_summary_json=latest_summary,
    )
    return SessionVizResult(blocked_result=result, outputs=outputs)
