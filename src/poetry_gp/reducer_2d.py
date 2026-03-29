from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _require_umap():
    try:
        from umap import UMAP
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "UMAP projection requires the 'umap-learn' package. Install it in the project environment first."
        ) from exc
    return UMAP


def fit_umap_projection(
    x: np.ndarray,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    metric: str = "cosine",
    random_state: int = 0,
):
    UMAP = _require_umap()
    x = np.asarray(x, dtype=np.float64)
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        transform_seed=random_state,
    )
    z = reducer.fit_transform(x)
    return reducer, np.asarray(z, dtype=np.float64)


def transform_with_reducer(reducer, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = reducer.transform(x)
    return np.asarray(z, dtype=np.float64)


def save_reducer(reducer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(reducer, f)


def load_reducer(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)
