from __future__ import annotations

import os
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


def _build_umap(*, n_neighbors: int, min_dist: float, metric: str, random_state: int, deterministic: bool, n_jobs: int | None):
    UMAP = _require_umap()
    kwargs = {
        "n_components": 2,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "low_memory": True,
    }
    if deterministic:
        kwargs["random_state"] = random_state
        kwargs["transform_seed"] = random_state
    if n_jobs is not None:
        kwargs["n_jobs"] = int(n_jobs)
    try:
        return UMAP(**kwargs)
    except TypeError:
        kwargs.pop("n_jobs", None)
        kwargs.pop("low_memory", None)
        return UMAP(**kwargs)


def fit_umap_projection(
    x: np.ndarray,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    metric: str = "cosine",
    random_state: int = 0,
    deterministic: bool = False,
    n_jobs: int | None = None,
):
    x = np.asarray(x, dtype=np.float32)
    reducer = _build_umap(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        deterministic=deterministic,
        n_jobs=n_jobs,
    )
    z = reducer.fit_transform(x)
    return reducer, np.asarray(z, dtype=np.float32)


def transform_with_reducer(reducer, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    z = reducer.transform(x)
    return np.asarray(z, dtype=np.float32)


def default_umap_jobs() -> int | None:
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        return None
    return cpu_count


def save_reducer(reducer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(reducer, f)


def load_reducer(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)
