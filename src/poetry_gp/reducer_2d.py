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


def _fit_pca_prereducer(x: np.ndarray, out_dims: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if out_dims <= 0 or out_dims >= x.shape[1]:
        return x, np.zeros((1, x.shape[1]), dtype=np.float32), np.eye(x.shape[1], dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    xc = x - mean
    cov = (xc.T.astype(np.float64) @ xc.astype(np.float64)) / max(x.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:out_dims]
    components = evecs[:, order].astype(np.float32, copy=False)
    z = xc @ components
    return z.astype(np.float32, copy=False), mean, components


def _apply_pca_prereducer(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - mean) @ components


def fit_umap_projection(
    x: np.ndarray,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    metric: str = "cosine",
    random_state: int = 0,
    deterministic: bool = False,
    n_jobs: int | None = 1,
    pre_reduce_dims: int | None = 50,
):
    x = np.asarray(x, dtype=np.float32)
    pre_mean = None
    pre_components = None
    x_umap = x
    if pre_reduce_dims is not None and 0 < int(pre_reduce_dims) < x.shape[1]:
        x_umap, pre_mean, pre_components = _fit_pca_prereducer(x, int(pre_reduce_dims))
    reducer = _build_umap(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        deterministic=deterministic,
        n_jobs=n_jobs,
    )
    z = reducer.fit_transform(x_umap)
    bundle = {
        "umap": reducer,
        "pre_mean": pre_mean,
        "pre_components": pre_components,
        "pre_reduce_dims": None if pre_components is None else int(pre_components.shape[1]),
        "metric": metric,
    }
    return bundle, np.asarray(z, dtype=np.float32)


def transform_with_reducer(reducer_bundle, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pre_mean = reducer_bundle.get("pre_mean")
    pre_components = reducer_bundle.get("pre_components")
    if pre_mean is not None and pre_components is not None:
        x = _apply_pca_prereducer(x, pre_mean, pre_components)
    z = reducer_bundle["umap"].transform(x)
    return np.asarray(z, dtype=np.float32)


def save_reducer(reducer_bundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(reducer_bundle, f)


def load_reducer(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)
