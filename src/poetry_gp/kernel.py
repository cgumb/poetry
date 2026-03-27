from __future__ import annotations

import numpy as np


def rowwise_squared_norms(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sum(x * x, axis=1)


def pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute all pairwise squared distances between row vectors in x and y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_norm = rowwise_squared_norms(x)[:, None]
    y_norm = rowwise_squared_norms(y)[None, :]
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(d2, 0.0)


def rbf_kernel(x: np.ndarray, y: np.ndarray, length_scale: float = 1.0, variance: float = 1.0) -> np.ndarray:
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")
    if variance <= 0:
        raise ValueError("variance must be positive")
    d2 = pairwise_squared_distances(x, y)
    return variance * np.exp(-0.5 * d2 / (length_scale * length_scale))
