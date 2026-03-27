from __future__ import annotations

import numpy as np


def smooth_scalar_field(
    coords_2d: np.ndarray,
    values: np.ndarray,
    *,
    grid_size: int = 150,
    bandwidth: float = 0.15,
) -> dict[str, np.ndarray]:
    coords_2d = np.asarray(coords_2d, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError("coords_2d must have shape (n, 2)")
    if values.ndim != 1 or values.shape[0] != coords_2d.shape[0]:
        raise ValueError("values must have shape (n,)")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    x_min, y_min = coords_2d.min(axis=0)
    x_max, y_max = coords_2d.max(axis=0)
    x_pad = 0.05 * max(x_max - x_min, 1e-12)
    y_pad = 0.05 * max(y_max - y_min, 1e-12)

    xs = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
    ys = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    d2 = np.sum((grid[:, None, :] - coords_2d[None, :, :]) ** 2, axis=2)
    w = np.exp(-0.5 * d2 / (bandwidth * bandwidth))
    denom = np.sum(w, axis=1)
    denom = np.maximum(denom, 1e-12)
    zz = (w @ values) / denom
    zz = zz.reshape(grid_size, grid_size)

    return {
        "xs": xs,
        "ys": ys,
        "zz": zz,
        "xx": xx,
        "yy": yy,
    }
