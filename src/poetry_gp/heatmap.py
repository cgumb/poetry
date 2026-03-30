from __future__ import annotations

import numpy as np


def _downsample_points(
    coords_2d: np.ndarray,
    values: np.ndarray,
    *,
    max_points: int | None,
    preserve_indices: np.ndarray | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0 or len(coords_2d) <= max_points:
        return coords_2d, values
    preserve = np.asarray([], dtype=np.int64) if preserve_indices is None else np.unique(np.asarray(preserve_indices, dtype=np.int64))
    preserve = preserve[(preserve >= 0) & (preserve < len(coords_2d))]
    if len(preserve) >= max_points:
        keep = preserve[:max_points]
        return coords_2d[keep], values[keep]
    pool_mask = np.ones(len(coords_2d), dtype=bool)
    pool_mask[preserve] = False
    pool = np.flatnonzero(pool_mask)
    need = max_points - len(preserve)
    rng = np.random.default_rng(random_state)
    extra = rng.choice(pool, size=min(need, len(pool)), replace=False) if len(pool) else np.asarray([], dtype=np.int64)
    keep = np.sort(np.concatenate([preserve, extra]))
    return coords_2d[keep], values[keep]


def smooth_scalar_field(
    coords_2d: np.ndarray,
    values: np.ndarray,
    *,
    grid_size: int = 150,
    bandwidth: float = 0.15,
    max_points: int | None = 5000,
    preserve_indices: np.ndarray | None = None,
    random_state: int = 0,
    grid_block_size: int = 4096,
) -> dict[str, np.ndarray]:
    coords_2d = np.asarray(coords_2d, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError("coords_2d must have shape (n, 2)")
    if values.ndim != 1 or values.shape[0] != coords_2d.shape[0]:
        raise ValueError("values must have shape (n,)")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    coords_2d, values = _downsample_points(
        coords_2d,
        values,
        max_points=max_points,
        preserve_indices=preserve_indices,
        random_state=random_state,
    )

    x_min, y_min = coords_2d.min(axis=0)
    x_max, y_max = coords_2d.max(axis=0)
    x_pad = 0.05 * max(float(x_max - x_min), 1e-12)
    y_pad = 0.05 * max(float(y_max - y_min), 1e-12)

    xs = np.linspace(x_min - x_pad, x_max + x_pad, grid_size, dtype=np.float32)
    ys = np.linspace(y_min - y_pad, y_max + y_pad, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32, copy=False)

    zz_flat = np.empty(len(grid), dtype=np.float32)
    denom_eps = np.float32(1e-12)
    inv_bw2 = np.float32(0.5 / (bandwidth * bandwidth))
    for start in range(0, len(grid), grid_block_size):
        stop = min(start + grid_block_size, len(grid))
        block = grid[start:stop]
        d2 = np.sum((block[:, None, :] - coords_2d[None, :, :]) ** 2, axis=2, dtype=np.float32)
        w = np.exp(-d2 * inv_bw2).astype(np.float32, copy=False)
        denom = np.sum(w, axis=1, dtype=np.float32)
        denom = np.maximum(denom, denom_eps)
        zz_flat[start:stop] = (w @ values) / denom
    zz = zz_flat.reshape(grid_size, grid_size)

    return {"xs": xs, "ys": ys, "zz": zz, "xx": xx, "yy": yy}
