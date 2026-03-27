from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed(stats: dict[str, float], key: str):
    start = perf_counter()
    try:
        yield
    finally:
        stats[key] = stats.get(key, 0.0) + (perf_counter() - start)
