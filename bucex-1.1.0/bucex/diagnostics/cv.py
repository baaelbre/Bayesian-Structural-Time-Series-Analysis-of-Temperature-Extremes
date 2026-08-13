from __future__ import annotations

from typing import Iterator, Tuple


def rolling_origin_splits(n_time: int, initial: int, horizon: int = 1, step: int = 1) -> Iterator[Tuple[range, range]]:
    """Yield rolling-origin train/test ranges for time-series CV."""
    if initial <= 0:
        raise ValueError("initial must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")

    train_end = initial
    while train_end + horizon <= n_time:
        yield range(0, train_end), range(train_end, train_end + horizon)
        train_end += step
