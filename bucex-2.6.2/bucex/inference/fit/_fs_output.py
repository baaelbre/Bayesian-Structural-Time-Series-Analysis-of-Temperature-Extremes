"""Private transport object produced by the FS kernels before normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FSOutput:
    draws_static: dict[str, np.ndarray]
    draws_states: np.ndarray
    logpost: np.ndarray
    acceptance: dict[str, float]
    meta: dict[str, Any]

    @property
    def n_draws(self) -> int:
        return int(np.asarray(self.draws_states).shape[0])


__all__ = ["FSOutput"]
