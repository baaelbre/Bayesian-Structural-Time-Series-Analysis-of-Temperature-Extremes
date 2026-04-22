from __future__ import annotations

import numpy as np


def one_step_ahead_residuals(y, eta):
    """Simple observation residuals y_t - E[Y_t | state].

    In v0.1 this is a lightweight baseline diagnostic. Standardized and
    observation-specific residuals can be layered on top later.
    """
    y = np.asarray(y, dtype=float)
    eta = np.asarray(eta, dtype=float)
    return y - eta
