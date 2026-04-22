from __future__ import annotations

import warnings

from ...api.fit import fit_gaussian_structural

warnings.warn(
    "bucex.inference.fit.gaussian_structural is a thin compatibility wrapper. "
    "Prefer bucex.api.fit.fit_gaussian_structural.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["fit_gaussian_structural"]
