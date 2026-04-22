from __future__ import annotations

import warnings

from ...api.fit import fit_gev_structural

warnings.warn(
    "bucex.inference.fit.gev_structural is a thin compatibility wrapper. "
    "Prefer bucex.api.fit.fit_gev_structural.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["fit_gev_structural"]
