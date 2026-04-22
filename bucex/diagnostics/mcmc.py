from __future__ import annotations

from typing import Dict

from ..core.results import PosteriorBundle


def acceptance_summary(bundle: PosteriorBundle) -> Dict[str, float]:
    return dict(bundle.acceptance)
