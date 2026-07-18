# src/sts_extremes/io/load.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..simulate.statespace import SimResult


def load_simresult_npz(path: str | Path) -> Tuple[SimResult, Dict[str, Any]]:
    """
    Load:
      - <path>.npz arrays
      - <path>.json metadata
    """
    path = Path(path)
    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")

    z = np.load(npz_path, allow_pickle=True)
    state_names = tuple(z["state_names"].tolist())

    sim = SimResult(
        x=np.asarray(z["x"]),
        mu=np.asarray(z["mu"]),
        y=np.asarray(z["y"]),
        state_names=state_names,
    )

    meta: Dict[str, Any] = {}
    if json_path.exists():
        meta = json.loads(json_path.read_text())

    return sim, meta