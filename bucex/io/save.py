# src/sts_extremes/io/save.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..simulate.statespace import SimResult


def save_simresult_npz(
    path: str | Path,
    sim: SimResult,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save simulation result to:
      - <path>.npz : arrays
      - <path>.json: metadata (including state_names)

    Usage:
      save_simresult_npz("results/sim/ucl_TXx_run1", sim, meta={...})
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        x=sim.x,
        mu=sim.mu,
        y=sim.y,
        state_names=np.array(sim.state_names, dtype=object),
    )

    payload = {"state_names": list(sim.state_names)}
    if meta:
        payload["meta"] = meta

    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return npz_path


def _json_default(obj: Any):
    # allow numpy scalars/arrays, dataclasses, Path
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    try:
        return asdict(obj)  # dataclass
    except Exception:
        return str(obj)