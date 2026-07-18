# %% simulator/dlm_plotter_utils.py
from __future__ import annotations

import os
import re
import math
import ast
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# Small utils (moved out of dlm_plotter.py)
# =============================================================================
def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _maybe(draws: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    v = draws.get(key, None)
    return None if v is None else np.asarray(v)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _robust_sd_from_mad(mad: float) -> float:
    # For Normal: MAD ≈ 0.6745 * sd  => sd ≈ MAD/0.6745
    if not np.isfinite(mad) or mad <= 0:
        return 0.0
    return float(mad) / 0.6745


def _acf(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 1:
        return np.array([1.0 if n == 1 else np.nan])
    x = x - np.mean(x)
    denom = float(np.dot(x, x)) + 1e-300
    L = int(min(max_lag, n - 1))
    ac = np.empty(L + 1, dtype=float)
    for k in range(L + 1):
        ac[k] = float(np.dot(x[: n - k], x[k:])) / denom
    return ac


def _ess(x: np.ndarray, max_lag: int = 200) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float(max(1, x.size))

    ac = _acf(x, max_lag=max_lag)
    if not np.all(np.isfinite(ac)) or ac.size <= 1:
        return float(len(x))

    s = 0.0
    for k in range(1, ac.size):
        if ac[k] <= 0:
            break
        s += 2.0 * ac[k]
    n = len(x)
    return float(n) / max(1e-12, (1.0 + s))


def _geweke_z(x: np.ndarray, first_frac: float = 0.1, last_frac: float = 0.5) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return float("nan")
    a = max(2, int(np.floor(first_frac * n)))
    b = max(2, int(np.floor(last_frac * n)))
    xa, xb = x[:a], x[n - b :]
    ma, mb = float(np.mean(xa)), float(np.mean(xb))
    va = float(np.var(xa, ddof=1)) / max(1, xa.size)
    vb = float(np.var(xb, ddof=1)) / max(1, xb.size)
    denom = math.sqrt(max(1e-300, va + vb))
    return (ma - mb) / denom


def _normalize_center(center: str) -> str:
    c = str(center).strip().lower()
    if c in ("median", "q50", "q0.5", "quantile", "quantile50"):
        return "median"
    if c in ("mean", "avg", "average", "expectation"):
        return "mean"
    raise ValueError("center must be one of {'median','mean'} (aliases allowed: q50/avg/...).")


def _center_label(center: str) -> str:
    return "mean" if _normalize_center(center) == "mean" else "median"


def _parse_date_ymd(s: str) -> datetime:
    s = str(s).strip()
    parts = [int(p) for p in s.split("-")]
    if len(parts) == 1:
        return datetime(parts[0], 1, 1)
    if len(parts) == 2:
        return datetime(parts[0], parts[1], 1)
    if len(parts) == 3:
        return datetime(parts[0], parts[1], parts[2])
    raise ValueError(f"Bad date string {s!r}. Use YYYY, YYYY-MM or YYYY-MM-DD.")


def _monthly_time_axis_from_meta(meta: Dict[str, Any], T: int) -> Tuple[Optional[np.ndarray], bool]:
    """
    Build a monthly axis of length T.

    Recognized meta keys: start_date, start, t0 (strings like YYYY-MM-DD).
    Returns (t, is_time) where t is an array suitable for matplotlib plotting.
    """
    start = meta.get("start_date", None)
    if start is None:
        start = meta.get("start", None)
    if start is None:
        start = meta.get("t0", None)
    if start is None:
        return None, False

    try:
        start_m = np.datetime64(str(start), "M")
        t_m = start_m + np.arange(int(T), dtype=int)  # monthly increments
        t_d = t_m.astype("datetime64[D]")
        t_py = t_d.astype("O")
        return np.asarray(t_py, dtype=object), True
    except Exception:
        return None, False


def _format_time_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
        lab.set_horizontalalignment("center")


def _extract_ts_from_path(path_str: str) -> Optional[float]:
    """
    If a path contains YYYYMMDD_HHMMSS, return a sortable timestamp float.
    """
    m = re.search(r"(\d{8})_(\d{6})", path_str)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def find_latest_posterior_npz(root: str) -> Optional[str]:
    """
    Fallback recursive search for posterior*.npz and pick most recent by embedded timestamp
    (YYYYMMDD_HHMMSS) if present, else by mtime.
    """
    root_p = Path(root)
    if not root_p.exists():
        return None
    cands = list(root_p.rglob("posterior*.npz"))
    if not cands:
        return None

    def key(p: Path) -> Tuple[int, float]:
        ts = _extract_ts_from_path(str(p))
        if ts is not None:
            return (1, ts)
        return (0, p.stat().st_mtime)

    return str(max(cands, key=key))


# =============================================================================
# CLI kwarg parsing helpers (moved out of dlm_plotter.py)
# =============================================================================
def _parse_value(raw: str):
    s = raw.strip()
    low = s.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _set_nested(d: dict, key: str, value):
    parts = [p for p in key.split(".") if p]
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_kv_list(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected K=V, got: {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        val = _parse_value(v)
        if "." in k:
            _set_nested(out, k, val)
        else:
            out[k] = val
    return out
