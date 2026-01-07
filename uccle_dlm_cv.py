# %% simulator/uccle_dlm_crossval.py
from __future__ import annotations
"""
Uccle DLM Cross-Validation (TXm, TNm; Monthly)
=============================================

Thin wrapper around simulator.dlm_crossval.DLMCrossValidator with Uccle defaults.

Uccle defaults / conventions
----------------------------
- Monthly calendar: period=12
- Forced origin: start_date = 1892-01-01 (unless --start-date overrides)
- Robust run discovery (same as uccle_dlm_forecast.py):
    1) optimization.posterior_bundle.find_latest_run (posterior.npz runs)
    2) fallback recursive search for posterior*.npz

Uses the generic rolling-origin logic:
- loads y from the posterior bundle (draws['y'])
- for each split: refits DLM on prefix, forecasts H months, computes metrics,
  writes per-split artifacts + global summary.

Note
----
The base DLMCrossValidator plots include title + legend. This wrapper does NOT
override plotting to keep things short; if you want "no title/legend" here too,
I’ll do it with a tiny plot function swap (no class needed), but keeping base
behavior is simplest.

Outputs (default: <run>/crossval)
---------------------------------
<run>/crossval/
  cv_split_<label>_t<idx>_h<H>/
      forecast_fine.png
      metrics.json
  crossval_summary.csv
  crossval_summary.json
"""

import os
import sys
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
from simulator.utils import _ensure_dir, _parse_date_ymd  # type: ignore

from simulator.dlm_cv import DLMCrossValidator  # type: ignore
from optimization.dlm_3 import Priors, SamplerConfig  # type: ignore


# =============================================================================
# Uccle paths / discovery (copy of uccle_dlm_forecast.py logic)
# =============================================================================
def default_root(series: str) -> str:
    base = "results/uccle"
    if series == "TXm":
        return os.path.join(base, "TX", "TXm", "Monthly")
    if series == "TNm":
        return os.path.join(base, "TN", "TNm", "Monthly")
    if series == "Precm":
        return os.path.join(base, "Prec", "Precm", "Monthly")
    raise ValueError(f"Unknown series {series!r}.")


def _extract_ts_from_path(path_str: str) -> Optional[float]:
    m = re.search(r"(\d{8})_(\d{6})", path_str)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def find_latest_posterior_npz(root: str) -> Optional[str]:
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


def resolve_bundle(*, target: Optional[str], series: str, root: Optional[str]) -> Any:
    if target:
        return load_posterior(target)

    search_root = root or default_root(series)
    print(f"[info] searching latest posterior run under: {search_root!r}")

    run_path = find_latest_run(root=search_root)
    if run_path is not None:
        print(f"[info] using latest run: {run_path}")
        return load_posterior(run_path)

    npz_path = find_latest_posterior_npz(search_root)
    if npz_path is None:
        raise SystemExit(
            f"[error] No posterior runs found under {search_root!r}.\n"
            f"  → Tried find_latest_run() (posterior.npz runs) and recursive search (posterior*.npz)."
        )

    print(f"[info] find_latest_run found nothing; using latest npz: {npz_path}")
    return load_posterior(npz_path)


def _parse_date_optional(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    return _parse_date_ymd(ss)


# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Uccle rolling-origin cross-validation for the Gaussian DLM (wrapper around simulator.dlm_crossval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # discovery
    p.add_argument("--target", type=str, default=None, help="Run dir or posterior.npz. If omitted, uses latest under --root.")
    p.add_argument("--series", type=str, choices=["TXm", "TNm", "Precm"], default="TXm")
    p.add_argument("--root", type=str, default=None, help="Search root when --target is omitted (defaults to Uccle layout).")

    # CV spec
    p.add_argument("--splits", type=str, default="0.6,0.8,0.9", help="Comma-separated split specs (idx, fraction, or date).")
    p.add_argument("--horizon", type=int, default=12 * 10, help="Forecast horizon in months.")

    # plot / output
    p.add_argument("--level", type=float, default=0.90, help="Forecast band level.")
    p.add_argument("--seed-forecast", type=int, default=123, help="Seed for predictive simulation.")
    p.add_argument("--window", type=int, default=12 * 50, help="Plot window (last N training months).")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--out", type=str, default=None, help="Output dir. Default: <run>/crossval")

    # start date (Uccle default if omitted)
    p.add_argument("--start-date", type=str, default=None, help="Override start date (YYYY / YYYY-MM / YYYY-MM-DD).")

    # per-split MCMC
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed-mcmc", type=int, default=40)
    p.add_argument("--no-progress", action="store_true", default=False)
    p.add_argument("--progress-every", type=int, default=0)

    # small prior overrides (optional)
    p.add_argument("--prior-a-sigma", type=float, default=None)
    p.add_argument("--prior-b-sigma", type=float, default=None)
    p.add_argument("--prior-a-lambda", type=float, default=None)
    p.add_argument("--prior-b-lambda", type=float, default=None)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    bundle = resolve_bundle(target=args.target, series=args.series, root=args.root)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    if "y" not in draws:
        raise SystemExit("[error] posterior bundle must include draws['y'].")

    # Force Uccle monthly origin + period (matches uccle_dlm_forecast.py)
    meta = dict(meta)
    meta["start_date"] = "1892-01-01"
    meta["freq"] = "Monthly"
    meta["period"] = 12

    # start date: forced meta unless user overrides
    sd = _parse_date_optional(args.start_date) if args.start_date else _parse_date_optional(meta.get("start_date"))

    # output
    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "crossval")
    _ensure_dir(out_dir)

    # priors: start from meta if available, then override selected fields
    pri = Priors()
    if isinstance(meta.get("priors", None), dict):
        try:
            pri = Priors(**dict(meta["priors"]))
        except Exception:
            pri = Priors()

    if args.prior_a_sigma is not None:
        pri.a_sigma = float(args.prior_a_sigma)
    if args.prior_b_sigma is not None:
        pri.b_sigma = float(args.prior_b_sigma)
    if args.prior_a_lambda is not None:
        pri.a_lambda = float(args.prior_a_lambda)
    if args.prior_b_lambda is not None:
        pri.b_lambda = float(args.prior_b_lambda)

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed_mcmc),
        progress=(not bool(args.no_progress)),
        progress_every=int(args.progress_every),
    )

    # splits
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if not splits:
        raise SystemExit("[error] --splits parsed to an empty list.")

    # build CV runner using the generic class
    cv = DLMCrossValidator(
        y=np.asarray(draws["y"], float).ravel(),
        meta=meta,
        priors=pri,
        cfg=cfg,
        out_dir=out_dir,
        level=float(args.level),
        start_date_override=sd,
        seed_forecast=int(args.seed_forecast),
        window=int(args.window),
        show=bool(args.show),
    )

    print(f"[info] using posterior: {npz_path}")
    print(f"[info] writing crossval outputs to: {out_dir}")
    print(f"[info] series={args.series}, T={cv.y.size}, period={cv.period}, start_date={cv.meta.get('start_date')}")

    cv.run(splits=splits, horizon=int(args.horizon))
    print("[done] cross-validation finished.")


if __name__ == "__main__":
    main()
