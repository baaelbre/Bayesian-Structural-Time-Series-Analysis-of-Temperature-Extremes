# %% simulator/uccle_dgev_laplace_cv.py
from __future__ import annotations
"""
Uccle DGEV Laplace Cross-Validation (TXx, TXn, TNx, TNn, Precx; Seasonal / Monthly)
=================================================================================

Thin wrapper around simulator.dgev_laplace_cv.DGEVCrossValidator with Uccle defaults.

Uccle defaults / conventions
----------------------------
- Forced origin: start_date = 1892-01-01 (unless --start-date overrides)
- Aggregation:
    * Monthly  -> period = 12
    * Seasonal -> period = 4
- Robust run discovery:
    1) optimization.posterior_bundle.find_latest_run (posterior.npz runs)
    2) fallback recursive search for posterior*.npz (simulator.utils.find_latest_posterior_npz)
- IMPORTANT: Your optimizer stores FFBS knobs under meta['knobs'].
  DGEVCrossValidator expects them as top-level meta keys, so we lift:
    ffbs_C0_scale, ffbs_C0_A, ffbs_jitter (if present), sigma2_eff

Plot conventions (Uccle-style)
-----------------------------
- Observations (train + held-out): black
- Forecast median + credible band: TX* red, TN* blue (else C0)
- No title, no legend

Outputs (default: <run>/crossval)
---------------------------------
<run>/crossval/
  cv_split_<label>_t<idx>_h<H>/
      forecast_fine.png
      metrics.json
      forecast_payload.npz
  crossval_summary.csv
  crossval_summary.json
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
from simulator.utils import _ensure_dir, _parse_date_ymd, find_latest_posterior_npz  # type: ignore

from simulator.dgev_laplace_cv import DGEVCrossValidator  # type: ignore
from optimization.dgev_laplace_2 import Priors, SamplerConfig  # type: ignore


# =============================================================================
# Uccle paths / discovery
# =============================================================================
def _parse_agg(agg: str) -> str:
    a = str(agg).strip().lower()
    if a.startswith("s"):
        return "Seasonal"
    if a.startswith("m"):
        return "Monthly"
    raise ValueError(f"Unknown aggregation {agg!r} (use Seasonal or Monthly).")


def default_root(series: str, agg: str) -> str:
    base = "results/uccle"
    s = str(series).strip()
    agg_dir = _parse_agg(agg)

    mapping = {
        "TXx": os.path.join(base, "TX", "TXx", agg_dir, "Laplace"),
        "TXn": os.path.join(base, "TX", "TXn", agg_dir, "Laplace"),
        "TNx": os.path.join(base, "TN", "TNx", agg_dir, "Laplace"),
        "TNn": os.path.join(base, "TN", "TNn", agg_dir, "Laplace"),
        "Precx": os.path.join(base, "Prec", "Precx", agg_dir, "Laplace"),
    }
    if s not in mapping:
        raise ValueError(f"Unknown series {series!r}.")
    return mapping[s]


def resolve_bundle(*, target: Optional[str], series: str, agg: str, root: Optional[str]) -> Any:
    if target:
        return load_posterior(target)

    search_root = root or default_root(series, agg)
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


def _default_ylabel(series: str) -> str:
    s = str(series).strip()
    if s.startswith(("TX", "TN")):
        return "T (°C)"
    if s.startswith(("Prec", "PREC")):
        return "Prec (mm)"
    return "y"


# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Uccle rolling-origin cross-validation for the DGEV Laplace NCP (wrapper around simulator.dgev_laplace_cv).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # discovery
    p.add_argument("--target", type=str, default=None, help="Run dir or posterior.npz. If omitted, uses latest under --root.")
    p.add_argument("--series", type=str, choices=["TXx", "TXn", "TNx", "TNn", "Precx"], default="TNn")
    p.add_argument("--agg", type=str, choices=["Seasonal", "Monthly"], default="Monthly")
    p.add_argument("--root", type=str, default=None, help="Search root when --target is omitted (defaults to Uccle layout).")

    # CV spec
    p.add_argument("--splits", type=str, default="0.6,0.8,0.9", help="Comma-separated split specs (idx, fraction, or date).")
    p.add_argument("--horizon", type=int, default=None, help="Forecast horizon in steps (months or seasons).")

    # plot / output
    p.add_argument("--level", type=float, default=0.90, help="Forecast band level.")
    p.add_argument("--seed-forecast", type=int, default=123, help="Seed for predictive simulation.")
    p.add_argument("--window", type=int, default=None, help="Plot window (last N training steps).")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--out", type=str, default=None, help="Output dir. Default: <run>/crossval")
    p.add_argument("--ylabel", type=str, default=None, help="Y-axis label for forecast plots.")

    # start date (Uccle default if omitted)
    p.add_argument("--start-date", type=str, default=None, help="Override start date (YYYY / YYYY-MM / YYYY-MM-DD).")

    # per-split MCMC (mirrors uccle_dlm_crossval)
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed-mcmc", type=int, default=40)
    p.add_argument("--no-progress", action="store_true", default=False)
    p.add_argument("--progress-every", type=int, default=0)

    # small prior overrides (optional)
    p.add_argument("--prior-a-sigma", type=float, default=None)
    p.add_argument("--prior-b-sigma", type=float, default=None)
    p.add_argument("--prior-xi-lower", type=float, default=None)
    p.add_argument("--prior-xi-upper", type=float, default=None)
    p.add_argument("--prior-a-lambda", type=float, default=None)
    p.add_argument("--prior-b-lambda", type=float, default=None)

    # DGEV FFBS knobs (optional overrides; otherwise read from meta/knobs)
    p.add_argument("--ffbs-C0-scale", type=float, default=None)
    p.add_argument("--ffbs-C0-A", type=float, default=None)
    p.add_argument("--ffbs-jitter", type=float, default=None)
    p.add_argument("--sigma2-eff", type=float, default=None)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    bundle = resolve_bundle(target=args.target, series=args.series, agg=args.agg, root=args.root)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    if "y" not in draws:
        raise SystemExit("[error] posterior bundle must include draws['y'].")

    agg_dir = _parse_agg(args.agg)
    period = 12 if agg_dir == "Monthly" else 4

    # Force Uccle origin + agg context, but keep other meta fields from the run
    meta = dict(meta)
    meta["start_date"] = "1892-01-01"
    meta["freq"] = agg_dir
    meta["agg"] = agg_dir
    meta["period"] = int(period)
    meta["series"] = str(args.series)  # drives TX/TN coloring + minima detection

    # IMPORTANT: lift knobs saved by your optimizer from meta['knobs'] to top-level keys
    knobs = meta.get("knobs", None)
    if isinstance(knobs, dict):
        for k in ("ffbs_C0_scale", "ffbs_C0_A", "ffbs_jitter", "sigma2_eff"):
            if k not in meta and k in knobs:
                meta[k] = knobs[k]

    # start date: forced meta unless user overrides
    sd = _parse_date_optional(args.start_date) if args.start_date else _parse_date_optional(meta.get("start_date"))

    # output
    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "crossval")
    _ensure_dir(out_dir)

    # defaults depend on agg
    horizon_default = (12 * 10) if agg_dir == "Monthly" else (4 * 10)
    window_default = (12 * 50) if agg_dir == "Monthly" else (4 * 50)

    horizon = int(args.horizon) if args.horizon is not None else int(horizon_default)
    window = int(args.window) if args.window is not None else int(window_default)
    ylabel = str(args.ylabel) if args.ylabel is not None else _default_ylabel(args.series)

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
    if args.prior_xi_lower is not None:
        pri.xi_lower = float(args.prior_xi_lower)
    if args.prior_xi_upper is not None:
        pri.xi_upper = float(args.prior_xi_upper)
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

    cv = DGEVCrossValidator(
        y=np.asarray(draws["y"], float).ravel(),  # MODEL scale series (TXn/TNn already negated by your runner)
        meta=meta,
        priors=pri,
        cfg=cfg,
        out_dir=out_dir,
        level=float(args.level),
        start_date_override=sd,
        seed_forecast=int(args.seed_forecast),
        window=int(window),
        show=bool(args.show),
        ylabel=ylabel,
        path_hint=str(npz_path),
        # optional overrides (if None, CV uses meta[...] which we lifted from meta['knobs'])
        ffbs_C0_scale=args.ffbs_C0_scale,
        ffbs_C0_A=args.ffbs_C0_A,
        ffbs_jitter=args.ffbs_jitter,
        sigma2_eff=args.sigma2_eff,
    )

    print(f"[info] using posterior: {npz_path}")
    print(f"[info] writing crossval outputs to: {out_dir}")
    print(
        f"[info] series={args.series}, agg={agg_dir}, T={cv.y.size}, period={cv.period}, "
        f"start_date={cv.meta.get('start_date')}, minima={cv.minima}"
    )
    # helpful: show the knobs the CV will use
    print(
        f"[info] knobs: ffbs_C0_scale={cv.meta.get('ffbs_C0_scale', None)} "
        f"ffbs_C0_A={cv.meta.get('ffbs_C0_A', None)} "
        f"ffbs_jitter={cv.meta.get('ffbs_jitter', None)} "
        f"sigma2_eff={cv.meta.get('sigma2_eff', None)}"
    )

    cv.run(splits=splits, horizon=int(horizon))
    print("[done] cross-validation finished.")


if __name__ == "__main__":
    main()
