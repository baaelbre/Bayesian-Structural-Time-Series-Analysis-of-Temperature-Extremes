# %% simulator/uccle_dgev_laplace_crossval.py
from __future__ import annotations
"""
Uccle DGEV Laplace Cross-Validation (TXx, TXn, TNx, TNn, Precx; Monthly/Seasonal)
================================================================================

Thin Uccle wrapper around simulator.dgev_crossval.DGEVCrossValidator.

What this adds (Uccle conventions)
---------------------------------
- Robust "latest run" discovery (same as uccle_dgev_laplace_forecast.py):
    1) optimization.posterior_bundle.find_latest_run (posterior.npz runs)
    2) fallback recursive search for posterior*.npz
- Default root layout:
    results/uccle/<GROUP>/<SERIES>/<FREQ>/Laplace
  where GROUP inferred from SERIES prefix (TX/TN/Prec).
- Forced calendar origin default: start_date=1892-01-01 (unless meta has start_date or --start-date overrides)
- Optional --minima/--maxima override (for meta-based detection).

Everything else (rolling-origin splits, refit-per-split, metrics, per-split outputs)
is handled by the generic DGEVCrossValidator.

Note
----
The base DGEVCrossValidator uses titled/legend plots (like dlm_crossval). This wrapper
does not override plotting (keeps it short, like the dlm wrapper you asked for).
"""

import os
import sys
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, List

import numpy as np

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
from simulator.utils import _ensure_dir, _parse_date_ymd  # type: ignore

from simulator.dgev_crossval import DGEVCrossValidator  # type: ignore
from optimization.dgev_laplace_2 import Priors, SamplerConfig  # type: ignore


# =============================================================================
# Uccle root selection + robust latest discovery (mirrors uccle_dgev_laplace_forecast.py)
# =============================================================================
def _series_group(series: str) -> str:
    s = str(series).strip()
    if s.startswith("TX"):
        return "TX"
    if s.startswith("TN"):
        return "TN"
    if s.lower().startswith("prec"):
        return "Prec"
    head = "".join([c for c in s if c.isalpha()])
    return head if head else "misc"


def default_root(series: str, freq: str) -> str:
    return os.path.join("results", "uccle", _series_group(series), str(series), str(freq), "Laplace")


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


def resolve_bundle(*, target: Optional[str], series: str, root: Optional[str], freq: str) -> Any:
    if target:
        return load_posterior(target)

    search_root = root or default_root(series, freq)
    print(f"[info] searching latest posterior run under: {search_root!r}")

    run_path = find_latest_run(root=search_root)
    if run_path is not None:
        print(f"[info] using latest run: {run_path}")
        return load_posterior(run_path)

    npz_path = find_latest_posterior_npz(search_root)
    if npz_path is None:
        raise SystemExit(
            f"[error] No posterior runs found under {search_root!r}.\n"
            f"  → Tried find_latest_run() (posterior.npz runs) and recursive search (posterior*.npz).\n"
            f"  → Either run the sampler first, or provide --target."
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


def _parse_csv_floats(s: Optional[str], expected_len: Optional[int] = None) -> Optional[List[float]]:
    if s is None:
        return None
    txt = str(s).strip()
    if txt == "" or txt.lower() in ("none", "null"):
        return None
    txt = txt.strip("[](){}")
    parts = [p for p in txt.replace(",", " ").split() if p]
    vals = [float(p) for p in parts]
    if expected_len is not None and len(vals) != int(expected_len):
        raise ValueError(f"Expected {expected_len} floats, got {len(vals)} from {s!r}.")
    return vals


# =============================================================================
# One runner
# =============================================================================
def run_one(series: str, args: argparse.Namespace) -> None:
    bundle = resolve_bundle(target=args.target, series=series, root=args.root, freq=args.freq)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    if "y" not in draws:
        raise SystemExit("[error] posterior bundle must include draws['y'] (MODEL-scale y).")

    # --- meta: enforce Uccle-ish defaults without clobbering run info too much
    meta = dict(meta)
    meta["freq"] = str(args.freq)
    if "start_date" not in meta or not meta["start_date"]:
        meta["start_date"] = "1892-01-01"

    # period: keep if present (seasonal runs often have period=4); else default from freq
    if "period" not in meta or meta["period"] in (None, "", 0):
        meta["period"] = 12 if str(args.freq) == "Monthly" else 4

    # manual minima/maxima override (affects detection inside crossval/forecast)
    if args.minima:
        meta["minima"] = True
    if args.maxima:
        meta["minima"] = False

    # output
    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "crossval")
    _ensure_dir(out_dir)

    # start date override: CLI > meta
    sd = _parse_date_optional(args.start_date) if args.start_date else _parse_date_optional(meta.get("start_date"))

    # priors: meta first, then selected overrides
    pri = Priors()
    if isinstance(meta.get("priors", None), dict):
        try:
            pri = Priors(**dict(meta["priors"]))
        except Exception:
            pri = Priors()

    period = int(meta.get("period", 12) or 12)
    K = max(0, period - 1)

    # scalar priors
    if args.prior_a_sigma is not None:
        pri.a_sigma = float(args.prior_a_sigma)
    if args.prior_b_sigma is not None:
        pri.b_sigma = float(args.prior_b_sigma)
    if args.prior_xi_lower is not None:
        pri.xi_lower = float(args.prior_xi_lower)
    if args.prior_xi_upper is not None:
        pri.xi_upper = float(args.prior_xi_upper)

    if args.prior_m0_alpha is not None:
        pri.m0_alpha = float(args.prior_m0_alpha)
    if args.prior_P0_alpha is not None:
        pri.P0_alpha = float(args.prior_P0_alpha)
    if args.prior_m0_beta is not None:
        pri.m0_beta = float(args.prior_m0_beta)
    if args.prior_P0_beta is not None:
        pri.P0_beta = float(args.prior_P0_beta)
    if args.prior_P0_gamma is not None:
        pri.P0_gamma = float(args.prior_P0_gamma)

    if args.prior_m0_gamma is not None:
        pri.m0_gamma = _parse_csv_floats(args.prior_m0_gamma, expected_len=K)

    if args.prior_a_lambda is not None:
        pri.a_lambda = float(args.prior_a_lambda)
    if args.prior_b_lambda is not None:
        pri.b_lambda = float(args.prior_b_lambda)

    # Sampler config for refits
    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed_mcmc),
        progress=(not bool(args.no_progress)),
        progress_every=int(args.progress_every),
    )

    # optional safety knobs (only if the dataclass has them)
    for name, val, cast in [
        ("max_tries_block", args.max_tries_block, int),
        ("s_cap", args.s_cap, float),
        ("laplace_z_clip", args.laplace_z_clip, float),
    ]:
        if val is not None and hasattr(cfg, name):
            setattr(cfg, name, cast(val))

    # splits
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if not splits:
        raise SystemExit("[error] --splits parsed to an empty list.")

    # build CV runner (generic)
    cv = DGEVCrossValidator(
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
        ffbs_C0_scale=args.ffbs_C0_scale,
        ffbs_C0_A=args.ffbs_C0_A,
        ffbs_jitter=args.ffbs_jitter,
        sigma2_eff=args.sigma2_eff,
    )

    print(f"\n[info] series={series} | using posterior: {npz_path}")
    print(f"[info] writing crossval outputs to: {out_dir}")
    print(f"[info] T={cv.y.size}, period={cv.period}, minima={cv.minima}, start_date={cv.meta.get('start_date')}")

    cv.run(splits=splits, horizon=int(args.horizon))
    print("[done] cross-validation finished.")


# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Uccle rolling-origin cross-validation for DGEV Laplace (wrapper around simulator.dgev_crossval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # discovery
    p.add_argument("--target", type=str, default=None, help="Run dir or posterior.npz. If omitted, uses latest under --root.")
    p.add_argument("--root", type=str, default=None, help="Search root when --target is omitted (defaults to Uccle Laplace layout).")

    p.add_argument("--series", type=str, default="TXn", help="Series code (TXx, TXn, TNx, TNn, Precx, ...).")
    p.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run TXx, TXn, TNx, TNn sequentially (ignores --series).",
    )
    p.add_argument("--freq", type=str, choices=["Monthly", "Seasonal"], default="Monthly")

    # CV spec
    p.add_argument("--splits", type=str, default="0.6,0.8,0.9", help="Comma-separated split specs (idx, fraction, or date).")
    p.add_argument("--horizon", type=int, default=12 * 10, help="Forecast horizon in native block units.")

    # plot / output
    p.add_argument("--level", type=float, default=0.90, help="Forecast band level.")
    p.add_argument("--seed-forecast", type=int, default=123, help="Seed for predictive simulation.")
    p.add_argument("--window", type=int, default=12 * 50, help="Plot window (last N training points).")
    p.add_argument("--show", action="store_true", default=False)
    p.add_argument("--out", type=str, default=None, help="Output dir. Default: <run>/crossval")

    # start date (Uccle default if omitted)
    p.add_argument("--start-date", type=str, default=None, help="Override start date (YYYY / YYYY-MM / YYYY-MM-DD).")

    # minima/maxima override
    g = p.add_mutually_exclusive_group()
    g.add_argument("--minima", action="store_true", default=False, help="Force minima=True in meta (affects plotting/transform).")
    g.add_argument("--maxima", action="store_true", default=False, help="Force minima=False in meta.")

    # per-split MCMC
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed-mcmc", type=int, default=40)
    p.add_argument("--no-progress", action="store_true", default=False)
    p.add_argument("--progress-every", type=int, default=0)

    # optional safety knobs (SamplerConfig fields, if present)
    p.add_argument("--max-tries-block", type=int, default=None)
    p.add_argument("--s-cap", type=float, default=None)
    p.add_argument("--laplace-z-clip", type=float, default=None)

    # prior overrides (subset)
    p.add_argument("--prior-a-sigma", type=float, default=None)
    p.add_argument("--prior-b-sigma", type=float, default=None)
    p.add_argument("--prior-xi-lower", type=float, default=None)
    p.add_argument("--prior-xi-upper", type=float, default=None)

    p.add_argument("--prior-m0-alpha", type=float, default=None)
    p.add_argument("--prior-P0-alpha", type=float, default=None)
    p.add_argument("--prior-m0-beta", type=float, default=None)
    p.add_argument("--prior-P0-beta", type=float, default=None)
    p.add_argument("--prior-m0-gamma", type=str, default=None, help="CSV floats length period-1 (optional).")
    p.add_argument("--prior-P0-gamma", type=float, default=None)

    p.add_argument("--prior-a-lambda", type=float, default=None)
    p.add_argument("--prior-b-lambda", type=float, default=None)

    # DGEV refit knobs (try to mirror original run)
    p.add_argument("--ffbs-C0-scale", dest="ffbs_C0_scale", type=float, default=None)
    p.add_argument("--ffbs-C0-A", dest="ffbs_C0_A", type=float, default=None)
    p.add_argument("--ffbs-jitter", dest="ffbs_jitter", type=float, default=None)
    p.add_argument("--sigma2-eff", dest="sigma2_eff", type=float, default=None)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.all:
        for s in ["TXx", "TXn", "TNx", "TNn"]:
            run_one(s, args)
    else:
        run_one(str(args.series), args)


if __name__ == "__main__":
    main()
