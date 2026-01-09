# %% simulator/uccle_dgev_laplace_return_levels.py
from __future__ import annotations
"""
Uccle wrapper for DGEV Laplace return levels/periods.

- Default root: results/uccle/<GROUP>/<SERIES>/<FREQ>/Laplace
- Robust latest-run discovery (find_latest_run + fallback posterior*.npz)
- Applies --burn/--thin (post-hoc) correctly
- Uses the generic engine simulator.dgev_return_levels.DGEVReturnLevels
- Uccle colors: TX red, TN blue, Prec default
"""

import os
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
from simulator.dgev_laplace_return_levels import DGEVReturnLevels, DGEVReturnLevelsConfig  # type: ignore


def _series_group(series: str) -> str:
    s = series.strip()
    if s.startswith("TX"): return "TX"
    if s.startswith("TN"): return "TN"
    if s.lower().startswith("prec"): return "Prec"
    head = "".join([c for c in s if c.isalpha()])
    return head or "misc"

def default_root(series: str, freq: str) -> str:
    return os.path.join("results", "uccle", _series_group(series), series, freq, "Laplace")

def _extract_ts(path_str: str) -> Optional[float]:
    m = re.search(r"(\d{8})_(\d{6})", path_str)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
    except Exception:
        return None

def _find_latest_npz(root: str) -> Optional[str]:
    rp = Path(root)
    if not rp.exists():
        return None
    cands = list(rp.rglob("posterior*.npz"))
    if not cands:
        return None

    def key(p: Path) -> Tuple[int, float]:
        ts = _extract_ts(str(p))
        return (1, ts) if ts is not None else (0, p.stat().st_mtime)

    return str(max(cands, key=key))

def resolve_bundle(*, target: Optional[str], series: str, root: Optional[str], freq: str):
    if target:
        return load_posterior(target)
    search_root = root or default_root(series, freq)
    run = find_latest_run(root=search_root)
    if run is not None:
        return load_posterior(run)
    npz = _find_latest_npz(search_root)
    if npz is None:
        raise SystemExit(f"No posterior found under {search_root!r}. Provide --target or run sampler.")
    return load_posterior(npz)

def _series_color(series: str) -> str:
    s = series.upper()
    if s.startswith("TX"): return "red"
    if s.startswith("TN"): return "blue"
    return "C0"

def _apply_burn_thin(draws: Dict[str, Any], burn: int, thin: int) -> Dict[str, Any]:
    burn = int(max(0, burn))
    thin = int(max(1, thin))
    if burn == 0 and thin == 1:
        return draws

    # infer S from sigma if possible
    S = None
    if "sigma" in draws:
        try:
            S = int(np.asarray(draws["sigma"]).shape[0])
        except Exception:
            S = None

    out: Dict[str, Any] = {}
    for k, v in draws.items():
        a = np.asarray(v)
        if S is not None and a.ndim >= 1 and a.shape[0] == S:
            out[k] = a[burn::thin].copy()
        else:
            out[k] = v
    return out

def _parse_csv_floats(s: Optional[str]) -> List[float]:
    if s is None:
        return []
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out

def run_one(series: str, args: argparse.Namespace) -> None:
    b = resolve_bundle(target=args.target, series=series, root=args.root, freq=args.freq)

    draws = _apply_burn_thin(b.draws, args.burn, args.thin)
    meta = dict(b.meta) if isinstance(b.meta, dict) else {}
    meta.setdefault("start_date", "1892-01-01")  # Uccle default

    # minima/maxima override
    if args.minima: meta["minima"] = True
    if args.maxima: meta["minima"] = False

    Ns = tuple(int(round(x)) for x in _parse_csv_floats(args.N) if x > 1)
    thr = tuple(_parse_csv_floats(args.threshold)) if args.threshold else ()

    out_dir = args.out or os.path.join(os.path.dirname(b.npz_path), "return_levels")
    if args.all and args.out:
        out_dir = os.path.join(args.out, series)
    os.makedirs(out_dir, exist_ok=True)

    cfg = DGEVReturnLevelsConfig(
        level=float(args.level),
        Ns=Ns,
        thresholds=thr,
        period_unit=str(args.period_unit),
        start_date=str(args.start_date) if args.start_date else None,
        skip_annual=bool(args.skip_annual),
        window_blocks=int(args.window_blocks),
        window_years=int(args.window_years),
        log10_period=bool(args.log10_period),
        show=bool(args.show),
        no_title=True,
        color=_series_color(series),
        obs_color="0.25",
        prefix=f"{series}_",
    )

    rl = DGEVReturnLevels(draws=draws, meta=meta, npz_path=b.npz_path, cfg=cfg, out_dir=out_dir)
    rl.run()
    print(f"[done] {series} -> {out_dir}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Uccle DGEV Laplace return levels/periods.",
    )
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--out", type=str, default=None)

    p.add_argument("--series", type=str, default="TNx")
    p.add_argument("--all", action="store_true", default=False)
    p.add_argument("--freq", choices=["Monthly", "Seasonal"], default="Monthly")

    p.add_argument("--level", type=float, default=0.90)
    p.add_argument("--N", type=str, default="20,50,100")
    p.add_argument("--threshold", type=str, default="35,37,39,39.7")

    p.add_argument("--period-unit", choices=["blocks", "years"], default="years")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--skip-annual", action="store_true", default=False)

    p.add_argument("--burn", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)

    p.add_argument("--window-blocks", type=int, default=0)
    p.add_argument("--window-years", type=int, default=0)

    p.add_argument("--log10-period", action="store_true", default=True)
    p.add_argument("--show", action="store_true", default=False)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--minima", action="store_true", default=False)
    g.add_argument("--maxima", action="store_true", default=False)
    return p

def main() -> None:
    args = build_argparser().parse_args()
    if args.all:
        for s in ["TXx", "TXn", "TNx", "TNn", "Precx"]:
            run_one(s, args)
    else:
        run_one(args.series, args)

if __name__ == "__main__":
    main()
