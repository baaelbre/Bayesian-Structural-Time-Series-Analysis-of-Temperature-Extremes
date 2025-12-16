# %% simulator/uccle_dlm_plotter.py
from __future__ import annotations
"""
Uccle DLM Plotter (TXm, TNm, Precm; Seasonal / Monthly)
======================================================

Thin wrapper around the generic Gaussian DLM plotter:

    simulator/dlm_plotter.DLMPlotter

Supports:
  - Non-centred DLM with dummy seasonality + shrinkage priors (double-gamma, lasso, ...)
  - Older runs as long as they save a posterior bundle readable by
    optimization.posterior_bundle.load_posterior

Robust run discovery:
  1) optimization.posterior_bundle.find_latest_run (expects posterior.npz in run dirs)
  2) fallback: recursively search for posterior*.npz under the root and pick the most recent
     (by YYYYMMDD_HHMMSS in the path, else by file mtime).

Default Uccle roots:
  TXm, Seasonal   → results/uccle/TX/TXm/Seasonal
  TXm, Monthly    → results/uccle/TX/TXm/Monthly
  TNm, Seasonal   → results/uccle/TN/TNm/Seasonal
  TNm, Monthly    → results/uccle/TN/TNm/Monthly
  Precm, Seasonal → results/uccle/Prec/Precm/Seasonal
  Precm, Monthly  → results/uccle/Prec/Precm/Monthly

Kwarg overrides (repeatable):
  --overview-kw K=V
  --traceacf-kw K=V
  --states-kw   K=V
  --quick-kw    K=V

Nested dict kwargs: use dot-notation, e.g.
  --states-kw ylims.slope=(-2,2)
  --overview-kw ylims_mu=(0,30)

Examples
--------
# Latest TXm / Monthly
python uccle_dlm_plotter.py --series TXm --freq Monthly

# Explicit run directory or posterior file
python uccle_dlm_plotter.py --target path/to/run_or_posterior.npz

# Post-hoc burn & thinning
python uccle_dlm_plotter.py --series TNm --freq Monthly --burn 1000 --thin 5

# Override titles/labels
python uccle_dlm_plotter.py --series TXm --freq Monthly ^
  --overview-kw title_mu="Posterior $\\mu_t$ (TXm)" ^
  --states-kw title_slope="Trend $\\beta_t$" --states-kw slope_scale=120
"""

import os
import sys
import re
import math
import argparse
import ast
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt  # noqa: F401 (used by DLMPlotter internally)

# ---------------------------------------------------------------------
# Make project root importable: optimization/, simulator/, etc.
# ---------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
from simulator.dlm_plotter import DLMPlotter  # type: ignore


# ---------------------------------------------------------------------
# Paths / discovery
# ---------------------------------------------------------------------
def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def default_root(series: str, freq: str) -> str:
    base = "results/uccle"
    freq = str(freq)
    if series == "TXm":
        return os.path.join(base, "TX", "TXm", freq)
    if series == "TNm":
        return os.path.join(base, "TN", "TNm", freq)
    if series == "Precm":
        return os.path.join(base, "Prec", "Precm", freq)
    raise ValueError(f"Unknown series {series!r} for default root.")


def _extract_ts_from_path(path_str: str) -> Optional[float]:
    """Extract YYYYMMDD_HHMMSS from a path string and return epoch seconds, else None."""
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
    Recursively find latest posterior*.npz under root.

    Preference:
      1) largest YYYYMMDD_HHMMSS timestamp found in the *path*
      2) fallback: largest file modification time
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

    best = max(cands, key=key)
    return str(best)


def resolve_bundle(
    *,
    target: Optional[str],
    series: str,
    freq: str,
    root: Optional[str],
) -> Any:
    """
    Resolve a posterior bundle either from explicit --target or via searching.
    Returns the Bundle object from optimization.posterior_bundle.load_posterior.
    """
    if target:
        return load_posterior(target)

    search_root = root or default_root(series, freq)
    print(f"[info] searching latest posterior run under: {search_root!r}")

    # First try the official finder
    run_path = find_latest_run(root=search_root)
    if run_path is not None:
        print(f"[info] using latest run: {run_path}")
        return load_posterior(run_path)

    # Fallback: accept posterior_*.npz etc.
    npz_path = find_latest_posterior_npz(search_root)
    if npz_path is None:
        print(
            f"[error] No posterior runs found under {search_root!r}.\n"
            f"  → Tried find_latest_run() (posterior.npz) and recursive search (posterior*.npz).\n"
            f"  → Either run the sampler first, or provide --target."
        )
        raise SystemExit(1)

    print(f"[info] find_latest_run found nothing; using latest npz: {npz_path}")
    return load_posterior(npz_path)


# ---------------------------------------------------------------------
# Post-processing: burn-in + thinning (post-hoc)
# ---------------------------------------------------------------------
def apply_burn_thin(
    draws: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    burn: int = 0,
    thin: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply *extra* burn-in and thinning to all arrays whose first dimension equals
    the sample size inferred from draws["mu"].

    burn : discard first `burn` draws (>=0)
    thin : keep every `thin`-th draw after burn (>=1)
    """
    if "mu" not in draws:
        print("[warn] 'mu' not in draws; skipping burn/thin.")
        return draws, meta

    burn = int(burn or 0)
    thin = int(thin or 1)
    if burn < 0:
        raise ValueError(f"--burn must be >= 0, got {burn}")
    if thin < 1:
        raise ValueError(f"--thin must be >= 1, got {thin}")

    mu_arr = np.asarray(draws["mu"])
    if mu_arr.ndim < 2:
        print("[warn] 'mu' does not look like (n_samp, T); skipping burn/thin.")
        return draws, meta

    n_samp = int(mu_arr.shape[0])
    if burn >= n_samp:
        raise ValueError(f"--burn={burn} ≥ number of saved samples ({n_samp}).")

    idx = slice(burn, None, thin)
    n_used = int(math.ceil((n_samp - burn) / thin))

    print(f"[info] post-processing chains: raw n={n_samp}, burn={burn}, thin={thin} → used n={n_used}")

    for k, v in list(draws.items()):
        if not isinstance(v, np.ndarray):
            continue
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n_samp:
            draws[k] = arr[idx, ...]

    postproc = meta.get("postproc", {})
    postproc.update(
        {
            "extra_burn": int(burn),
            "thin": int(thin),
            "n_samples_raw": int(n_samp),
            "n_samples_used": int(np.asarray(draws["mu"]).shape[0]),
        }
    )
    meta["postproc"] = postproc
    return draws, meta


# ---------------------------------------------------------------------
# CLI kwarg overrides (same style as simulator/dlm_plotter.py)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Uccle DLM Plotter (TXm/TNm/Precm; Seasonal/Monthly)\n"
            "Uses the generic Gaussian DLMPlotter (non-centred, double-gamma, lasso, FS, etc.).\n"
            "Use --<section>-kw K=V (repeatable) to override ANY plot kwargs.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Posterior selection
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Run directory or posterior .npz. If provided, overrides --series/--freq/--root.",
    )
    p.add_argument(
        "--series",
        type=str,
        choices=["TXm", "TNm", "Precm"],
        default="Precm",
        help="Series code when searching by default roots (ignored if --target is given).",
    )
    p.add_argument(
        "--freq",
        type=str,
        choices=["Seasonal", "Monthly"],
        default="Monthly",
        help="Frequency tag when searching by default roots (ignored if --target is given).",
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Search root when --target is omitted (defaults to Uccle layout).",
    )

    # Plot options
    p.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--out", type=str, default=None, help="Directory to save figures (default: <run>/figures).")

    # Skips (match simulator/dlm_plotter.py)
    p.add_argument("--skip-overview", action="store_true", help="Skip overview figure.")
    p.add_argument("--skip-traceacf", action="store_true", help="Skip trace+hist+ACF panels.")
    p.add_argument("--skip-states", action="store_true", help="Skip separate state plots.")
    p.add_argument("--skip-quick", action="store_true", help="Skip quick 1x3 report.")

    # Post-hoc chain processing
    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    # Per-figure kwargs (repeatable)
    p.add_argument(
        "--overview-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_overview(...). Repeatable. Supports nested keys via dots.",
    )
    p.add_argument(
        "--traceacf-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_trace_acf_core(...). Repeatable. Supports nested keys via dots.",
    )
    p.add_argument(
        "--states-kw",
        action="append",
        default=["slope_scale=120"],  
        metavar="K=V",
        help="Override kwargs for plotter.figure_states_separate(...). Repeatable. Supports nested keys via dots.",
    )
    p.add_argument(
        "--quick-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.quick_report(...). Repeatable. Supports nested keys via dots.",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Resolve posterior bundle
    bundle = resolve_bundle(target=args.target, series=args.series, freq=args.freq, root=args.root)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    # Post-hoc burn/thin
    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, meta, burn=args.burn, thin=args.thin)

    # Output directory
    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)

    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving plots to: {out_dir}")

    # Plotter
    plotter = DLMPlotter(draws=draws, meta=meta, level=float(args.level))

    # Parse per-figure kw overrides
    overview_kw = _parse_kv_list(args.overview_kw)
    traceacf_kw = _parse_kv_list(args.traceacf_kw)
    states_kw = _parse_kv_list(args.states_kw)
    quick_kw = _parse_kv_list(args.quick_kw)

    # Generate figures (filenames default to the generic plotter, unless you override via *-kw fname=...)
    if not args.skip_overview:
        plotter.figure_overview(save_dir=out_dir, show=args.show, **overview_kw)

    if not args.skip_traceacf:
        plotter.figure_trace_acf_core(save_dir=out_dir, show=args.show, **traceacf_kw)

    if not args.skip_states:
        plotter.figure_states_separate(save_dir=out_dir, show=args.show, **states_kw)

    if not args.skip_quick:
        plotter.quick_report(save_dir=out_dir, show=args.show, **quick_kw)

    print("[done] plots written.")


if __name__ == "__main__":
    main()
