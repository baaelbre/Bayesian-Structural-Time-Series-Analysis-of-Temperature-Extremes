# %% simulator/uccle_dlm_plotter.py
from __future__ import annotations
"""
Uccle DLM Plotter (TXm, TNm; Monthly)
"""

import os
import sys
import re
import math
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
from simulator.dlm_plotter import DLMPlotter  # type: ignore
import matplotlib as mpl

mpl.rcParams.update({
    # global base font
    "font.size": 18,

    # titles + axis labels
    "axes.titlesize": 16,
    "axes.labelsize": 16,

    # tick labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    # legends
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
})

# Prefer using shared utils when available (keeps behavior aligned with DLMPlotter)
try:
    from simulator.utils import _ensure_dir, find_latest_posterior_npz, _parse_kv_list  # type: ignore
except Exception:
    # Minimal fallbacks (kept tiny; wrapper still works if utils import fails)
    def _ensure_dir(path: Optional[str]) -> None:
        if path:
            os.makedirs(path, exist_ok=True)

    def find_latest_posterior_npz(root: str) -> Optional[str]:
        root_p = Path(root)
        if not root_p.exists():
            return None
        cands = list(root_p.rglob("posterior*.npz"))
        if not cands:
            return None
        return str(max(cands, key=lambda p: p.stat().st_mtime))

    def _parse_kv_list(items: List[str]) -> Dict[str, Any]:
        # very small K=V parser; keep consistent with your other wrappers where possible
        import ast

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
# Uccle layout helpers
# ---------------------------------------------------------------------
def default_root(series: str) -> str:
    base = "results/uccle"
    if series == "TXm":
        return os.path.join(base, "TX", "TXm", "Monthly")
    if series == "TNm":
        return os.path.join(base, "TN", "TNm", "Monthly")
    if series == "Precm":
        return os.path.join(base, "Prec", "Precm", "Monthly")
    raise ValueError(f"Unknown series {series!r} for default root.")


def resolve_bundle(*, target: Optional[str], series: str, root: Optional[str]) -> Any:
    """
    Robust run discovery:
      1) find_latest_run(root=...) -> expects a run dir / posterior.npz
      2) fallback find_latest_posterior_npz(root=...) -> recursive posterior*.npz
    """
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

    meta = dict(meta)
    postproc = dict(meta.get("postproc", {}) or {})
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
# Series colors (fixed)
# ---------------------------------------------------------------------
def series_color(series: str) -> str:
    s = str(series).upper()
    if s.startswith("TX"):
        return "red"
    if s.startswith("TN"):
        return "blue"
    if s.startswith("PREC"):
        return "tab:green"
    return "C0"


def default_slope_scale(series: str) -> float:
    """
    For temperature DLMs (TXm/TNm), slope is usually per month; default to per-decade scale.
    For precipitation, leave slope on its native scale unless overridden.
    """
    return 120.0 if series in ("TXm", "TNm") else 1.0


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Uccle DLM Plotter (TXm/TNm/Precm; Monthly)\n"
            "Uses simulator.dlm_plotter.DLMPlotter.\n"
            "Calendar origin is forced to 1892-01-01 monthly.\n"
            "Prints level/slope summaries at chosen years (January).\n"
            "Prints static parameter summaries (median/CI) by default.\n"
            "Seasonal diagnostics are produced via DLMPlotter.\n"
            "Use --<section>-kw K=V (repeatable) to override plot kwargs.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--target", type=str, default=None, help="Run directory or posterior .npz. Overrides --series/--root.")
    p.add_argument("--series", type=str, choices=["TXm", "TNm", "Precm"], default="TNm")
    p.add_argument("--root", type=str, default=None, help="Search root when --target is omitted (defaults to Uccle layout).")

    p.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--out", type=str, default=None, help="Directory to save figures (default: <run>/figures).")

    p.add_argument("--skip-overview", action="store_true", help="Skip overview figure.")
    p.add_argument("--skip-traceacf", action="store_true", help="Skip trace+hist+ACF panels.")
    p.add_argument("--skip-states", action="store_true", help="Skip separate state plots.")
    p.add_argument("--skip-quick", action="store_true", help="Skip quick report.")
    p.add_argument("--skip-qhist", action="store_true", help="Skip separate log10(Q) histogram (no title).")

    # seasonal diagnostics
    p.add_argument("--skip-seasonal-patterns", action="store_true", help="Skip seasonal pattern figure.")
    p.add_argument("--skip-seasonal-variance", action="store_true", help="Skip seasonal variance-by-year figure.")
    p.add_argument(
        "--seasonal-years", type=str, default="auto",
        help="Comma-separated calendar years for seasonal pattern plot, or 'auto'. Example: 1950,1980,2020"
    )

    # post-hoc chain processing
    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    # summaries
    p.add_argument("--summary-years", type=str, default="1892,1950,2020", help="Comma-separated calendar years (January).")
    p.add_argument("--summary-slope-scale", type=float, default=None, help="Override slope scale in printed level/slope summary.")

    # static parameter printing
    p.add_argument("--print-static", action="store_true", default=True, help="Print static parameter summaries (median/CI).")
    p.add_argument("--no-print-static", action="store_true", default=False, help="Disable printing static parameter summaries.")
    p.add_argument("--static-level", type=float, default=None, help="Credible level for static params (default: --level).")
    p.add_argument("--static-center", type=str, default="median", choices=["median", "mean"])
    p.add_argument("--static-digits", type=int, default=4)
    p.add_argument("--static-max-cols", type=int, default=None)
    p.add_argument("--static-no-diag", action="store_true", default=False)
    p.add_argument("--print-log-process", action="store_true", default=True)
    p.add_argument("--log-eps", type=float, default=1e-20)

    # kwargs overrides for plotting (defaults enforce Uccle style)
    p.add_argument("--overview-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--traceacf-kw", action="append", default=[], metavar="K=V")
    p.add_argument(
        "--states-kw",
        action="append",
        default=[
            # enforce "no titles" by default
            "title_level=''",
            "title_slope=''",
            "title_seasonality=''",
            # enforce "no legend" by default
            "show_legend=False",
            # common Uccle scaling: per-decade slope for temperatures (overridden in main() if needed)
            # keep a placeholder; main() will set slope_scale if user didn't specify it
        ],
        metavar="K=V",
    )
    p.add_argument("--quick-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--qhist-kw", action="append", default=[], metavar="K=V")

    # seasonal kw
    p.add_argument("--seasonal-patterns-kw", action="append", default=["title=''"], metavar="K=V")
    p.add_argument("--seasonal-variance-kw", action="append", default=["title=''"], metavar="K=V")

    return p


def _parse_year_list(s: str) -> List[int]:
    out: List[int] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def main() -> None:
    args = build_argparser().parse_args()

    bundle = resolve_bundle(target=args.target, series=args.series, root=args.root)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    # Force Uccle monthly calendar origin
    meta = dict(meta)
    meta["start_date"] = "1892-01-01"
    meta["freq"] = "Monthly"
    meta.setdefault("period", 12)
    meta.setdefault("series", args.series)

    # Post-hoc burn/thin
    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, meta, burn=args.burn, thin=args.thin)

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)

    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving plots to: {out_dir}")
    print(f"[info] calendar origin forced to: {meta['start_date']} (monthly)")

    plotter = DLMPlotter(draws=draws, meta=meta, level=float(args.level))

    # ---- PRINT level/slope summaries at chosen years (January of each year) ----
    try:
        years = _parse_year_list(args.summary_years)
        if years:
            times = [f"{y}-01" for y in years]
            sc = float(args.summary_slope_scale) if (args.summary_slope_scale is not None) else default_slope_scale(args.series)
            plotter.print_level_slope_at(times=times, slope_scale=sc)
    except Exception as e:
        print(f"[warn] could not print level/slope summaries: {e}")

    # ---- PRINT static params (median/CI) ----
    do_print_static = bool(args.print_static) and (not bool(args.no_print_static))
    if do_print_static:
        try:
            plotter.print_static_params(
                level=(float(args.static_level) if args.static_level is not None else None),
                center=str(args.static_center),
                digits=int(args.static_digits),
                max_vector_cols=(int(args.static_max_cols) if args.static_max_cols is not None else None),
                include_diagnostics=(not bool(args.static_no_diag)),
                include_log_process=bool(args.print_log_process),
                log_eps=float(args.log_eps),
            )
        except Exception as e:
            print(f"[warn] could not print static parameter summaries: {e}")

    # ---- parse kw overrides ----
    overview_kw = _parse_kv_list(args.overview_kw)
    traceacf_kw = _parse_kv_list(args.traceacf_kw)
    states_kw = _parse_kv_list(args.states_kw)
    quick_kw = _parse_kv_list(args.quick_kw)
    qhist_kw = _parse_kv_list(args.qhist_kw)
    seasonal_patterns_kw = _parse_kv_list(args.seasonal_patterns_kw)
    seasonal_variance_kw = _parse_kv_list(args.seasonal_variance_kw)

    # Fixed TX/TN colors (line + shading)
    col = series_color(args.series)
    for d in (overview_kw, states_kw, quick_kw):
        d.setdefault("color", col)

    # Enforce Uccle defaults for states unless user explicitly overrides
    states_kw.setdefault("title_level", "")
    states_kw.setdefault("title_slope", "")
    states_kw.setdefault("title_seasonality", "")
    states_kw.setdefault("show_legend", False)
    states_kw.setdefault("slope_scale", default_slope_scale(args.series))
    states_kw.setdefault("ylims", {"slope": (-0.5, 1.5)})

    # Enforce "no title" defaults for seasonal figs (unless user overrides)
    seasonal_patterns_kw.setdefault("title", "")
    seasonal_variance_kw.setdefault("title", "")

    # seasonal-years parsing
    sy = str(args.seasonal_years).strip()
    if sy.lower() == "auto" or sy == "":
        seasonal_years: Union[str, List[int]] = "auto"
    else:
        seasonal_years = [int(z) for z in sy.split(",") if z.strip() != ""]

    # ---- figures ----
    if not args.skip_overview:
        plotter.figure_overview(save_dir=out_dir, show=args.show, **overview_kw)

    if not args.skip_traceacf:
        plotter.figure_trace_acf_core(save_dir=out_dir, show=args.show, **traceacf_kw)

    if not args.skip_states:
        plotter.figure_states_separate(save_dir=out_dir, show=args.show, **states_kw)

    if not args.skip_quick:
        plotter.quick_report(save_dir=out_dir, show=args.show, **quick_kw)

    if not args.skip_qhist:
        # DLMPlotter.figure_process_variances_hist has "NO title" requirement by design
        plotter.figure_process_variances_hist(save_dir=out_dir, show=args.show, **qhist_kw)

    # ---- seasonal diagnostics (delegated to DLMPlotter) ----
    if not args.skip_seasonal_patterns:
        plotter.figure_seasonal_patterns(
            years=seasonal_years,
            save_dir=out_dir,
            show=args.show,
            **seasonal_patterns_kw,
        )

    if not args.skip_seasonal_variance:
        plotter.figure_seasonal_variance(
            save_dir=out_dir,
            show=args.show,
            **seasonal_variance_kw,
        )

    print("[done] plots written.")


if __name__ == "__main__":
    main()