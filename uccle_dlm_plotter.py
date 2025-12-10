from __future__ import annotations
"""
Uccle DLM Plotter (TXm, TNm, Precm; Seasonal / Monthly)
======================================================

Wrapper around the generic Gaussian DLM plotter:

    simulator/dlm_plotter.DLMPlotter

It works for:
  * The new non-centred DLM with dummy monthly seasonality and
    double-gamma global-local priors (DLMGibbsConjugate),
  * Older FS-style / harmonic runs, as long as they save a standard
    posterior bundle via optimization.posterior_bundle.

Default Uccle roots:
  TXm, Seasonal  → results/uccle/TX/TXm/Seasonal
  TXm, Monthly   → results/uccle/TX/TXm/Monthly
  TNm, Seasonal  → results/uccle/TN/TNm/Seasonal
  TNm, Monthly   → results/uccle/TN/TNm/Monthly
  Precm, Seasonal→ results/uccle/Prec/Precm/Seasonal
  Precm, Monthly → results/uccle/Prec/Precm/Monthly

Examples
--------
# Latest TXm / Monthly (double-gamma DLM with dummies):
python uccle_dlm_plotter.py --series TXm --freq Monthly

# Latest TNm / Monthly:
python uccle_dlm_plotter.py --series TNm --freq Monthly

# Seasonal roots (works for any posterior bundle):
python uccle_dlm_plotter.py --series TXm --freq Seasonal

# Explicit run directory or posterior.npz:
python uccle_dlm_plotter.py --target path/to/run_or_posterior.npz

# With extra post-hoc burn-in and thinning:
python uccle_dlm_plotter.py --series TNm --freq Monthly --burn 1000 --thin 5
"""

import os
import sys
import math
import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt  # noqa: F401  (used by DLMPlotter internally)

# ---------------------------------------------------------------------
# Make parent dir importable: optimization/, simulator/, etc.
# ---------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
from simulator.dlm_plotter import DLMPlotter  # type: ignore


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _default_root(series: str, freq: str) -> str:
    """
    Default root for given series & frequency.

    series in {TXm, TNm, Precm}
    freq   in {Seasonal, Monthly}

    Layout:
      TXm, Seasonal  → results/uccle/TX/TXm/Seasonal
      TXm, Monthly   → results/uccle/TX/TXm/Monthly
      TNm, Seasonal  → results/uccle/TN/TNm/Seasonal
      TNm, Monthly   → results/uccle/TN/TNm/Monthly
      Precm, Seasonal→ results/uccle/Prec/Precm/Seasonal
      Precm, Monthly → results/uccle/Prec/Precm/Monthly
    """
    base = "results/uccle"
    freq = str(freq)
    if series == "TXm":
        return os.path.join(base, "TX", "TXm", freq)
    if series == "TNm":
        return os.path.join(base, "TN", "TNm", freq)
    if series == "Precm":
        return os.path.join(base, "Prec", "Precm", freq)
    raise ValueError(f"Unknown series {series!r} for default root.")


# ---------------------------------------------------------------------
# Post-processing: burn-in + thinning (post-hoc)
# ---------------------------------------------------------------------
def apply_burn_thin(
    draws: Dict[str, Any],
    meta: Dict[str, Any],
    burn: int = 0,
    thin: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply *extra* burn-in and thinning to all arrays whose first dimension
    matches the sample size (inferred from draws["mu"]).

    burn : number of *initial* draws to discard (>=0)
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
    n_used = math.ceil((n_samp - burn) / thin)

    print(
        f"[info] post-processing chains: raw n={n_samp}, burn={burn}, "
        f"thin={thin} → used n={n_used}"
    )

    for k, v in list(draws.items()):
        if not isinstance(v, np.ndarray):
            continue
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n_samp:
            draws[k] = arr[idx, ...]

    # Book-keeping in meta
    postproc = meta.get("postproc", {})
    postproc.update(
        {
            "extra_burn": burn,
            "thin": thin,
            "n_samples_raw": n_samp,
            "n_samples_used": int(np.asarray(draws["mu"]).shape[0]),
        }
    )
    meta["postproc"] = postproc
    return draws, meta


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Uccle DLM Plotter (TXm/TNm/Precm; Seasonal/Monthly)\n"
            "Uses the generic Gaussian DLMPlotter (non-centred, double-gamma, FS, etc.)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # How to find the posterior bundle
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Run directory or posterior.npz. "
            "If provided, overrides --series/--freq/--root and uses that run directly."
        ),
    )
    p.add_argument(
        "--series",
        type=str,
        choices=["TXm", "TNm", "Precm"],
        default="TNm",
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
        help=(
            "Search root when --target is omitted. "
            "If not given, a default root is built from --series and --freq."
        ),
    )

    # Plot / posterior settings
    p.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    p.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show figures interactively in addition to saving them.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Directory to save figures (default: <run>/figures).",
    )

    # Skip toggles (match simulator/dlm_plotterCLI)
    p.add_argument("--skip-overview", action="store_true", help="Skip overview figure.")
    p.add_argument("--skip-traceacf", action="store_true", help="Skip trace+hist+ACF panels.")
    p.add_argument("--skip-states", action="store_true", help="Skip state ribbons.")
    p.add_argument("--skip-corr", action="store_true", help="Skip correlation scatter matrices.")
    p.add_argument("--skip-quick", action="store_true", help="Skip quick 1x3 report.")

    # Extra post-hoc burn-in and thinning
    p.add_argument(
        "--burn",
        type=int,
        default=0,
        help="Extra burn-in draws to discard from the front (post-hoc).",
    )
    p.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Extra thinning factor k: keep every k-th draw after burn-in (post-hoc).",
    )

    args = p.parse_args()

    # -----------------------------------------------------------------
    # Resolve which run to use
    # -----------------------------------------------------------------
    if args.target:
        # Explicit run directory or posterior.npz
        bundle = load_posterior(args.target)
    else:
        # Use default Uccle roots or user-supplied root
        if args.root:
            search_root = args.root
        else:
            search_root = _default_root(args.series, args.freq)

        print(f"[info] searching latest posterior run under: {search_root!r}")
        run_path = find_latest_run(root=search_root)
        if run_path is None:
            print(
                f"[error] No posterior runs found under {search_root!r}.\n"
                f"  → Either run the DLM sampler first, or provide --target."
            )
            sys.exit(1)
        print(f"[info] using latest run: {run_path}")
        bundle = load_posterior(run_path)

    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    # -----------------------------------------------------------------
    # Apply extra burn/thin (post-hoc)
    # -----------------------------------------------------------------
    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, meta, burn=args.burn, thin=args.thin)

    # -----------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------
    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)
    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving plots to: {out_dir}")

    # -----------------------------------------------------------------
    # Plot via generic DLMPlotter
    # -----------------------------------------------------------------
    plotter = DLMPlotter(draws=draws, meta=meta, level=float(args.level))

    if not args.skip_overview:
        plotter.figure_overview(save_dir=out_dir, fname_prefix="overview", show=args.show)

    if not args.skip_traceacf:
        plotter.figure_trace_acf_core(save_dir=out_dir, show=args.show)

    if not args.skip_states:
        plotter.figure_states(save_dir=out_dir, fname_prefix="states", show=args.show)

    if not args.skip_corr:
        plotter.figure_correlations(save_dir=out_dir, show=args.show)

    if not args.skip_quick:
        plotter.quick_report(save_dir=out_dir, fname_prefix="quick_report", show=args.show)

    print("[done] plots written.")
