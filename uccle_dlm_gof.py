# %% simulator/uccle_dlm_gof.py
from __future__ import annotations
"""simulator/uccle_dlm_gof.py
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simulator.dlm_gof import DLMGoodnessOfFit, GOFConfig  # type: ignore

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
# ---------------------------------------------------------------------
# Uccle defaults
# ---------------------------------------------------------------------
def default_root(series: str) -> str:
    base = "results/uccle"
    if series == "TXm":
        return os.path.join(base, "TX", "TXm", "Monthly")
    if series == "TNm":
        return os.path.join(base, "TN", "TNm", "Monthly")
    raise ValueError(f"Unknown series {series!r} for default root.")


def series_color(series: str) -> str:
    s = str(series).upper()
    if s.startswith("TX"):
        return "red"
    if s.startswith("TN"):
        return "blue"
    return "C0"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Uccle DLM GOF wrapper (PIT + PPC KS) using simulator.dlm_gof (class-based).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--target", type=str, default=None, help="Run directory or posterior.npz path.")
    p.add_argument("--series", type=str, choices=["TXm", "TNm"], default="TNm", help="Series used for default root + colors.")
    p.add_argument("--root", type=str, default=None, help="Search root if --target omitted (defaults to Uccle layout).")

    p.add_argument("--out", type=str, default=None, help="Directory to save figures + JSON (default: <run>/gof).")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    p.add_argument("--level", type=float, default=0.90, help="Credible band level for PIT plots.")
    p.add_argument("--bins", type=int, default=20, help="Number of bins for PIT histogram.")

    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    p.add_argument("--max-draws", type=int, default=None, help="Subsample posterior draws for speed (None = all).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed (subsampling + PPC simulation).")

    p.add_argument("--skip-pit", action="store_true", help="Skip PIT histogram and PP plot.")
    p.add_argument("--skip-ppc", action="store_true", help="Skip posterior predictive KS scatter.")

    p.add_argument("--json-name", type=str, default="gof_results.json", help="JSON filename to write in out dir.")
    p.add_argument("--json-full", action="store_true", help="Also store ks_obs/ks_rep arrays in JSON (large).")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    search_root = args.root or default_root(str(args.series))
    color = series_color(str(args.series))

    # Force Uccle monthly meta for reporting (harmless for GOF computations)
    meta_override: Dict[str, Any] = {
        "start_date": "1892-01-01",
        "freq": "Monthly",
        "period": 12,
        "series": str(args.series),
    }

    cfg = GOFConfig(
        level=float(args.level),
        bins=int(args.bins),
        seed=int(args.seed),
        max_draws=(None if args.max_draws is None else int(args.max_draws)),
        burn=int(args.burn),
        thin=int(args.thin),
        skip_pit=bool(args.skip_pit),
        skip_ppc=bool(args.skip_ppc),
        show=bool(args.show),
        json_name=str(args.json_name),
        json_full=bool(args.json_full),
        color=str(color),
    )

    runner = DLMGoodnessOfFit(level=cfg.level, bins=cfg.bins, color=cfg.color)
    runner.run_from_target(
        target=args.target,
        root=str(search_root),
        out_dir=args.out,
        cfg=cfg,
        meta_override=meta_override,
    )

    print("[done] Uccle DLM GOF written.")


if __name__ == "__main__":
    main()
