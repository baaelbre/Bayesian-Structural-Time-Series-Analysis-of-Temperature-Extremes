# %% scripts/run_uccle_dgev_seasonal.py
import os, sys, time, json
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

# Import from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.dgev_pgbs import DGEVParticleGibbs, Priors, SamplerConfig
from simulator.dgev_plotter import DGEVPlotter  # optional


# =========================
# I/O helpers
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_seasonals(
    start_year: int = 1892,
    end_year: int = 2022,
    data_dir: str = "data",
    txx_file: str = "TXx_seasonal.csv",
    txn_file: str = "TXn_seasonal.csv",
    tnx_file: str = "TNx_seasonal.csv",
    tnn_file: str = "TNn_seasonal.csv",
) -> dict:
    """
    Reads four SEASONAL CSVs with a date-like index (first column) and one value column:
      TXx (seasonal max of daily TX), TXn (seasonal min of daily TX),
      TNx (seasonal max of daily TN), TNn (seasonal min of daily TN).

    Index can be season start or any date within the season.
    We convert to PeriodIndex('Q-FEB'): Q1=DJF (Dec–Jan–Feb, labeled by Jan/Feb year),
    Q2=MAM, Q3=JJA, Q4=SON. Series are aligned on the common seasonal index.
    """

    def read_one(path: str) -> pd.Series:
        df = pd.read_csv(path, index_col=0)
        # pick numeric column
        col = df.columns[0] if df.shape[1] == 1 else next(
            (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None
        )
        if col is None:
            raise ValueError(f"No numeric column found in {path}.")
        ts = pd.to_datetime(df.index)
        s = pd.Series(df[col].to_numpy(dtype=float), index=ts.to_period("Q-FEB")).sort_index()
        s = s[(s.index.year >= start_year) & (s.index.year <= end_year)]
        return s

    d = {
        "TXx": read_one(os.path.join(data_dir, txx_file)),
        "TXn": read_one(os.path.join(data_dir, txn_file)),
        "TNx": read_one(os.path.join(data_dir, tnx_file)),
        "TNn": read_one(os.path.join(data_dir, tnn_file)),
    }

    # Align to common seasonal PeriodIndex
    common_idx = None
    for s in d.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    for k in d:
        d[k] = d[k].reindex(common_idx).sort_index()

    return d


def build_seasonal(period: int) -> np.ndarray:
    """Default smooth prior mean for first (p-1) seasonal entries (sum-to-zero)."""
    g = np.cos(2 * np.pi * np.arange(period) / period)
    g -= np.mean(g)
    return g[: period - 1].astype(float)


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DGEV PG-BS on Uccle SEASONAL TX/TN extremes (meteorological seasons)"
    )

    # Data & selection
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--txx-file", type=str, default="TXx_seasonal.csv")
    parser.add_argument("--txn-file", type=str, default="TXn_seasonal.csv")
    parser.add_argument("--tnx-file", type=str, default="TNx_seasonal.csv")
    parser.add_argument("--tnn-file", type=str, default="TNn_seasonal.csv")
    parser.add_argument("--start-year", type=int, default=1892, help="Season label year (e.g., DJF 1892).")
    parser.add_argument("--end-year", type=int, default=2022, help="Season label year (e.g., DJF 2022).")
    parser.add_argument(
        "--series",
        choices=["TXx", "TXn", "TNx", "TNn"],
        default="TXx",
        help="Which SEASONAL series to fit (DJF/MAM/JJA/SON timeline).",
    )

    # Structural model modes
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--period", type=int, default=4, help="Seasonal period (meteorological seasons = 4).")

    # Initial values
    parser.add_argument("--level-init", type=float, default=0.0)
    parser.add_argument("--slope-init", type=float, default=0.0)

    # Priors (match optimization/dgev_pgbs.Priors exactly)
    parser.add_argument("--prior-m-sigma", type=float, default=1.0, help="Mean of prior on log(sigma)")
    parser.add_argument("--prior-s-sigma", type=float, default=1.0, help="SD of prior on log(sigma)")
    parser.add_argument("--prior-m-xi", type=float, default=0.0)
    parser.add_argument("--prior-s-xi", type=float, default=0.1)

    # Per-component IG(a,b) for process noises
    parser.add_argument("--prior-aq-alpha", type=float, default=1.1)
    parser.add_argument("--prior-bq-alpha", type=float, default=1e-4)
    parser.add_argument("--prior-aq-beta",  type=float, default=1.1)
    parser.add_argument("--prior-bq-beta",  type=float, default=1e-12)
    parser.add_argument("--prior-aq-gamma", type=float, default=1.5)
    parser.add_argument("--prior-bq-gamma", type=float, default=5e-6)

    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s-level", type=float, default=10.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s-slope", type=float, default=10.0)
    parser.add_argument(
        "--prior-m-season",
        type=str,
        default=None,
        help="Comma-separated first (p-1) means for deterministic seasonal prior (length = period-1).",
    )
    parser.add_argument("--prior-s-season", type=float, default=5.0)

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=4000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=2)

    parser.add_argument("--step-logsigma", type=float, default=0.1)
    parser.add_argument("--step-xi", type=float, default=0.1)
    parser.add_argument("--step-level", type=float, default=0.05)
    parser.add_argument("--step-slope", type=float, default=0.001)
    parser.add_argument("--step-season", type=float, default=0.05)

    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--trans-eps", type=float, default=1e-8)

    # Progress & adaptation
    parser.add_argument("--progress", default=True, help="Print per-iteration progress info.")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Compact progress line every k iterations (0 = auto ≈ 2% of n_iter).")
    parser.add_argument("--adapt-steps", action="store_true")
    parser.add_argument("--adapt-every", type=int, default=25)
    parser.add_argument("--adapt-until", choices=["burn", "all"], default="burn")
    parser.add_argument("--adapt-eta0", type=float, default=0.05)
    parser.add_argument("--adapt-decay", type=float, default=0.75)
    parser.add_argument("--adapt-target-1d", type=float, default=0.44)
    parser.add_argument("--step-min", type=float, default=1e-5)
    parser.add_argument("--step-max", type=float, default=1.0)

    # Output & reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Load SEASONAL series
    series = load_seasonals(
        start_year=args.start_year, end_year=args.end_year, data_dir=args.data_dir,
        txx_file=args.txx_file, txn_file=args.txn_file, tnx_file=args.tnx_file, tnn_file=args.tnn_file,
    )
    s = series[args.series].dropna().sort_index()  # PeriodIndex('Q-FEB')
    y = s.to_numpy(dtype=float)
    T = y.size
    if T < 5:
        raise ValueError(f"Not enough observations after filtering by seasons/years; got T={T}.")

    # Parse deterministic seasonal prior (first p-1 entries)
    def _parse_csv_floats(val: str | None):
        if val is None: return None
        val = val.strip()
        return None if not val else [float(tok) for tok in val.split(",")]

    m_season_prior = _parse_csv_floats(args.prior_m_season)
    if args.season_mode == "deterministic":
        if m_season_prior is None:
            m_season_prior = build_seasonal(args.period).tolist()
        if len(m_season_prior) != args.period - 1:
            raise ValueError(f"--prior-m-season must have length {args.period - 1}.")

    # Priors — EXACTLY the fields your optimizer expects
    priors = Priors(
        m_sigma=float(args.prior_m_sigma), s_sigma=float(args.prior_s_sigma),
        m_xi=float(args.prior_m_xi),       s_xi=float(args.prior_s_xi),

        a_q_alpha=float(args.prior_aq_alpha), b_q_alpha=float(args.prior_bq_alpha),
        a_q_beta=float(args.prior_aq_beta),   b_q_beta=float(args.prior_bq_beta),
        a_q_gamma=float(args.prior_aq_gamma), b_q_gamma=float(args.prior_bq_gamma),

        m_level=float(args.prior_m_level), s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope), s_slope=float(args.prior_s_slope),

        m_season=(m_season_prior if args.season_mode == "deterministic" else None),
        s_season=float(args.prior_s_season),
    )

    cfg = SamplerConfig(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        step_logsigma=args.step_logsigma, step_xi=args.step_xi,
        step_level=args.step_level, step_slope=args.step_slope, step_season=args.step_season,
        n_particles=args.particles, trans_eps=args.trans_eps,
        random_seed=args.seed, progress=bool(args.progress),
        progress_every=int(args.progress_every),
        adapt_steps=bool(args.adapt_steps),
        adapt_every=int(args.adapt_every),
        adapt_until=str(args.adapt_until),
        adapt_eta0=float(args.adapt_eta0),
        adapt_eta_decay=float(args.adapt_decay),
        adapt_target_1d=float(args.adapt_target_1d),
        step_min=float(args.step_min),
        step_max=float(args.step_max),
    )

    # Seasonal init (deterministic case expects the p-1 vector; sampler fills the last to sum-zero)
    seasonal_init_pminus1 = np.asarray(m_season_prior, float) if args.season_mode == "deterministic" else None

    # Construct sampler
    sampler = DGEVParticleGibbs(
        y=y, period=args.period,  # 4 for seasons
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.season_mode,
        m0_level=args.level_init, v0_level=0.2,
        m0_trend=(args.slope_init if args.trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=(np.zeros(args.period - 1) if args.season_mode == "dynamic" else None),
        v0_season=(np.full(args.period - 1, 0.5) if args.season_mode == "dynamic" else None),
        priors=priors, cfg=cfg,
        seasonal_vector_init=seasonal_init_pminus1,
    )

    # Output
    tag = f"UccleSeasonal-{args.series}_{args.level_mode}-{args.trend_mode}-{args.season_mode}"
    out_dir = args.out_dir or os.path.join("results", "uccle", f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(out_dir); _ensure_dir(fig_dir)

    # Run
    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    # Save posterior + metadata
    npz_path = os.path.join(out_dir, "posterior.npz")
    arrays = dict(posterior); arrays["y"] = y
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "series": args.series, "T": int(T), "period": int(args.period),
        "index_type": "Q-FEB (DJF/MAM/JJA/SON), DJF labeled by Jan/Feb year",
        "modes": {"level_mode": args.level_mode, "trend_mode": args.trend_mode, "season_mode": args.season_mode},
        "cfg": asdict(cfg), "priors": asdict(priors),
        "elapsed_seconds": float(elapsed), "timestamp": datetime.now().isoformat(),
        "years": {"start": int(args.start_year), "end": int(args.end_year)},
    }
    with open(npz_path.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] Posterior -> {npz_path}")

    # Quick summaries
    if "sigma" in posterior and posterior["sigma"].size:
        print(f"Posterior mean sigma: {np.mean(posterior['sigma']):.3f}")
    if "xi" in posterior and posterior["xi"].size:
        print(f"Posterior mean xi:    {np.mean(posterior['xi']):.3f}")
    if "log_evidence" in posterior and posterior["log_evidence"].size:
        le = posterior["log_evidence"]
        print(f"Mean log p(y|theta): {np.nanmean(le):.3f} | Median: {np.nanmedian(le):.3f} | Best: {np.nanmax(le):.3f}")

    # Optional plotting
    if not args.no_plots:
        try:
            plotter = DGEVPlotter()
            time_index = s.index.to_timestamp(how="start")  # season starts for x-axis
            plotter.quick_overview(
                sampler=sampler, posterior=posterior, out_dir=fig_dir, show=args.show_plots,
                title=f"{tag} ({args.start_year}-{args.end_year})",
                time_index=time_index if hasattr(plotter, "quick_overview") else None,
            )
        except Exception as e:
            print(f"[warn] Plotting failed: {e}")
