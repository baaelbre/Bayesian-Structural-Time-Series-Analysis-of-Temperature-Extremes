# %% scripts/run_uccle_dgev.py
import os, sys, time, json
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

# Make sure we can import from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.dgev_pgbs import DGEVParticleGibbs, Priors, SamplerConfig
from simulator.dgev_plotter import DGEVPlotter  # optional


# =========================
# I/O helpers
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_monthlies(
    start_year: int = 1892,
    end_year: int = 2022,
    data_dir: str = "data",
    txx_file: str = "TXx.csv",
    txn_file: str = "TXn.csv",
    tnx_file: str = "TNx.csv",
    tnn_file: str = "TNn.csv",
) -> dict:
    """
    Reads four monthly CSVs with a datetime-like index (first column) and a single value column:
      TXx (monthly max of TX), TXn (monthly min of TX),
      TNx (monthly max of TN), TNn (monthly min of TN).

    Each CSV is expected to have:
      - Index column: parseable dates (e.g., 'YYYY-MM-01'), representing month starts.
      - One numeric column named like the file stem (TXx, TXn, TNx, TNn).

    Returns aligned Series on monthly start ("MS"), clipped to [start_year, end_year].
    Output keys: 'TXx', 'TXn', 'TNx', 'TNn'
    """

    def read_one(path: str) -> pd.Series:
        df = pd.read_csv(path, index_col=0)
        # Parse index to datetime and coerce to month-start freq
        df.index = pd.to_datetime(df.index)
        # Pick the first numeric column (or the only one)
        if df.shape[1] > 1:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise ValueError(f"No numeric column found in {path}.")
            s = df[num_cols[0]]
        else:
            s = df.iloc[:, 0]
        s = s.sort_index()
        s = s.loc[(s.index.year >= start_year) & (s.index.year <= end_year)]
        return s.asfreq("MS")

    d = {
        "TXx": read_one(os.path.join(data_dir, txx_file)),
        "TXn": read_one(os.path.join(data_dir, txn_file)),
        "TNx": read_one(os.path.join(data_dir, tnx_file)),
        "TNn": read_one(os.path.join(data_dir, tnn_file)),
    }

    # Align to common monthly index
    common_idx = None
    for s in d.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    for k in d:
        d[k] = d[k].reindex(common_idx)

    return d


# =========================
# Build seasonal prior mean if needed
# =========================
def build_seasonal(period: int) -> np.ndarray:
    """
    Default smooth seasonal for first (p-1) entries (last implied by sum-to-zero).
    Matches helper in your sampler module.
    """
    g = np.cos(2 * np.pi * np.arange(period) / period)
    g -= np.mean(g)
    return g[: period - 1].astype(float)


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DGEV PG-BS on Uccle monthly TX/TN extremes")

    # Data & selection
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--txx-file", type=str, default="TXx.csv")
    parser.add_argument("--txn-file", type=str, default="TXn.csv")
    parser.add_argument("--tnx-file", type=str, default="TNx.csv")
    parser.add_argument("--tnn-file", type=str, default="TNn.csv")
    parser.add_argument("--start-year", type=int, default=1892)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument(
        "--series",
        choices=["TXx", "TXn", "TNx", "TNn"],
        default="TXx",
        help="Which monthly series to fit.",
    )

    # Structural model modes
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--period", type=int, default=12, help="Seasonal period (months).")

    # Initial values (shared/prior means)
    parser.add_argument("--level-init", type=float, default=0.0)
    parser.add_argument("--slope-init", type=float, default=0.0)

    # Priors
    parser.add_argument("--prior-m-sigma", type=float, default=1.0, help="Mean of prior on log(sigma)")
    parser.add_argument("--prior-s-sigma", type=float, default=1.0, help="SD of prior on log(sigma)")
    parser.add_argument("--prior-m-xi", type=float, default=0.0)
    parser.add_argument("--prior-s-xi", type=float, default=0.1)
    parser.add_argument("--prior-aq", type=float, default=1.9, help="IG shape for Q")
    parser.add_argument("--prior-bq", type=float, default=1e-3, help="IG scale for Q")
    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s-level", type=float, default=10.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s-slope", type=float, default=10.0)
    parser.add_argument(
        "--prior-m-season",
        type=str,
        default=None,
        help="Comma-separated first (p-1) means for deterministic seasonal prior (e.g. '0,0,0,...').",
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

    # Progress controls
    parser.add_argument("--progress", default=True, help="Print per-iteration progress info.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Compact progress line every k iterations (0 = auto ≈ 2% of n_iter).",
    )

    # Adaptive RW–MH (optional)
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

    # Load selected monthly series
    series = load_monthlies(
        start_year=args.start_year,
        end_year=args.end_year,
        data_dir=args.data_dir,
        txx_file=args.txx_file,
        txn_file=args.txn_file,
        tnx_file=args.tnx_file,
        tnn_file=args.tnn_file,
    )
    y = series[args.series].dropna().to_numpy(dtype=float)
    T = y.size

    if T < 5:
        raise ValueError(f"Not enough observations after filtering by years; got T={T}.")

    # Parse seasonal prior means (first p-1 entries) if deterministic season chosen
    def parse_csv_floats(s):
        if s is None:
            return None
        s = s.strip()
        if not s:
            return None
        return [float(tok) for tok in s.split(",")]

    m_season_prior = parse_csv_floats(args.prior_m_season)
    if args.season_mode == "deterministic":
        if m_season_prior is None:
            m_season_prior = build_seasonal(args.period).tolist()
        if len(m_season_prior) != args.period - 1:
            raise ValueError(f"--prior-m-season must have length {args.period - 1}.")

    priors = Priors(
        m_sigma=float(args.prior_m_sigma),
        s_sigma=float(args.prior_s_sigma),
        m_xi=float(args.prior_m_xi),
        s_xi=float(args.prior_s_xi),
        a_q=float(args.prior_aq),
        b_q=float(args.prior_bq),
        m_level=float(args.prior_m_level),
        s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope),
        s_slope=float(args.prior_s_slope),
        m_season=(m_season_prior if args.season_mode == "deterministic" else None),
        s_season=float(args.prior_s_season),
    )

    cfg = SamplerConfig(
        n_iter=args.n_iter,
        burn=args.burn,
        thin=args.thin,
        step_logsigma=args.step_logsigma if hasattr(args, "step_logsigma") else args.step_logsigma,
        step_xi=args.step_xi,
        step_level=args.step_level,
        step_slope=args.step_slope,
        step_season=args.step_season,
        n_particles=args.particles,
        trans_eps=args.trans_eps,
        random_seed=args.seed,
        progress=bool(args.progress),
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
    seasonal_init_pminus1 = None
    if args.season_mode == "deterministic":
        seasonal_init_pminus1 = np.asarray(m_season_prior, float)

    # Construct sampler
    sampler = DGEVParticleGibbs(
        y=y,
        period=args.period,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.season_mode,
        m0_level=args.level_init,
        v0_level=0.2,
        m0_trend=(args.slope_init if args.trend_mode != "none" else 0.0),
        v0_trend=0.05,
        m0_season=(np.zeros(args.period - 1) if args.season_mode == "dynamic" else None),
        v0_season=(np.full(args.period - 1, 0.5) if args.season_mode == "dynamic" else None),
        priors=priors,
        cfg=cfg,
        seasonal_vector_init=seasonal_init_pminus1,
    )

    # Output directories
    tag = f"Uccle-{args.series}_{args.level_mode}-{args.trend_mode}-{args.season_mode}"
    out_dir = args.out_dir or os.path.join(
        "results", "uccle", f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # Run
    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    # Save posterior + metadata
    npz_path = os.path.join(out_dir, "posterior.npz")
    arrays = dict(posterior)
    arrays["y"] = y
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "series": args.series,
        "T": int(T),
        "period": int(args.period),
        "modes": {
            "level_mode": args.level_mode,
            "trend_mode": args.trend_mode,
            "season_mode": args.season_mode,
        },
        "cfg": asdict(cfg),
        "priors": asdict(priors),
        "elapsed_seconds": float(elapsed),
        "timestamp": datetime.now().isoformat(),
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
            plotter.quick_overview(
                sampler=sampler,
                posterior=posterior,
                out_dir=fig_dir,
                show=args.show_plots,
                title=f"{tag} ({args.start_year}-{args.end_year})",
            )
        except Exception as e:
            print(f"[warn] Plotting failed: {e}")
