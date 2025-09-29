# %% scripts/run_uccle_dgev.py
import os, sys, time
import numpy as np
import pandas as pd
from datetime import datetime

# Make sure we can import from parent dir
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Sampler + plotter
from optimization.dgev_pgbs import DGEVParticleGibbs, Priors, SamplerConfig
from simulator.dgev_plotter import DGEVPlotter


# =========================
# Uccle data
# =========================
def load_uccle(
    start_year=1892,
    end_year=2022,
    data_dir="data",
    max_file="Uccle_Temp_Max_monthly_anom2.csv",
    min_file="Uccle_Temp_Min_monthly_anom2.csv",
    max_avg_file="Uccle_Temp_Max_Avg_monthly_anom2.csv",
    min_avg_file="Uccle_Temp_Min_Avg_monthly_anom2.csv",
):
    """
    Read Uccle monthly TX maxima and TN minima (CSV). Return arrays:
      maxima           : monthly TX (ascending in time)
      minima_neg       : monthly -TN (negated so colder is larger)
      maxima_avg/minima_avg : optional monthly means (not used by the sampler, but kept)
    """
    def read_one(path):
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        if "year" not in df.columns:
            df["year"] = df.index.year
        return df

    df_max     = read_one(os.path.join(data_dir, max_file))
    df_min     = read_one(os.path.join(data_dir, min_file))
    df_max_avg = read_one(os.path.join(data_dir, max_avg_file))
    df_min_avg = read_one(os.path.join(data_dir, min_avg_file))

    # Year masks (exclusive start, inclusive end)
    def mask(df): return (df.year > start_year) & (df.year <= end_year)

    def col_or_first(df, pref):
        if pref in df.columns: return pref
        for c in df.columns:
            if c.lower() != "year":
                return c
        raise ValueError(f"No data column found among {df.columns}")

    col_TMAX = col_or_first(df_max, "TMAX")
    col_TMIN = col_or_first(df_min, "TMIN")

    maxima     = df_max.loc[mask(df_max), col_TMAX].astype(float).to_numpy()
    minima_neg = -df_min.loc[mask(df_min), col_TMIN].astype(float).to_numpy()

    maxima     = maxima[np.isfinite(maxima)]
    minima_neg = minima_neg[np.isfinite(minima_neg)]

    maxima_avg = df_max_avg.loc[mask(df_max_avg), col_or_first(df_max_avg, "TMAX")].astype(float).to_numpy()
    minima_avg = df_min_avg.loc[mask(df_min_avg), col_or_first(df_min_avg, "TMIN")].astype(float).to_numpy()

    return dict(
        maxima=maxima,
        minima_neg=minima_neg,
        maxima_avg=maxima_avg,
        minima_avg=minima_avg,
    )


# =========================
# Small helpers
# =========================
def estimate_season(y: np.ndarray, period: int = 12) -> np.ndarray:
    """
    Estimate a length-`period` seasonal vector by averaging y within each month-of-year,
    then force exact sum-to-zero. (Return full length p.)
    """
    y = np.asarray(y, float)
    means = np.zeros(period, float)
    counts = np.zeros(period, int)
    for t, val in enumerate(y):
        k = t % period
        means[k] += val
        counts[k] += 1
    counts[counts == 0] = 1
    means = means / counts
    means = means - np.mean(means)
    means[-1] = -np.sum(means[:-1])
    return means


# =========================
# Runner
# =========================
def run_uccle(
    y: np.ndarray,
    series_key: str,      # "TX" or "TN" -> used for directory split
    label: str,
    period: int,
    level_mode: str = "deterministic",
    trend_mode: str = "deterministic",
    season_mode: str = "deterministic",
    out_root: str = "uccle",
    seed: int = 7,
    n_iter: int = 4000,
    burn: int = 1000,
    thin: int = 1,
    n_particles: int = 250,
    trans_eps: float = 1e-8,
    step_logsigma: float = 0.08,
    step_xi: float = 0.08,
    step_level: float = 0.05,
    step_slope: float = 0.05,
    step_season: float = 0.05,
    # Priors (all optional; None => sensible default)
    prior_m_sigma: float | None = None,
    prior_s_sigma: float | None = 1.0,
    prior_m_xi:    float | None = -0.1,
    prior_s_xi:    float | None = 0.25,
    prior_a_q:     float | None = 2.0,
    prior_b_q:     float | None = 0.05,
    prior_m_level: float | None = 0.0,
    prior_s_level: float | None = 10.0,
    prior_m_slope: float | None = 0.0,
    prior_s_slope: float | None = 10.0,
    prior_m_season: str | None  = "auto",  # 'auto' | 'zero' | 'none' | CSV of p-1 floats
    prior_s_season: float | None = None,   # if None, fallback to s_season_prior
    # legacy knob kept for compatibility (overridden by prior_s_season if provided)
    s_season_prior: float = 5.0,
    show_plots: bool = False,
    progress: bool = True,
    # Plotting controls
    plot_level: float = 0.90,        # credible band level
    plot_center: str = "median",     # {'median','mean','map'} for separate component figs
    plot_map_bins: int = 60,         # bins for 'map' center estimate
    skip_states: bool = False,       # skip the stacked states/observations panel
    skip_separate: bool = False,     # skip per-component separate figures
):
    """
    Fit DGEV with selectable structural modes to Uccle series `y`, save posterior
    under uccle/<TX or TN>/<mode>/<timestamp>, and export figures with DGEVPlotter.
    """
    y = np.asarray(y, float)
    T = y.size
    period = int(period)

    # Basic scale for prior on log-sigma (used if prior_m_sigma is None)
    series_std = float(np.nanstd(y)) or 1.0

    # Determine prior m_season (length p-1) according to user request
    m_season_vec = None
    if isinstance(prior_m_season, str):
        key = prior_m_season.strip().lower()
        if key == "auto":
            if season_mode == "deterministic":
                full = estimate_season(y, period=period)
                m_season_vec = full[:-1].tolist()
            else:
                m_season_vec = None
        elif key == "zero":
            m_season_vec = [0.0] * (period - 1)
        elif key == "none" or key == "":
            m_season_vec = None
        else:
            # parse CSV list
            nums = [float(tok) for tok in prior_m_season.split(",")]
            if len(nums) != (period - 1):
                raise ValueError(f"--prior-m-season expects {period-1} comma-separated values, got {len(nums)}.")
            m_season_vec = nums
    elif prior_m_season is None:
        # default: behave like 'auto' when deterministic season
        if season_mode == "deterministic":
            full = estimate_season(y, period=period)
            m_season_vec = full[:-1].tolist()
    else:
        raise ValueError("--prior-m-season must be a string: 'auto'|'zero'|'none' or CSV list.")

    # Choose s_season (new flag takes precedence over legacy)
    s_season = float(prior_s_season) if (prior_s_season is not None) else float(s_season_prior)

    # Priors (note: m_sigma is on log-scale)
    priors = Priors(
        m_sigma = float(prior_m_sigma) if (prior_m_sigma is not None) else np.log(series_std),
        s_sigma = float(prior_s_sigma),
        m_xi    = float(prior_m_xi),
        s_xi    = float(prior_s_xi),
        a_q     = float(prior_a_q),
        b_q     = float(prior_b_q),
        m_level = float(prior_m_level),
        s_level = float(prior_s_level),
        m_slope = float(prior_m_slope),
        s_slope = float(prior_s_slope),
        # IMPORTANT: pass a list, not an ndarray
        m_season = m_season_vec,
        s_season = float(s_season),
    )

    # Sampler config
    cfg = SamplerConfig(
        n_iter=int(n_iter), burn=int(burn), thin=int(thin),
        step_logsigma=float(step_logsigma), step_xi=float(step_xi),
        step_level=float(step_level), step_slope=float(step_slope),
        step_season=float(step_season),
        n_particles=int(n_particles), trans_eps=float(trans_eps),
        random_seed=int(seed), progress=bool(progress),
    )

    # Initial values for latent/deterministic pieces
    m0_level = float(np.nanmean(y)) if level_mode == "dynamic" else 0.0
    v0_level = 0.5
    m0_trend = 0.0
    v0_trend = 0.05

    if season_mode == "dynamic":
        m0_season = np.zeros(period - 1, float)
        v0_season = np.full(period - 1, 0.5, float)
    else:
        m0_season = None
        v0_season = None

    level_value_init = float(np.nanmean(y))
    slope_value_init = 0.0

    # Build sampler
    sampler = DGEVParticleGibbs(
        y=y, period=period,
        level_mode=level_mode, trend_mode=trend_mode, seasonal_mode=season_mode,
        m0_level=m0_level, v0_level=v0_level,
        m0_trend=m0_trend, v0_trend=v0_trend,
        m0_season=m0_season, v0_season=v0_season,
        priors=priors, cfg=cfg,
        level_value_init=level_value_init,
        slope_value_init=slope_value_init,
        # seed deterministic seasonal vector if using deterministic season
        seasonal_vector_init=(np.array(m_season_vec, float) if (m_season_vec is not None and season_mode == "deterministic") else None),
    )

    # ---------- Output dirs: uccle/<TX or TN>/<mode>/<timestamp> ----------
    mode_tag = f"{level_mode}-{trend_mode}-{season_mode}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(out_root, series_key, mode_tag, timestamp)
    fig_root = os.path.join(save_root, "figures")
    os.makedirs(fig_root, exist_ok=True)

    print(f"\n=== {label} | mode: {mode_tag} | T={T}, p={period} ===")
    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"[done] elapsed {elapsed:.2f}s")

    # Save posterior bundle
    sampler.save_posterior(
        out_npz_path=os.path.join(save_root, "posterior.npz"),
        extra_meta={
            "label": label,
            "series_key": series_key,
            "mode_tag": mode_tag,
            "elapsed_seconds": float(elapsed)
        },
    )

    # ---------- Plotting with DGEVPlotter ----------
    # Set flags the plotter expects (so dynamic/deterministic branches render correctly)
    sampler.include_level = (sampler.idx_alpha is not None)                        # dynamic α_t present?
    sampler.include_trend = (sampler.idx_beta is not None) or (trend_mode == "deterministic")
    sampler.include_seasonality = (season_mode != "none")

    plotter = DGEVPlotter(sampler, level=float(plot_level))

    # Always save diagnostics; show only if requested
    plotter.plot_diagnostics(save_dir=fig_root, fname_prefix="diagnostics", show=bool(show_plots))
    plotter.plot_mcmc_diagnostics_extra(40, save_dir=fig_root, fname_prefix="mcmc_extra", show=bool(show_plots))

    if not bool(skip_states) and ("mu" in sampler.keep):
        plotter.plot_states_and_observations(save_dir=fig_root, fname_prefix="states", show=bool(show_plots))

    if not bool(skip_separate):
        plotter.plot_components_separately(
            save_dir=fig_root,
            fname_prefix="components",
            center=str(plot_center),
            map_bins=int(plot_map_bins),
            show=bool(show_plots),
        )

    plotter.plot_quick_hist_panel(posterior, save_dir=fig_root, fname_prefix="quick_hist", show=bool(show_plots))

    # Console summary
    print(f"[{label} | {mode_tag}] mean σ: {np.mean(posterior['sigma']):.3f}")
    print(f"[{label} | {mode_tag}] mean ξ: {np.mean(posterior['xi']):.3f}")
    if "Q" in posterior:
        if sampler.idx_alpha is not None:
            print(f"[{label} | {mode_tag}] mean Q_alpha: {np.mean(posterior['Q'][:, sampler.idx_alpha]):.4f}")
        if sampler.idx_beta is not None:
            print(f"[{label} | {mode_tag}] mean Q_beta:  {np.mean(posterior['Q'][:, sampler.idx_beta]):.4f}")
        if season_mode == "dynamic" and sampler.idx_gamma_end is not None:
            print(f"[{label} | {mode_tag}] mean Q_gamma: {np.mean(posterior['Q'][:, sampler.idx_gamma_end]):.4f}")

    return sampler, posterior


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Uccle DGEV runs on monthly TX (maxima) or TN (minima, negated) with selectable structural modes."
    )

    # Data options
    parser.add_argument("--series", choices=["TX", "TN"], default="TX",
                        help="TX: monthly maxima; TN: monthly minima (negated for maxima-like tail).")
    parser.add_argument("--start-year", type=int, default=1892, help="Exclusive lower year bound.")
    parser.add_argument("--end-year",   type=int, default=2022, help="Inclusive upper year bound.")
    parser.add_argument("--data-dir",   type=str, default="data")
    parser.add_argument("--max-file",   type=str, default="Uccle_Temp_Max_monthly_anom2.csv")
    parser.add_argument("--min-file",   type=str, default="Uccle_Temp_Min_monthly_anom2.csv")
    parser.add_argument("--max-avg-file", type=str, default="Uccle_Temp_Max_Avg_monthly_anom2.csv")
    parser.add_argument("--min-avg-file", type=str, default="Uccle_Temp_Min_Avg_monthly_anom2.csv")

    # Structural modes — default: deterministic-deterministic-deterministic
    parser.add_argument("--level-mode",  choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode",  choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")

    # Sampler config
    parser.add_argument("--period", type=int, default=12, help="Seasonal period (12 for months).")
    parser.add_argument("--n-iter", type=int, default=4000)
    parser.add_argument("--burn",   type=int, default=1000)
    parser.add_argument("--thin",   type=int, default=1)
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--trans-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=13)

    # MH steps
    parser.add_argument("--step-logsigma", type=float, default=0.04)
    parser.add_argument("--step-xi",       type=float, default=0.02)
    parser.add_argument("--step-level",    type=float, default=0.15)
    parser.add_argument("--step-slope",    type=float, default=0.0001)
    parser.add_argument("--step-season",   type=float, default=0.1)
    
    # Stepsizes tuned for good acceptance in deterministic-deterministic-deterministic mode:
    #   --step-logsigma 0.04 --step-xi 0.02 --step-level 0.15 --step-slope 0.0001 --step-season 0.1
    # Stepsizes tuned for good acceptance in dynamic-deterministic-deterministic mode:
    #   --step-logsigma 0.04 --step-xi 0.02 --step-level 0.15 --step-slope 0.0001 --step-season 0.1

    # PRIORS (new, all exposed)
    parser.add_argument("--prior-m-sigma", type=float, default=None,
                        help="Mean of log(σ) prior; default uses log(sample SD) of the series.")
    parser.add_argument("--prior-s-sigma", type=float, default=3.0,
                        help="Std of log(σ) prior.")
    parser.add_argument("--prior-m-xi",    type=float, default=-0.1,
                        help="Mean of ξ prior.")
    parser.add_argument("--prior-s-xi",    type=float, default=0.25,
                        help="Std of ξ prior.")
    parser.add_argument("--prior-a-q",     type=float, default=2.0,
                        help="Shape 'a' of Inverse-Gamma/IG-like prior for state innovation variances.")
    parser.add_argument("--prior-b-q",     type=float, default=0.05,
                        help="Scale 'b' of IG-like prior for state innovation variances.")
    parser.add_argument("--prior-m-level", type=float, default=0.0,
                        help="Mean of deterministic level prior (if used).")
    parser.add_argument("--prior-s-level", type=float, default=10.0,
                        help="Std of deterministic level prior (if used).")
    parser.add_argument("--prior-m-slope", type=float, default=0.0,
                        help="Mean of deterministic slope prior (if used).")
    parser.add_argument("--prior-s-slope", type=float, default=10.0,
                        help="Std of deterministic slope prior (if used).")
    parser.add_argument("--prior-m-season", type=str, default="auto",
                        help="Prior mean for seasonal (first p-1). Options: 'auto' (default), 'zero', 'none', "
                             "or CSV of p-1 floats.")
    parser.add_argument("--prior-s-season", type=float, default=5,
                        help="Std dev for deterministic seasonal prior (first p-1). If omitted, falls back to --s-season-prior.")

    # Output & plotting
    parser.add_argument("--out-root", type=str, default="uccle",
                        help="Root folder for outputs (default saves to uccle/<TX|TN>/<mode>/<timestamp>).")
    parser.add_argument("--show-plots", action="store_true", help="Also display figures interactively.")
    parser.add_argument("--no-plots",  action="store_true", help="Suppress tqdm chatter only (plots still saved).")

    # Plotter controls
    parser.add_argument("--plot-level", type=float, default=0.90, help="Credible interval level for plot bands.")
    parser.add_argument("--plot-center", type=str, choices=["median","mean","map"], default="median",
                        help="Center line for per-component figures.")
    parser.add_argument("--map-bins", type=int, default=60, help="Bins for MAP center in per-time histograms.")
    parser.add_argument("--skip-states", action="store_true", help="Skip stacked states/observations panel.")
    parser.add_argument("--skip-separate", action="store_true", help="Skip separate component figures.")

    args = parser.parse_args()
    np.random.seed(int(args.seed))

    series = load_uccle(
        start_year=args.start_year,
        end_year=args.end_year,
        data_dir=args.data_dir,
        max_file=args.max_file,
        min_file=args.min_file,
        max_avg_file=args.max_avg_file,
        min_avg_file=args.min_avg_file,
    )

    if args.series == "TX":
        y = series["maxima"]
        series_key = "TX"
        label = "Uccle TX maxima (monthly)"
    else:
        y = series["minima_neg"]
        series_key = "TN"
        label = "Uccle TN minima (monthly)"

    _sampler, _posterior = run_uccle(
        y=y,
        series_key=series_key,
        label=label,
        period=int(args.period),
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        season_mode=args.season_mode,
        out_root=args.out_root,
        seed=int(args.seed),
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        n_particles=int(args.particles),
        trans_eps=float(args.trans_eps),
        step_logsigma=float(args.step_logsigma),
        step_xi=float(args.step_xi),
        step_level=float(args.step_level),
        step_slope=float(args.step_slope),
        step_season=float(args.step_season),
        # Priors (all parsed)
        prior_m_sigma=args.prior_m_sigma,
        prior_s_sigma=float(args.prior_s_sigma),
        prior_m_xi=float(args.prior_m_xi),
        prior_s_xi=float(args.prior_s_xi),
        prior_a_q=float(args.prior_a_q),
        prior_b_q=float(args.prior_b_q),
        prior_m_level=float(args.prior_m_level),
        prior_s_level=float(args.prior_s_level),
        prior_m_slope=float(args.prior_m_slope),
        prior_s_slope=float(args.prior_s_slope),
        prior_m_season=str(args.prior_m_season),
        prior_s_season=(float(args.prior_s_season) if args.prior_s_season is not None else None),
        # Plotting / control
        show_plots=bool(args.show_plots),
        progress=not bool(args.no_plots),
        plot_level=float(args.plot_level),
        plot_center=str(args.plot_center),
        plot_map_bins=int(args.map_bins),
        skip_states=bool(args.skip_states),
        skip_separate=bool(args.skip_separate),
    )
    print("\n[all done]")
