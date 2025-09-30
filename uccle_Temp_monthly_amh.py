# %% scripts/run_uccle_dgev.py
"""
Uccle DGEV runner (clean rewrite)
---------------------------------
* Loads Uccle temperature series (TX maxima or TN minima-negated) for a given period
* Runs DGEV Particle Gibbs with optional adaptive MH
* Saves posterior bundle and generates **all** diagnostics via DGEVPlotter

Output directory structure:
  <out_root>/<series>/<level-trend-season>/<YYYYmmdd_HHMMSS>/
    - posterior.npz
    - figures/
        diagnostics.png
        mcmc_extra.png
        q_diagnostics.png               (if any dynamic Q present)
        deterministic_diagnostics.png    (if any deterministic params present)
        states.png                       (unless --skip-states)
        components_*.png                 (unless --skip-separate)
        quick_hist.png
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

# Ensure parent import path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Sampler + plotter
from optimization.dgev_pgbs_amh import DGEVParticleGibbs, Priors, SamplerConfig
from simulator.dgev_plotter import DGEVPlotter


# =========================
# Utilities
# =========================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================
# Uccle data
# =========================

def load_uccle(
    start_year: int = 1892,
    end_year: int = 2022,
    data_dir: str = "data",
    max_file: str = "Uccle_Temp_Max_monthly_anom2.csv",
    min_file: str = "Uccle_Temp_Min_monthly_anom2.csv",
    max_avg_file: str = "Uccle_Temp_Max_Avg_monthly_anom2.csv",
    min_avg_file: str = "Uccle_Temp_Min_Avg_monthly_anom2.csv",
) -> Dict[str, np.ndarray]:
    """Load Uccle monthly series and return arrays.

    Returns a dict with keys: maxima, minima_neg, maxima_avg, minima_avg.
    """
    def read_one(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        if "year" not in df.columns:
            df["year"] = df.index.year
        return df

    def mask(df: pd.DataFrame) -> np.ndarray:
        return (df.year > start_year) & (df.year <= end_year)

    def col_or_first(df: pd.DataFrame, pref: str) -> str:
        if pref in df.columns:
            return pref
        for c in df.columns:
            if c.lower() != "year":
                return c
        raise ValueError(f"No data column found among {df.columns}")

    # Read
    df_max = read_one(os.path.join(data_dir, max_file))
    df_min = read_one(os.path.join(data_dir, min_file))
    df_max_avg = read_one(os.path.join(data_dir, max_avg_file))
    df_min_avg = read_one(os.path.join(data_dir, min_avg_file))

    # Columns
    col_TMAX = col_or_first(df_max, "TMAX")
    col_TMIN = col_or_first(df_min, "TMIN")

    # Series
    maxima = df_max.loc[mask(df_max), col_TMAX].astype(float).to_numpy()
    minima_neg = -df_min.loc[mask(df_min), col_TMIN].astype(float).to_numpy()
    maxima = maxima[np.isfinite(maxima)]
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
# Runner
# =========================

def run_uccle(
    y: np.ndarray,
    series_key: str,
    label: str,
    period: int,
    level_mode: str,
    trend_mode: str,
    season_mode: str,
    priors: Priors,
    cfg: SamplerConfig,
    out_root: str,
    show_plots: bool = False,
    skip_states: bool = False,
    skip_separate: bool = False,
    plot_level: float = 0.90,
    plot_center: str = "median",
    plot_map_bins: int = 60,
    max_lag: int = 40,
    season_k: int = 6,
) -> tuple[DGEVParticleGibbs, Dict[str, Any]]:
    """Run a single DGEV fit and generate all plots.

    Returns (sampler, posterior_dict).
    """
    y = np.asarray(y, float)
    T = int(y.size)

    # Initial values (consistent with model modes)
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

    # Sampler
    sampler = DGEVParticleGibbs(
        y=y, period=period,
        level_mode=level_mode, trend_mode=trend_mode, seasonal_mode=season_mode,
        m0_level=m0_level, v0_level=v0_level,
        m0_trend=m0_trend, v0_trend=v0_trend,
        m0_season=m0_season, v0_season=v0_season,
        priors=priors, cfg=cfg,
        level_value_init=level_value_init,
        slope_value_init=slope_value_init,
    )

    # Output dirs
    mode_tag = f"{level_mode}-{trend_mode}-{season_mode}"
    save_root = os.path.join(out_root, series_key, mode_tag, _now_tag())
    fig_root = os.path.join(save_root, "figures")
    _ensure_dir(fig_root)

    # Run
    print(f"\n=== {label} | mode: {mode_tag} | T={T}, p={period} ===")
    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"[done] elapsed {elapsed:.2f}s")

    # Save posterior
    sampler.save_posterior(
        out_npz_path=os.path.join(save_root, "posterior.npz"),
        extra_meta={
            "label": label,
            "series_key": series_key,
            "mode_tag": mode_tag,
            "elapsed_seconds": float(elapsed),
        },
    )

    # Console summaries
    print(f"Posterior mean sigma: {np.mean(posterior['sigma']):.3f}")
    print(f"Posterior mean xi:    {np.mean(posterior['xi']):.3f}")

    if "Q" in posterior and getattr(sampler, "dim", 0) > 0 and posterior["Q"].size > 0:
        if sampler.idx_alpha is not None:
            print(f"Posterior mean Q_alpha: {np.mean(posterior['Q'][:, sampler.idx_alpha]):.6g}")
        if sampler.idx_beta is not None:
            print(f"Posterior mean Q_beta:  {np.mean(posterior['Q'][:, sampler.idx_beta]):.6g}")
        if season_mode == "dynamic" and sampler.idx_gamma_end is not None:
            print(f"Posterior mean Q_gamma: {np.mean(posterior['Q'][:, sampler.idx_gamma_end]):.6g}")

    if "log_evidence" in posterior and posterior["log_evidence"].size > 0:
        le = posterior["log_evidence"]
        print(
            f"Mean log p(y|theta): {np.nanmean(le):.3f} | "
            f"Median: {np.nanmedian(le):.3f} | Best: {np.nanmax(le):.3f}"
        )

    # Flags so plotter knows what to draw
    sampler.include_level = sampler.idx_alpha is not None
    sampler.include_trend = (sampler.idx_beta is not None) or (trend_mode == "deterministic")
    sampler.include_seasonality = (season_mode != "none")

    # Plots
    plotter = DGEVPlotter(sampler, level=float(plot_level))
    plotter.plot_diagnostics(save_dir=fig_root, fname_prefix="diagnostics", show=show_plots)
    plotter.plot_mcmc_diagnostics_extra(max_lag, save_dir=fig_root, fname_prefix="mcmc_extra", show=show_plots)
    plotter.plot_process_noise_diagnostics(max_lag=max_lag, save_dir=fig_root, fname_prefix="q_diagnostics", show=show_plots)
    plotter.plot_deterministic_param_diagnostics(
        max_lag=max_lag, season_k=season_k, save_dir=fig_root,
        fname_prefix="deterministic_diagnostics", show=show_plots,
    )

    if (not skip_states) and ("mu" in sampler.keep):
        plotter.plot_states_and_observations(save_dir=fig_root, fname_prefix="states", show=show_plots)

    if not skip_separate:
        plotter.plot_components_separately(
            save_dir=fig_root, fname_prefix="components",
            center=str(plot_center), map_bins=int(plot_map_bins), show=show_plots,
        )

    plotter.plot_quick_hist_panel(posterior, save_dir=fig_root, fname_prefix="quick_hist", show=show_plots)

    return sampler, posterior


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Uccle DGEV runs with adaptive MH options.")

    # Data
    p.add_argument("--series", choices=["TX", "TN"], default="TX")
    p.add_argument("--start-year", type=int, default=1892)
    p.add_argument("--end-year", type=int, default=2022)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--period", type=int, default=12)

    # Modes
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="deterministic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")

    # Sampler config
    p.add_argument("--n-iter", type=int, default=7000)
    p.add_argument("--burn", type=int, default=2000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--particles", type=int, default=250)
    p.add_argument("--trans-eps", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=13)

    # MH step sizes
    p.add_argument("--step-logsigma", type=float, default=0.04)
    p.add_argument("--step-xi", type=float, default=0.02)
    p.add_argument("--step-level", type=float, default=0.14)
    p.add_argument("--step-slope", type=float, default=0.0002)
    p.add_argument("--step-season", type=float, default=0.11)

    # Adaptive options
    p.add_argument("--adapt-steps", action="store_true")
    p.add_argument("--adapt-every", type=int, default=25)
    p.add_argument("--adapt-until", choices=["burn", "all"], default="all")
    p.add_argument("--adapt-eta0", type=float, default=0.05)
    p.add_argument("--adapt-decay", type=float, default=0.75)
    p.add_argument("--adapt-target-1d", type=float, default=0.44)
    p.add_argument("--step-min", type=float, default=1e-5)
    p.add_argument("--step-max", type=float, default=1.0)

    # Priors
    p.add_argument("--prior-m-sigma", type=float, default=None)
    p.add_argument("--prior-s-sigma", type=float, default=3.0)
    p.add_argument("--prior-m-xi", type=float, default=-0.1)
    p.add_argument("--prior-s-xi", type=float, default=0.25)
    p.add_argument("--prior-a-q", type=float, default=2.0)
    p.add_argument("--prior-b-q", type=float, default=0.05)
    p.add_argument("--prior-m-level", type=float, default=0.0)
    p.add_argument("--prior-s-level", type=float, default=10.0)
    p.add_argument("--prior-m-slope", type=float, default=0.0)
    p.add_argument("--prior-s-slope", type=float, default=10.0)
    p.add_argument("--prior-m-season", type=str, default="auto")  # reserved for future use
    p.add_argument("--prior-s-season", type=float, default=5.0)

    # Plotting
    p.add_argument("--out-root", type=str, default="uccle")
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--skip-states", action="store_true")
    p.add_argument("--skip-separate", action="store_true")
    p.add_argument("--plot-level", type=float, default=0.90)
    p.add_argument("--plot-center", type=str, default="median", choices=["median", "mean", "map"])
    p.add_argument("--plot-map-bins", type=int, default=60)
    p.add_argument("--max-lag", type=int, default=40)
    p.add_argument("--season-k", type=int, default=6)

    args = p.parse_args()

    # Seed
    np.random.seed(int(args.seed))

    # Data
    series = load_uccle(start_year=args.start_year, end_year=args.end_year, data_dir=args.data_dir)
    y = series["maxima"] if args.series == "TX" else series["minima_neg"]
    label = "Uccle TX maxima" if args.series == "TX" else "Uccle TN minima"

    # Priors (data-informed log-sigma mean if not provided)
    series_std = float(np.nanstd(y)) or 1.0
    priors = Priors(
        m_sigma=float(args.prior_m_sigma) if args.prior_m_sigma is not None else np.log(series_std),
        s_sigma=float(args.prior_s_sigma),
        m_xi=float(args.prior_m_xi),
        s_xi=float(args.prior_s_xi),
        a_q=float(args.prior_a_q), b_q=float(args.prior_b_q),
        m_level=float(args.prior_m_level), s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope), s_slope=float(args.prior_s_slope),
        m_season=None, s_season=float(args.prior_s_season),
    )

    # Sampler config
    cfg = SamplerConfig(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        step_logsigma=args.step_logsigma, step_xi=args.step_xi,
        step_level=args.step_level, step_slope=args.step_slope, step_season=args.step_season,
        n_particles=args.particles, trans_eps=args.trans_eps, random_seed=args.seed,
        adapt_steps=args.adapt_steps, adapt_every=args.adapt_every, adapt_until=args.adapt_until,
        adapt_eta0=args.adapt_eta0, adapt_eta_decay=args.adapt_decay,
        adapt_target_1d=args.adapt_target_1d,
        step_min=args.step_min, step_max=args.step_max,
    )

    _sampler, _posterior = run_uccle(
        y=y, series_key=args.series, label=label,
        period=args.period, level_mode=args.level_mode,
        trend_mode=args.trend_mode, season_mode=args.season_mode,
        priors=priors, cfg=cfg, out_root=args.out_root,
        show_plots=args.show_plots, skip_states=args.skip_states,
        skip_separate=args.skip_separate,
        plot_level=args.plot_level, plot_center=args.plot_center,
        plot_map_bins=args.plot_map_bins, max_lag=args.max_lag, season_k=args.season_k,
    )

    print("\n[all done]")
