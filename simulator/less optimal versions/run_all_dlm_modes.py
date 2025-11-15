# run_all_dlm_modes.py
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
import argparse
import pandas as pd

from mean_time_series_volatility import Mean_Time_Series  


def _parse_date(s: str | None):
    if not s:
        return None
    parts = [int(p) for p in s.split("-")]
    if len(parts) == 1:
        return datetime(parts[0], 1, 1)
    if len(parts) == 2:
        return datetime(parts[0], parts[1], 1)
    if len(parts) == 3:
        return datetime(parts[0], parts[1], parts[2])
    raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")


def _csv_floats(s: str | None) -> list[float] | None:
    if s is None or s.strip() == "":
        return None
    return [float(x) for x in s.split(",")]


def _maybe_parse(vec_str: str | None, default_val: float, length: int) -> list[float]:
    out = _csv_floats(vec_str)
    return out if out is not None else [default_val] * length


def _sanitize(*parts: str) -> str:
    return "__".join(p.replace(" ", "") for p in parts)


def run_one_combo(args, m0_season, v0_season, m0_season_sigma, v0_season_sigma,
                  lm, tm, sm, lS, tS, sS, start_date, out_dir, seed_bump=0,
                  save_csv: bool = False):
    np.random.seed(args.seed + seed_bump)

    mts = Mean_Time_Series(
        sigma=args.sigma,
        # mean block
        level_mode=lm,
        trend_mode=tm,
        seasonal_mode=sm,
        # sigma block
        level_mode_sigma=lS,
        trend_mode_sigma=tS,
        seasonal_mode_sigma=sS,
        # period
        period=args.period,
        # mean innovations
        q_level=args.q_level,
        q_trend=args.q_trend,
        q_season=args.q_season,
        # sigma innovations
        q_level_sigma=args.q_level_sigma,
        q_trend_sigma=args.q_trend_sigma,
        q_season_sigma=args.q_season_sigma,
        # mean priors
        m0_level=args.m0_level,
        v0_level=args.v0_level,
        m0_trend=(0.0 if tm == "none" else args.m0_trend),
        v0_trend=args.v0_trend,
        m0_season=m0_season,
        v0_season=v0_season,
        # sigma priors (ln sigma)
        m0_level_sigma=args.m0_level_sigma,
        v0_level_sigma=args.v0_level_sigma,
        m0_trend_sigma=(0.0 if tS == "none" else args.m0_trend_sigma),
        v0_trend_sigma=args.v0_trend_sigma,
        m0_season_sigma=m0_season_sigma,
        v0_season_sigma=v0_season_sigma,
        # time
        start_date=start_date,
    )

    y = []
    for _ in range(args.T):
        mts.move()
        y.append(mts.measure())

    truths = mts.get_truth_paths(as_numpy=True)
    y_arr     = np.asarray(y, float)
    dates_T   = truths["index"][:args.T]
    mu_arr    = truths["mu_t"][1:1 + args.T]
    alpha_mu  = truths["alpha_mu_t"][1:1 + args.T]
    beta_mu   = truths["beta_mu_t"][1:1 + args.T]
    gamma_mu  = truths["gamma_mu_t"][1:1 + args.T]
    eta_arr   = truths["eta_t"][1:1 + args.T]
    sigma_arr = truths["sigma_t"][1:1 + args.T]
    alpha_sig = truths["alpha_sig_t"][1:1 + args.T]
    beta_sig  = truths["beta_sig_t"][1:1 + args.T]
    gamma_sig = truths["gamma_sig_t"][1:1 + args.T]

    # Optional CSV per combo
    if save_csv:
        df = pd.DataFrame({
            "date": dates_T,
            "y_t": y_arr,
            "mu_t": mu_arr,
            "alpha_mu_t": alpha_mu,
            "beta_mu_t": beta_mu,
            "gamma_mu_t": gamma_mu,
            "eta_t": eta_arr,
            "sigma_t": sigma_arr,
            "alpha_sig_t": alpha_sig,
            "beta_sig_t": beta_sig,
            "gamma_sig_t": gamma_sig,
        })
        stem = _sanitize(f"mu-l{lm}-t{tm}-s{sm}", f"sig-l{lS}-t{tS}-s{sS}")
        csv_path = os.path.join(out_dir, f"{stem}.csv")
        df.to_csv(csv_path, index=False)

    # Filestem
    stem = _sanitize(f"mu-l{lm}-t{tm}-s{sm}", f"sig-l{lS}-t{tS}-s{sS}")

    # ---- Fig 1: y vs mu ----
    plt.figure(figsize=(10, 4))
    plt.plot(dates_T, y_arr, label=r"$y_t$", linewidth=1.0)
    plt.plot(dates_T, mu_arr, "--", label=r"$\mu_t$", linewidth=1.0)
    ttl = (f"[mu] level={lm}, trend={tm}, season={sm} | "
           f"[eta] level={lS}, trend={tS}, season={sS}")
    plt.title(ttl)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}__y_mu.png"), dpi=200)
    plt.close()

    # ---- Fig 2: truth paths (mu) ----
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax2[0].plot(dates_T, alpha_mu, linewidth=1.0); ax2[0].set_ylabel(r"$\alpha^{(\mu)}_t$"); ax2[0].grid(True)
    ax2[1].plot(dates_T, beta_mu,  linewidth=1.0); ax2[1].set_ylabel(r"$\beta^{(\mu)}_t$");  ax2[1].grid(True)
    ax2[2].plot(dates_T, gamma_mu, linewidth=1.0); ax2[2].set_ylabel(r"$\gamma^{(\mu)}_t$"); ax2[2].set_xlabel("time"); ax2[2].grid(True)
    fig2.suptitle("Truth paths — mean block")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"{stem}__mu_paths.png"), dpi=200)
    plt.close(fig2)

    # ---- Fig 3: truth paths (sigma) ----
    fig3, ax3 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax3[0].plot(dates_T, alpha_sig, linewidth=1.0); ax3[0].set_ylabel(r"$\alpha^{(\eta)}_t$"); ax3[0].grid(True)
    ax3[1].plot(dates_T, beta_sig,  linewidth=1.0); ax3[1].set_ylabel(r"$\beta^{(\eta)}_t$");  ax3[1].grid(True)
    ax3[2].plot(dates_T, gamma_sig, linewidth=1.0); ax3[2].set_ylabel(r"$\gamma^{(\eta)}_t$"); ax3[2].set_xlabel("time"); ax3[2].grid(True)
    fig3.suptitle("Truth paths — log-sigma block")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, f"{stem}__sig_paths.png"), dpi=200)
    plt.close(fig3)

    # ---- Fig 4: sigma and eta ----
    fig4, ax4 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig4.suptitle("Scale paths")
    ax4[0].plot(dates_T, sigma_arr, linewidth=1.0); ax4[0].set_ylabel(r"$\sigma_t$"); ax4[0].grid(True)
    ax4[1].plot(dates_T, eta_arr,   linewidth=1.0); ax4[1].set_ylabel(r"$\eta_t=\ln\sigma_t$"); ax4[1].set_xlabel("time"); ax4[1].grid(True)
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, f"{stem}__scale.png"), dpi=200)
    plt.close(fig4)


def main():
    p = argparse.ArgumentParser(
        description="Run all mu×sigma structural mode combinations using Mean_Time_Series and save plots."
    )

    # Core simulation controls
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--seed", type=int, default=42)

    # MEAN innovations
    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=0.002)
    p.add_argument("--q-season", type=float, default=0.15)

    # SIGMA innovations (ln sigma)
    p.add_argument("--q-level-sigma", type=float, default=0.05)
    p.add_argument("--q-trend-sigma", type=float, default=0.01)
    p.add_argument("--q-season-sigma", type=float, default=0.10)

    # MEAN priors / fixed
    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.015)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default="")
    p.add_argument("--v0-season", type=str, default="")

    # SIGMA priors / fixed (for ln sigma!)
    # Default here: log-sigma level prior mean/log-variance a bit loose; tweak as desired
    p.add_argument("--m0-level-sigma", type=float, default=1.0)
    p.add_argument("--v0-level-sigma", type=float, default=4.0)
    p.add_argument("--m0-trend-sigma", type=float, default=0.01)
    p.add_argument("--v0-trend-sigma", type=float, default=0.05)
    p.add_argument("--m0-season-sigma", type=str, default="")
    p.add_argument("--v0-season-sigma", type=str, default="")

    # Output
    p.add_argument("--out-dir", type=str, default=os.path.join("results", "DLM"))
    p.add_argument("--save-csv", action="store_true", help="Also save a CSV per combination.")

    args = p.parse_args()

    # Build seasonal vectors (period-1) if provided or default
    p_minus_1 = max(0, args.period - 1)
    v_default_mu = 0.5
    v_default_sig = 0.5

    m0_season = _maybe_parse(args.m0_season, 5.0, p_minus_1)
    v0_season = _maybe_parse(args.v0_season, v_default_mu, p_minus_1)

    m0_season_sigma = _maybe_parse(args.m0_season_sigma, 0.0, p_minus_1)
    v0_season_sigma = _maybe_parse(args.v0_season_sigma, v_default_sig, p_minus_1)

    start_date = _parse_date(args.start_date)

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Define grids
    mu_levels   = ["dynamic", "deterministic"]
    mu_trends   = ["dynamic", "deterministic", "none"]
    mu_seasons  = ["dynamic", "deterministic", "none"]

    sig_levels  = ["dynamic", "deterministic"]
    sig_trends  = ["dynamic", "deterministic", "none"]
    sig_seasons = ["dynamic", "deterministic", "none"]

    combo_idx = 0
    for (lm, tm, sm), (lS, tS, sS) in product(product(mu_levels, mu_trends, mu_seasons),
                                              product(sig_levels, sig_trends, sig_seasons)):
        run_one_combo(
            args=args,
            m0_season=m0_season,
            v0_season=v0_season,
            m0_season_sigma=m0_season_sigma,
            v0_season_sigma=v0_season_sigma,
            lm=lm, tm=tm, sm=sm,
            lS=lS, tS=tS, sS=sS,
            start_date=start_date,
            out_dir=args.out_dir,
            seed_bump=combo_idx,
            save_csv=args.save_csv
        )
        combo_idx += 1

    print(f"Saved figures for {combo_idx} combinations to {args.out_dir}")


if __name__ == "__main__":
    main()
