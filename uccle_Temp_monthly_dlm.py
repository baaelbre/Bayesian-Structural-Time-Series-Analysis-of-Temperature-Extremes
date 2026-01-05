# run_uccle_dlm_lasso_monthly.py
# ------------------------------------------------------------
# Uccle monthly DLM (Gaussian) with NCP FFBS + hierarchical Bayesian lasso
# Uses: optimization/dlm_3.py (DLMGibbsConjugate) + optimization/ffbs.py (ffbs_dlm_ncp)
#
# Examples
#   python -u run_uccle_dlm_lasso_monthly.py --series TXm
#   python -u run_uccle_dlm_lasso_monthly.py --series TNm
#   python -u run_uccle_dlm_lasso_monthly.py --all
#
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # always safe for batch runs
import matplotlib.pyplot as plt

from optimization.dlm_3 import DLMGibbsConjugate, Priors, SamplerConfig


# =============================================================================
# Small helpers
# =============================================================================
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


def _parse_csv_floats(s: Optional[str], expected_len: Optional[int] = None) -> Optional[List[float]]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "":
        return None
    vals = [float(z) for z in ss.split(",")]
    if expected_len is not None and len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated floats, got {len(vals)}")
    return vals


def _series_out_root(series: str, base: Path) -> Path:
    """
    Uccle directory convention (monthly means):
      TXm -> results/uccle/TX/TXm/Monthly
      TNm -> results/uccle/TN/TNm/Monthly
    """
    mapping = {
        "TXm": base / "TX" / "TXm" / "Monthly",
        "TNm": base / "TN" / "TNm" / "Monthly",
    }
    if series not in mapping:
        raise ValueError(f"Unknown series '{series}'. Expected one of {list(mapping)}.")
    return mapping[series]


def _date_tag_from_index(idx: pd.Index) -> str:
    if len(idx) == 0:
        return "NA-NA"
    if isinstance(idx, pd.DatetimeIndex):
        return f"{idx[0].strftime('%Y-%m')}-{idx[-1].strftime('%Y-%m')}"
    if isinstance(idx, pd.PeriodIndex):
        return f"{str(idx[0])}-{str(idx[-1])}"
    return f"{str(idx[0])}-{str(idx[-1])}"


def _index_to_strings(idx: pd.Index) -> np.ndarray:
    # robust for DatetimeIndex/PeriodIndex/RangeIndex
    return np.asarray([str(x) for x in idx], dtype="U")


# =============================================================================
# Data loading (monthly)
# =============================================================================
def load_monthly_series(csv_path: Path, *, series_name_hint: Optional[str] = None) -> pd.Series:
    """
    Load a univariate monthly series from CSV.

    Accepted index formats:
      (i) a single date-like column among: date/time/Date/Time
      (ii) year+month columns among: (year|Year|YYYY) and (month|Month|MM)
      (iii) otherwise: RangeIndex

    Value column:
      - 'value' if present
      - else column equal to series_name_hint if present (e.g. TXm/TNm)
      - else first numeric column

    Post-processing:
      - drop NA
      - sort by index (when possible)
      - trim leading months until January (so seasonal dummy alignment is clean)
      - trim length to multiple of 12 (whole number of years)
    """
    df = pd.read_csv(csv_path)

    # ---- values ----
    if "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce")
    elif series_name_hint is not None and series_name_hint in df.columns:
        vals = pd.to_numeric(df[series_name_hint], errors="coerce")
    else:
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            # try coerce everything once
            tmp = df.copy()
            for c in tmp.columns:
                tmp[c] = pd.to_numeric(tmp[c], errors="ignore")
            numcols = [c for c in tmp.columns if pd.api.types.is_numeric_dtype(tmp[c])]
            df = tmp
        if not numcols:
            raise ValueError(f"No numeric column found in {csv_path}")
        vals = pd.to_numeric(df[numcols[0]], errors="coerce")

    # ---- index ----
    idx: pd.Index | None = None

    # (i) common date column
    for cand in ("date", "time", "Date", "Time"):
        if cand in df.columns:
            dt = pd.to_datetime(df[cand], errors="coerce")
            ok = dt.notna() & vals.notna()
            idx = pd.DatetimeIndex(dt[ok])
            vals = vals[ok].reset_index(drop=True)
            break

    # (ii) year+month
    if idx is None:
        year_col = next((c for c in ("year", "Year", "YYYY") if c in df.columns), None)
        month_col = next((c for c in ("month", "Month", "MM") if c in df.columns), None)
        if year_col is not None and month_col is not None:
            yy = pd.to_numeric(df[year_col], errors="coerce")
            mm = pd.to_numeric(df[month_col], errors="coerce")
            ok = yy.notna() & mm.notna() & vals.notna()
            idx = pd.PeriodIndex(
                year=yy[ok].astype(int),
                month=mm[ok].astype(int),
                freq="M",
            )
            vals = vals[ok].reset_index(drop=True)

    # (iii) fallback
    if idx is None:
        ok = vals.notna()
        vals = vals[ok].reset_index(drop=True)
        idx = pd.RangeIndex(len(vals), name="t")

    ser = pd.Series(np.asarray(vals, float), index=idx, name=series_name_hint or csv_path.stem).dropna()

    # sort where possible
    try:
        ser = ser.sort_index()
    except Exception:
        pass

    # enforce start at January for clean monthly seasonal alignment
    if isinstance(ser.index, (pd.DatetimeIndex, pd.PeriodIndex)) and len(ser) > 0:
        months = ser.index.month
        if int(months[0]) != 1:
            # drop until first January
            mask = (months == 1)
            if bool(np.any(mask)):
                first = int(np.argmax(np.asarray(mask, dtype=bool)))
                ser = ser.iloc[first:]

    # trim to whole years
    n = len(ser) - (len(ser) % 12)
    if n < 12:
        raise ValueError(f"{csv_path} yields < 12 observations after trimming; cannot run monthly DLM.")
    return ser.iloc[:n]


# =============================================================================
# Core runner
# =============================================================================
def run_one(
    *,
    series: str,
    y_ser: pd.Series,
    out_root: Path,
    period: int,
    priors: Priors,
    cfg: SamplerConfig,
    sigma_init: float,
    s_alpha_init: float,
    s_beta_init: float,
    s_gamma_init: float,
    gamma0_init: Optional[List[float]] = None,
    make_plots: bool = True,
) -> None:
    y = y_ser.to_numpy(dtype=float)
    T = int(y.size)
    idx = y_ser.index

    # ---- initial baseline guesses ----
    # alpha0/beta0 here are "time-0" baseline level/trend for the CP mapping.
    # keep it simple/robust: level ~ mean, trend ~ 0.
    alpha0_init = float(np.mean(y))
    beta0_init = 0.0

    # gamma0 baseline vector (length p-1)
    K = period - 1
    if gamma0_init is not None and len(gamma0_init) != K:
        raise ValueError(f"gamma0_init must have length {K} for period={period}")

    # ---- build sampler ----
    sampler = DLMGibbsConjugate(
        y=y,
        period=period,
        alpha0=alpha0_init,
        beta0=beta0_init,
        gamma0=gamma0_init,                 # None => zeros
        sigma2_init=float(sigma_init) ** 2,
        s_alpha_init=float(s_alpha_init),
        s_beta_init=float(s_beta_init),
        s_gamma_init=float(s_gamma_init),
        priors=priors,
        cfg=cfg,
    )

    # ---- output paths ----
    _ensure_dir(out_root)
    date_tag = _date_tag_from_index(idx)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{series}_monthly_{date_tag}_{stamp}"
    _ensure_dir(run_dir)

    # ---- run ----
    print(f"\n[Uccle DLM] series={series} | T={T} | period={period}")
    print(f"out: {run_dir}")
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"[done] elapsed = {elapsed:.1f}s")

    # ---- save posterior ----
    out_npz = run_dir / "posterior.npz"
    sampler.save_posterior(
        out_npz_path=str(out_npz),
        extra_meta={
            "series": series,
            "frequency": "monthly",
            "period": int(period),
            "T": int(T),
            "date_tag": date_tag,
            "index_type": type(idx).__name__,
            "index_start": str(idx[0]),
            "index_end": str(idx[-1]),
            "elapsed_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
            "priors": asdict(priors),
            "cfg": asdict(cfg),
            "model_notes": {
                "ncp": True,
                "seasonal_noise_first_component_only": True,
                "centered_time_prior_alpha_c_beta_correlated": True,
                "process_sds": "hierarchical_bayesian_lasso_on_signed_s",
            },
        },
    )

    # also store the index as strings (so plots can be reproduced without CSV)
    np.save(run_dir / "index_strings.npy", _index_to_strings(idx))

    # ---- plots ----
    if make_plots:
        # fit plot (mu band)
        mu = post["mu"]  # (n_kept, T)
        mu_hat = mu.mean(axis=0)
        lo, hi = np.quantile(mu, [0.05, 0.95], axis=0)

        plt.figure(figsize=(12, 4))
        plt.plot(idx, y, lw=1, label=series)
        plt.plot(idx, mu_hat, lw=1.5, linestyle="--", label=r"$\hat\mu_t$ (post. mean)")
        plt.fill_between(idx, lo, hi, alpha=0.2, label="90% CI for $\\mu_t$")
        plt.title(f"{series} — monthly DLM (NCP FFBS + Bayesian lasso), period={period}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "fit_mu.png", dpi=150)
        plt.close()

        # traces (a few key scalars)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(post["sigma"], lw=1, label="sigma")
        ax.plot(np.sqrt(np.maximum(post["Q_alpha"], 0.0)), lw=1, label="sqrt(Q_alpha)")
        ax.plot(np.sqrt(np.maximum(post["Q_beta"], 0.0)), lw=1, label="sqrt(Q_beta)")
        ax.plot(np.sqrt(np.maximum(post["Q_gamma"], 0.0)), lw=1, label="sqrt(Q_gamma)")
        ax.plot(post["lambda2"], lw=1, label="lambda2")
        ax.set_title(f"{series} — traces (kept draws)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "traces.png", dpi=150)
        plt.close(fig)

    print(f"[saved] {out_npz}")
    print(f"[saved] {run_dir / 'fit_mu.png' if make_plots else '(plots disabled)'}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Uccle monthly Gaussian DLM (NCP FFBS) with hierarchical Bayesian lasso on process SDs."
    )

    # data / series
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--series", type=str, default="TNm", choices=["TXm", "TNm"])
    p.add_argument("--all", action="store_true", help="Run both TXm and TNm.")
    p.add_argument("--txm-csv", type=str, default="TXm.csv")
    p.add_argument("--tnm-csv", type=str, default="TNm.csv")

    # output
    p.add_argument("--results-root", type=str, default="results/uccle")
    p.add_argument("--plots", default=True)

    # sampler
    p.add_argument("--n-iter", type=int, default=5000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=10)

    # priors
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=2.0)
    p.add_argument("--prior-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-P0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-P0-beta", type=float, default=1e-5)
    p.add_argument("--prior-m0-gamma", type=str, default=None, help="CSV floats length 11 (period-1), else zeros")
    p.add_argument("--prior-P0-gamma", type=float, default=5.0)
    p.add_argument("--prior-a-lambda", type=float, default=0.001) # put to 2,0.05 for more shrinkage
    p.add_argument("--prior-b-lambda", type=float, default=0.001) # standard: 0.001, 0.001

    # initials
    p.add_argument("--sigma-init", type=float, default=None, help="If omitted: 0.3*sd(y)")
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-3)
    p.add_argument("--gamma0-init", type=str, default=None, help="CSV floats length 11 (period-1), else zeros")

    args = p.parse_args()

    period = 12
    K = period - 1

    data_dir = Path(args.data_dir)
    results_root = Path(args.results_root)

    make_plots = _parse_bool(args.plots)
    do_progress = _parse_bool(args.progress)

    # priors
    m0_gamma = _parse_csv_floats(args.prior_m0_gamma, expected_len=K)
    if m0_gamma is None:
        m0_gamma = [0.0] * K

    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        m0_alpha=float(args.prior_m0_alpha),
        P0_alpha=float(args.prior_P0_alpha),
        m0_beta=float(args.prior_m0_beta),
        P0_beta=float(args.prior_P0_beta),
        m0_gamma=m0_gamma,
        P0_gamma=float(args.prior_P0_gamma),
        a_lambda=float(args.prior_a_lambda),
        b_lambda=float(args.prior_b_lambda),
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(do_progress),
        progress_every=int(args.progress_every),
    )

    gamma0_init = _parse_csv_floats(args.gamma0_init, expected_len=K)

    # decide which series to run
    todo = ["TXm", "TNm"] if bool(args.all) else [str(args.series)]

    for series in todo:
        csv_name = args.txm_csv if series == "TXm" else args.tnm_csv
        csv_path = data_dir / csv_name

        y_ser = load_monthly_series(csv_path, series_name_hint=series)

        # sigma init
        if args.sigma_init is None:
            sd = float(np.std(y_ser.to_numpy(dtype=float), ddof=1))
            sigma_init = max(0.3 * sd, 1e-6)
        else:
            sigma_init = float(args.sigma_init)

        out_root = _series_out_root(series, results_root)

        run_one(
            series=series,
            y_ser=y_ser,
            out_root=out_root,
            period=period,
            priors=priors,
            cfg=cfg,
            sigma_init=sigma_init,
            s_alpha_init=float(args.s_alpha_init),
            s_beta_init=float(args.s_beta_init),
            s_gamma_init=float(args.s_gamma_init),
            gamma0_init=gamma0_init,
            make_plots=make_plots,
        )

    print("\n[finished]")
    for series in todo:
        print(f"  {series}: {str(_series_out_root(series, results_root))}")


if __name__ == "__main__":
    main()
