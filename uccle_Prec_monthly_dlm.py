# run_uccle_dlm_prec_monthly.py
# ------------------------------------------------------------
# Uccle monthly precipitation (Precm, period=12) with:
#   - Gaussian DLM
#   - dummy monthly seasonal component (p=12; seasonal baseline length p-1; sum-to-zero)
#   - non-centred parametrisation (NCP) + FFBS
#   - hierarchical Bayesian lasso prior on signed process SDs
#
# Style/behavior matches run_uccle_dlm_lasso_monthly.py.
#
# Examples:
#   python -u run_uccle_dlm_prec_monthly.py
#   python -u run_uccle_dlm_prec_monthly.py --n-iter 30000 --burn 15000 --thin 2
#   python -u run_uccle_dlm_prec_monthly.py --gamma0-init "0,0,0,0,0,0,0,0,0,0,0"
#
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # always safe for batch runs
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Import from project (robust)
# ---------------------------------------------------------------------
try:
    from optimization.dlm_3 import DLMGibbsConjugate, Priors, SamplerConfig  # type: ignore
except Exception:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from optimization.dlm_3 import DLMGibbsConjugate, Priors, SamplerConfig  # type: ignore


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


def _parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "":
        return None
    return [float(z) for z in ss.split(",") if str(z).strip() != ""]


def _series_out_root(series: str, base: Path) -> Path:
    """
    Uccle directory convention (monthly precipitation):
      Precm -> results/uccle/Prec/Precm/Monthly
    """
    mapping = {
        "Precm": base / "Prec" / "Precm" / "Monthly",
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


def _start_end_from_index(idx: pd.Index) -> Tuple[str, str]:
    if len(idx) == 0:
        return ("NA", "NA")
    if isinstance(idx, pd.DatetimeIndex):
        return (idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d"))
    return (str(idx[0]), str(idx[-1]))


def _index_to_strings(idx: pd.Index) -> np.ndarray:
    return np.asarray([str(x) for x in idx], dtype="U")


# =============================================================================
# Seasonal utilities (sum-to-zero parametrisation, gamma0 length p-1)
# =============================================================================
def _to_gamma0_pminus1(vals: List[float], period: int) -> np.ndarray:
    """
    Convert a user-provided seasonal pattern to gamma0 (length p-1) for sum-to-zero:
      seasonal_full = [gamma0..., -sum(gamma0)]

    Accepts:
      - p-1 floats: interpreted directly as gamma0
      - p floats  : mean-center to sum-to-zero, then take first p-1 as gamma0
    """
    p = int(period)
    arr = np.asarray(vals, float).ravel()
    if arr.size == p - 1:
        return arr.copy()
    if arr.size == p:
        arr = arr - float(arr.mean())
        arr = arr - float(arr.sum()) / float(p)  # enforce exact sum-to-zero numerically
        return arr[: p - 1].copy()
    raise ValueError(f"Season pattern must have length {p-1} or {p}, got {arr.size}.")


def _month_ids(idx: pd.Index, period: int) -> np.ndarray:
    """
    Month IDs in {0,...,period-1}, aligned with index.
    """
    n = len(idx)
    if isinstance(idx, pd.PeriodIndex):
        return (idx.month - 1).astype(int)
    if isinstance(idx, pd.DatetimeIndex):
        return (idx.month - 1).astype(int)
    return (np.arange(n) % int(period)).astype(int)


def _gamma0_init_from_monthly_means(y: np.ndarray, idx: pd.Index, period: int) -> np.ndarray:
    """
    Data-driven init for gamma0 (length p-1):
      - compute mean per month
      - center across months so sum-to-zero
      - take first p-1 entries (last implied)
    """
    p = int(period)
    mid = _month_ids(idx, p)
    eff = np.zeros(p, float)
    overall = float(np.mean(y)) if y.size else 0.0
    for m in range(p):
        mask = (mid == m)
        eff[m] = float(np.mean(y[mask])) if np.any(mask) else overall
    eff = eff - float(np.mean(eff))
    eff = eff - float(eff.sum()) / float(p)
    return eff[: p - 1].copy()


# =============================================================================
# Data loading (monthly)
# =============================================================================
def load_monthly_series(
    csv_path: Path,
    *,
    series_name_hint: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    force_start_january: bool = True,
    trim_full_years: bool = True,
) -> pd.Series:
    """
    Load a univariate monthly series from CSV.

    Accepted index formats:
      (i) a single date-like column among: date/time/Date/Time/datetime
      (ii) year+month columns among: (year|Year|YYYY) and (month|Month|MM)
      (iii) otherwise: RangeIndex

    Value column:
      - 'value' if present
      - else column equal to series_name_hint if present (e.g. Precm)
      - else first numeric column

    Post-processing:
      - drop NA
      - sort by index (when possible)
      - optional year filter (if datetime/period index)
      - optionally trim leading months until January (clean seasonal dummy alignment)
      - optionally trim length to multiple of 12 (whole number of years)
      - convert DatetimeIndex to PeriodIndex('M') for consistent tagging
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
    for cand in ("date", "time", "Date", "Time", "datetime", "Datetime", "DATE", "TIME"):
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
            idx = pd.PeriodIndex(year=yy[ok].astype(int), month=mm[ok].astype(int), freq="M")
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

    # year filter (only when index supports it)
    if start_year is not None or end_year is not None:
        if isinstance(ser.index, pd.DatetimeIndex):
            y = ser.index.year
            lo = -10**9 if start_year is None else int(start_year)
            hi = 10**9 if end_year is None else int(end_year)
            ser = ser[(y >= lo) & (y <= hi)]
        elif isinstance(ser.index, pd.PeriodIndex):
            y = ser.index.year
            lo = -10**9 if start_year is None else int(start_year)
            hi = 10**9 if end_year is None else int(end_year)
            ser = ser[(y >= lo) & (y <= hi)]

    # enforce start at January for clean monthly seasonal alignment
    if force_start_january and isinstance(ser.index, (pd.DatetimeIndex, pd.PeriodIndex)) and len(ser) > 0:
        months = ser.index.month
        if int(months[0]) != 1:
            mask = (months == 1)
            if bool(np.any(mask)):
                first = int(np.argmax(np.asarray(mask, dtype=bool)))
                ser = ser.iloc[first:]

    # trim to whole years
    if trim_full_years:
        n = len(ser) - (len(ser) % 12)
        if n < 12:
            raise ValueError(f"{csv_path} yields < 12 observations after trimming; cannot run monthly DLM.")
        ser = ser.iloc[:n]

    # convert DatetimeIndex to PeriodIndex for consistent month tagging
    if isinstance(ser.index, pd.DatetimeIndex):
        ser.index = ser.index.to_period("M")

    return ser


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
    gamma0_init: Optional[np.ndarray],
    level_mode: str = "dynamic",
    trend_mode: str = "dynamic",
    seasonal_mode: str = "dynamic",
    make_plots: bool = True,
) -> None:
    y = y_ser.to_numpy(dtype=float)
    T = int(y.size)
    idx = y_ser.index

    # ---- initial baseline guesses ----
    alpha0_init = float(np.mean(y))
    beta0_init = 0.0

    # ---- build sampler (pass modes if supported; otherwise fall back) ----
    kwargs = dict(
        y=y,
        period=int(period),
        alpha0=float(alpha0_init),
        beta0=float(beta0_init),
        gamma0=None if gamma0_init is None else gamma0_init.astype(float),
        sigma2_init=float(sigma_init) ** 2,
        s_alpha_init=float(s_alpha_init),
        s_beta_init=float(s_beta_init),
        s_gamma_init=float(s_gamma_init),
        priors=priors,
        cfg=cfg,
    )
    try:
        sampler = DLMGibbsConjugate(
            level_mode=str(level_mode),
            trend_mode=str(trend_mode),
            seasonal_mode=str(seasonal_mode),
            **kwargs,
        )
    except TypeError:
        sampler = DLMGibbsConjugate(**kwargs)

    # ---- output paths ----
    _ensure_dir(out_root)
    date_tag = _date_tag_from_index(idx)
    start_date, end_date = _start_end_from_index(idx)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes_tag = f"{level_mode}_{trend_mode}_{seasonal_mode}"
    run_dir = out_root / f"{series}_monthly_{date_tag}_{modes_tag}_{stamp}"
    _ensure_dir(run_dir)

    # ---- run ----
    print(f"\n[Uccle DLM Prec] series={series} | T={T} | period={period}")
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
            "start_date": start_date,
            "end_date": end_date,
            "index_type": type(idx).__name__,
            "index_start": str(idx[0]),
            "index_end": str(idx[-1]),
            "elapsed_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
            "priors": asdict(priors),
            "cfg": asdict(cfg),
            "model": "DLM_GAUSSIAN_NCP_LASSO_DUMMIES",
            "modes": {
                "level_mode": str(level_mode),
                "trend_mode": str(trend_mode),
                "seasonal_mode": str(seasonal_mode),
            },
            "model_notes": {
                "ncp": True,
                "seasonality": "dummy_encoded_sum_to_zero_length_p_minus_1",
                "process_sds": "hierarchical_bayesian_lasso_on_signed_s",
            },
        },
    )

    # also store the index as strings (so plots can be reproduced without CSV)
    np.save(run_dir / "index_strings.npy", _index_to_strings(idx))

    # ---- plots ----
    if make_plots:
        mu = post["mu"]  # (n_kept, T)
        mu_hat = mu.mean(axis=0)
        lo, hi = np.quantile(mu, [0.05, 0.95], axis=0)

        x = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else idx

        plt.figure(figsize=(12, 4))
        plt.plot(x, y, lw=1, label=series)
        plt.plot(x, mu_hat, lw=1.5, linestyle="--", label=r"$\hat\mu_t$ (post. mean)")
        plt.fill_between(x, lo, hi, alpha=0.2, label="90% CI for $\\mu_t$")
        plt.title(f"{series} — monthly DLM (NCP FFBS + Bayesian lasso), period={period}, modes={modes_tag}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "fit_mu.png", dpi=150)
        plt.close()

        # traces (a few key scalars)
        fig, ax = plt.subplots(figsize=(12, 6))
        if "sigma" in post:
            ax.plot(post["sigma"], lw=1, label="sigma")
        for name in ("Q_alpha", "Q_beta", "Q_gamma"):
            if name in post:
                ax.plot(np.sqrt(np.maximum(post[name], 0.0)), lw=1, label=f"sqrt({name})")
        if "lambda2" in post:
            ax.plot(post["lambda2"], lw=1, label="lambda2")
        ax.set_title(f"{series} — traces (kept draws)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "traces.png", dpi=150)
        plt.close(fig)

    print(f"[saved] {out_npz}")
    if make_plots:
        print(f"[saved] {run_dir / 'fit_mu.png'}")
        print(f"[saved] {run_dir / 'traces.png'}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Uccle monthly precipitation Gaussian DLM (NCP FFBS) with Bayesian lasso on process SDs."
    )

    # data
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--precm-file", type=str, default="Precm.csv")
    p.add_argument("--start-year", type=int, default=1892)
    p.add_argument("--end-year", type=int, default=2022)
    p.add_argument("--force-start-january", default=True)
    p.add_argument("--trim-full-years", default=True)

    # output
    p.add_argument("--results-root", type=str, default="results/uccle")
    p.add_argument("--plots", default=True)

    # model
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--level-mode", type=str, default="dynamic")
    p.add_argument("--trend-mode", type=str, default="dynamic")
    p.add_argument("--seasonal-mode", type=str, default="dynamic")

    # sampler
    p.add_argument("--n-iter", type=int, default=20000)
    p.add_argument("--burn", type=int, default=10000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=10)

    # priors
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=2.0)
    p.add_argument("--prior-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-P0-alpha", type=float, default=100.0)
    p.add_argument("--prior-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-P0-beta", type=float, default=10.0)
    p.add_argument(
        "--prior-m0-gamma",
        type=str,
        default=None,
        help="CSV floats length p-1 or p; if omitted: zeros.",
    )
    p.add_argument("--prior-P0-gamma", type=float, default=100.0)
    p.add_argument("--prior-a-lambda", type=float, default=0.001)
    p.add_argument("--prior-b-lambda", type=float, default=0.001)

    # initials
    p.add_argument("--sigma-init", type=float, default=None, help="If omitted: 0.3*sd(y)")
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-3)
    p.add_argument(
        "--gamma0-init",
        type=str,
        default=None,
        help="CSV floats length p-1 or p; if omitted: init from monthly means (sum-to-zero).",
    )

    args = p.parse_args()
    np.random.seed(int(args.seed))

    series = "Precm"
    period = int(args.period)
    K = period - 1

    data_dir = Path(args.data_dir)
    results_root = Path(args.results_root)

    make_plots = _parse_bool(args.plots)
    do_progress = _parse_bool(args.progress)
    force_start_january = _parse_bool(args.force_start_january)
    trim_full_years = _parse_bool(args.trim_full_years)

    # ---- load data ----
    csv_path = data_dir / str(args.precm_file)
    y_ser = load_monthly_series(
        csv_path,
        series_name_hint=series,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        force_start_january=force_start_january,
        trim_full_years=trim_full_years,
    )

    y = y_ser.to_numpy(dtype=float)

    # ---- priors ----
    m0_gamma_vals = _parse_csv_floats(args.prior_m0_gamma)
    if m0_gamma_vals is None:
        m0_gamma = [0.0] * K
    else:
        m0_gamma = _to_gamma0_pminus1(m0_gamma_vals, period).tolist()

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

    # ---- initials ----
    sd = float(np.std(y, ddof=1)) if y.size > 1 else 1.0
    if args.sigma_init is None:
        sigma_init = max(0.3 * sd, 1e-6)
    else:
        sigma_init = float(args.sigma_init)

    gamma0_init_vals = _parse_csv_floats(args.gamma0_init)
    if gamma0_init_vals is None:
        gamma0_init = _gamma0_init_from_monthly_means(y, y_ser.index, period=period)
    else:
        gamma0_init = _to_gamma0_pminus1(gamma0_init_vals, period)

    out_root = _series_out_root(series, results_root)

    run_one(
        series=series,
        y_ser=y_ser,
        out_root=out_root,
        period=period,
        priors=priors,
        cfg=cfg,
        sigma_init=float(sigma_init),
        s_alpha_init=float(args.s_alpha_init),
        s_beta_init=float(args.s_beta_init),
        s_gamma_init=float(args.s_gamma_init),
        gamma0_init=gamma0_init,
        level_mode=str(args.level_mode),
        trend_mode=str(args.trend_mode),
        seasonal_mode=str(args.seasonal_mode),
        make_plots=make_plots,
    )

    print("\n[finished]")
    print(f"  {series}: {str(out_root)}")


if __name__ == "__main__":
    main()
