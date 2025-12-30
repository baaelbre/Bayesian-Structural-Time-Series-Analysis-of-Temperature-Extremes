# run_uccle_dlm_prec_monthly.py  — monthly precipitation (period = 12)
# Gaussian DLM with:
#   - dummy monthly seasonal component (p=12; seasonal state length p-1, newest-first rotation)
#   - non-centred parametrisation (NCP) + FFBS
#   - hierarchical Bayesian lasso prior on signed process SDs
#
# Assumes: optimization/dlm.py exports (DLMGibbsConjugate, Priors, SamplerConfig)

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optimization.less_optimal_versions.dlm import (
    DLMGibbsConjugate,
    Priors,
    SamplerConfig,
)

DATA_DIR = Path("data")


# ======================================================================
# Small helpers
# ======================================================================
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _idx_for_matplotlib(idx: pd.Index):
    # Matplotlib cannot plot Period objects; convert to timestamps
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp()  # DatetimeIndex
    return idx


def _series_out_root(series: str) -> Path:
    """
    Map series name to required output root directories:

      Precm → results/uccle/Prec/Precm/Monthly/NCP_LASSO
    """
    base = Path("results/uccle")
    mapping = {
        "Precm": base / "Prec" / "Precm" / "Monthly",
    }

    if series not in mapping:
        raise ValueError(f"Unknown series '{series}' for output mapping.")
    return mapping[series]


def _date_tag_from_index(idx: pd.Index) -> str:
    """
    Build a date tag 'start-end' from a pandas index.
    • For DatetimeIndex: YYYY-MM-DD
    • For PeriodIndex: str(period)
    • Otherwise: str(index_value)
    """
    if len(idx) == 0:
        return "NA-NA"

    if isinstance(idx, pd.DatetimeIndex):
        start = idx[0].strftime("%Y-%m-%d")
        end = idx[-1].strftime("%Y-%m-%d")
    elif isinstance(idx, pd.PeriodIndex):
        start = str(idx[0])
        end = str(idx[-1])
    else:
        start = str(idx[0])
        end = str(idx[-1])

    return f"{start}-{end}"


def _month_id_from_index(idx: pd.Index, period: int) -> np.ndarray:
    """
    Return integer month IDs in {0,...,period-1} aligned with idx.
    If idx is monthly PeriodIndex/DatetimeIndex: use calendar month.
    Otherwise: fallback to t % period.
    """
    n = len(idx)
    if isinstance(idx, pd.PeriodIndex) and (idx.freqstr is not None) and ("M" in idx.freqstr):
        return (idx.month - 1).astype(int)
    if isinstance(idx, pd.DatetimeIndex):
        return (idx.month - 1).astype(int)
    return (np.arange(n) % period).astype(int)


def _gamma0_init_from_monthly_means(y: np.ndarray, idx: pd.Index, period: int) -> np.ndarray:
    """
    Sensible seasonal init (length period-1) from monthly means with sum-to-zero.

    Compute mean per month, center across months so sum_m effect_m = 0,
    then take the first period-1 entries; the last month is implied by
    -sum_{j=1}^{p-1} gamma_j.
    """
    month_id = _month_id_from_index(idx, period)
    effects = np.zeros(period, float)
    overall = float(np.mean(y)) if len(y) else 0.0
    for m in range(period):
        mask = (month_id == m)
        effects[m] = float(np.mean(y[mask])) if np.any(mask) else overall
    effects = effects - float(np.mean(effects))
    return effects[: period - 1].copy()


# ======================================================================
# Data loading
# ======================================================================
def load_series(csv_path: Path) -> pd.Series:
    """
    Load a univariate *monthly* precipitation time series from CSV and trim to a
    multiple of 12 (whole number of years).

    Expected flexible formats:
      - A single date-like column (date/time/Date/Time) parsable by pandas, OR
      - Separate year/month columns, OR
      - A Period-like column already.

    Values:
      - Uses 'value' column if present, otherwise 'Precm' if present,
        otherwise first numeric column.
    """
    df = pd.read_csv(csv_path)

    # --- values ---
    if "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce")
    elif "Precm" in df.columns:
        vals = pd.to_numeric(df["Precm"], errors="coerce")
    else:
        # fallback: first numeric column
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            raise ValueError(f"No numeric columns found in {csv_path}")
        vals = pd.to_numeric(df[numcols[0]], errors="coerce")

    # --- index ---
    idx = None

    # 1) common date column
    for cand in ["date", "time", "Date", "Time"]:
        if cand in df.columns:
            dt = pd.to_datetime(df[cand], errors="coerce")
            idx = dt
            break

    # 2) year+month columns
    if idx is None:
        year_col = None
        month_col = None
        for yc in ["year", "Year", "YYYY"]:
            if yc in df.columns:
                year_col = yc
                break
        for mc in ["month", "Month", "MM"]:
            if mc in df.columns:
                month_col = mc
                break

        if year_col is not None and month_col is not None:
            yy = pd.to_numeric(df[year_col], errors="coerce")
            mm = pd.to_numeric(df[month_col], errors="coerce")
            ok = yy.notna() & mm.notna()
            idx = pd.PeriodIndex(
                year=yy[ok].astype(int),
                month=mm[ok].astype(int),
                freq="M",
            )
            vals = vals[ok].reset_index(drop=True)

    # 3) fallback: RangeIndex
    if idx is None:
        idx = pd.RangeIndex(len(vals), name="t")

    ser = pd.Series(np.asarray(vals, float), index=idx, name=csv_path.stem).dropna()

    # If we have real dates, sort them
    try:
        ser = ser.sort_index()
    except Exception:
        pass

    # Optional: ensure we start on January for dummy-season alignment
    if isinstance(ser.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        month0 = int(ser.index[0].month)
        if month0 != 1:
            mask = ser.index.month == 1
            first_jan_pos = int(np.argmax(mask.to_numpy())) if mask.any() else 0
            ser = ser.iloc[first_jan_pos:]

    # trim to a multiple of 12 (full years)
    n = len(ser) - (len(ser) % 12)
    if n <= 0:
        raise ValueError(f"Series in {csv_path} is shorter than one full year (12 points).")
    ser = ser.iloc[:n]

    # for nicer tagging/plotting
    if isinstance(ser.index, pd.DatetimeIndex):
        ser.index = ser.index.to_period("M")

    return ser


# ======================================================================
# Core runner
# ======================================================================
def run_one(
    series: str,
    y_ser: pd.Series,
    out_root: Path,
    level_mode: str = "dynamic",
    trend_mode: str = "dynamic",
    seasonal_mode: str = "dynamic",
) -> None:
    """
    Run the non-centred DLM with dummy monthly seasonality and hierarchical Bayesian
    lasso prior on process SDs for a single *monthly* precipitation series.
    """
    y = y_ser.to_numpy(dtype=float)
    period = 12
    idx = y_ser.index[: len(y)]

    # --- priors ---
    # gamma0 prior vector must have length period-1 (newest-first convention inside model)
    m0_gamma_prior = [0.0] * (period - 1)

    pri = Priors(
        # obs precision prior: tau = 1/sigma^2 ~ Gamma(a_sigma, b_sigma) (shape-rate)
        a_sigma=2.0,
        b_sigma=2.0,
        # baseline priors (kept weak / scale-robust for precipitation)
        m0_alpha=0.0,
        P0_alpha=100.0,
        m0_beta=0.0,
        P0_beta=10.0,
        m0_gamma=m0_gamma_prior,
        P0_gamma=100.0,
        # lasso hyperprior on lambda^2
        a_lambda=0.001,
        b_lambda=0.001,
    )

    # --- sampler config ---
    cfg = SamplerConfig(
        n_iter=20000,
        burn=10000,
        thin=1,
        random_seed=42,
        progress=True,
        progress_every=10,
    )

    # --- initial values ---
    y_sd = float(np.std(y)) if len(y) > 1 else 1.0
    sigma2_init = float(np.var(y) * 0.2) if len(y) > 1 else 1.0

    init_level = float(np.mean(y[: min(len(y), period)])) if len(y) >= period else float(np.mean(y))
    init_trend = 0.0
    gamma0_init = _gamma0_init_from_monthly_means(y, idx, period=period)

    # signed process SDs (lasso will shrink if needed)
    s_alpha_init = 0.05 * y_sd
    s_beta_init = 0.01 * y_sd
    s_gamma_init = 0.05 * y_sd

    sampler = DLMGibbsConjugate(
        y=y,
        period=period,
        level_mode=level_mode,
        trend_mode=trend_mode,
        seasonal_mode=seasonal_mode,
        alpha0=init_level,
        beta0=init_trend,
        gamma0=gamma0_init,        # data-driven seasonal init (length p-1)
        sigma2_init=sigma2_init,
        s_alpha_init=s_alpha_init,
        s_beta_init=s_beta_init,
        s_gamma_init=s_gamma_init,
        priors=pri,
        cfg=cfg,
    )

    # --- output paths ---
    date_tag = _date_tag_from_index(idx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes_tag = f"{level_mode}_{trend_mode}_{seasonal_mode}"
    tag = f"{series}_{modes_tag}"

    outdir = out_root / f"{tag}_{timestamp}"
    _ensure_dir(outdir)

    # --- run ---
    print(f"Running Uccle monthly DLM (Bayesian lasso) for {series} with modes={modes_tag} ...")
    t0 = datetime.now().timestamp()
    post = sampler.run()
    elapsed = datetime.now().timestamp() - t0
    print(f"{series}: run time {elapsed:.2f} seconds")

    # --- save posterior ---
    npz_name = f"posterior_{series}_{date_tag}_{modes_tag}.npz"
    out_npz = outdir / npz_name

    sampler.save_posterior(
        out_npz_path=str(out_npz),
        extra_meta={
            "series": series,
            "label": f"{series}_monthly",
            "data_path": str(DATA_DIR / f"{series}.csv"),
            "index_type": type(idx).__name__,
            "index_values_preview": [str(ix) for ix in idx[: min(10, len(idx))]],
            "description": (
                "Gaussian DLM with dummy monthly seasonality (newest-first rotation), "
                "dynamic level/trend/season, non-centred parametrisation (FFBS), "
                "hierarchical Bayesian lasso prior on signed process SDs: "
                "s_k | τ_k, σ² ~ N(0, σ² τ_k), τ_k | λ² ~ Exp(λ²/2), "
                "λ² ~ Gamma(a_lambda, b_lambda)."
            ),
            "period": period,
            "date_tag": date_tag,
            "elapsed_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
            "lasso_hyperpriors": {
                "a_lambda": pri.a_lambda,
                "b_lambda": pri.b_lambda,
            },
            "modes": {
                "level_mode": level_mode,
                "trend_mode": trend_mode,
                "seasonal_mode": seasonal_mode,
            },
        },
    )

    # --- quick fit plot (post mean + 90% band) ---
    mu_draws = post["mu"]
    mu_hat = mu_draws.mean(axis=0)
    lo, hi = np.quantile(mu_draws, [0.05, 0.95], axis=0)

    idx_plot = _idx_for_matplotlib(idx)

    plt.figure(figsize=(12, 4))
    plt.plot(idx_plot, y, lw=1, label=series)
    plt.plot(idx_plot, mu_hat, "-.", lw=1.5, label=r"$\hat{\mu}_t$ (post. mean)")
    plt.fill_between(idx_plot, lo, hi, alpha=0.2, label=r"90% CI for $\mu_t$")
    plt.title(f"{series}: monthly DLM (Bayesian lasso on process SDs) — modes={modes_tag}, period={period}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fit_name = f"fit_{series}_{date_tag}_{modes_tag}.png"
    plt.savefig(outdir / fit_name, dpi=150)
    plt.close()


    print(f"Saved posterior to {out_npz}")
    print(f"Saved fit plot to {outdir / fit_name}\n")


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    series = "Precm"
    y_ser = load_series(DATA_DIR / f"{series}.csv")

    print(f"{series} head:\n{y_ser.head()}\n")

    out_root = _series_out_root(series)

    level_mode = "dynamic"
    trend_mode = "dynamic"
    seasonal_mode = "dynamic"

    run_one(series, y_ser, out_root, level_mode, trend_mode, seasonal_mode)

    print("Saved monthly results under:")
    print(f"  {out_root}/*")


if __name__ == "__main__":
    main()
