# run_uccle_precip_dummies_lasso_nc.py
# Monthly precipitation (period=12) using:
#   - dummy seasonal state-space (newest-first seasonal rotation)
#   - non-centred parametrisation (NCP) + FFBS
#   - hierarchical Bayesian lasso prior on signed process SDs
#
# IMPORTANT: adjust the import below to wherever you saved your DLMGibbsConjugate code.
# I assume: optimization/dlm_lasso_dummies_nc.py exports (DLMGibbsConjugate, Priors, SamplerConfig)

from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optimization.dlm import DLMGibbsConjugate, Priors, SamplerConfig

DATA_DIR = Path("data")


# ======================================================================
# Small helpers
# ======================================================================
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _series_out_root(series: str) -> Path:
    """
    Output roots (Uccle):

      Precm → results/uccle/Prec/Precm/Monthly/NCP_LASSO
    """
    base = Path("results/uccle")
    mapping = {
        "Precm": base / "Prec" / "Precm" / "Monthly" / "NCP_LASSO",
    }
    if series not in mapping:
        raise ValueError(f"Unknown series '{series}' for output mapping.")
    return mapping[series]


def _date_tag_from_index(idx: pd.Index) -> str:
    if len(idx) == 0:
        return "NA-NA"
    if isinstance(idx, pd.PeriodIndex):
        return f"{idx[0]}-{idx[-1]}"
    if isinstance(idx, pd.DatetimeIndex):
        return f"{idx[0].strftime('%Y-%m-%d')}-{idx[-1].strftime('%Y-%m-%d')}"
    return f"{str(idx[0])}-{str(idx[-1])}"


def _to_monthly_index(idx: pd.Index) -> pd.Index:
    """Try to coerce to a monthly PeriodIndex for nicer plotting/tagging."""
    if isinstance(idx, pd.PeriodIndex):
        if idx.freqstr and "M" in idx.freqstr:
            return idx
        return idx.asfreq("M", how="end")
    if isinstance(idx, pd.DatetimeIndex):
        return idx.to_period("M")
    return idx


def _month_id_from_index(idx: pd.Index, period: int) -> np.ndarray:
    """
    Return integer month IDs in {0,...,period-1} aligned with idx.
    If idx is monthly PeriodIndex/DatetimeIndex: use month number.
    Otherwise: fallback to t % period.
    """
    n = len(idx)
    if isinstance(idx, pd.PeriodIndex) and idx.freqstr and "M" in idx.freqstr:
        # month: 1..12 -> 0..11
        return (idx.month - 1).astype(int)
    if isinstance(idx, pd.DatetimeIndex):
        return (idx.month - 1).astype(int)
    return (np.arange(n) % period).astype(int)


def _gamma0_init_from_monthly_means(y: np.ndarray, idx: pd.Index, period: int) -> np.ndarray:
    """
    Build a sensible gamma0 init (length period-1) from monthly means with sum-to-zero.

    We compute mean per month, then center across months so sum_m effect_m = 0.
    Store the first period-1 entries; the last month is implicit in the model
    via -sum_{j=1}^{p-1} gamma0_j.
    """
    month_id = _month_id_from_index(idx, period)
    effects = np.zeros(period, float)
    for m in range(period):
        mask = (month_id == m)
        effects[m] = float(np.mean(y[mask])) if np.any(mask) else float(np.mean(y))
    effects = effects - float(np.mean(effects))  # enforce sum-to-zero across months
    return effects[: period - 1].copy()


# ======================================================================
# Data loading
# ======================================================================
def load_series(csv_path: Path) -> pd.Series:
    """
    Load a univariate monthly precipitation series from CSV and trim to full years (multiple of 12).

    Heuristics:
      - value column: 'value' if present else first numeric column.
      - index:
          * try 'date'/'time'/'Date'/'Time'
          * else try ('year','month') or ('Year','Month')
          * else RangeIndex
    """
    df = pd.read_csv(csv_path)

    # --- values ---
    if "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce")
    else:
        # ensure numeric coercion
        for c in df.columns:
            if c.lower() in {"date", "time"}:
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            raise ValueError(f"No numeric column found in {csv_path}")
        vals = df[numcols[0]]

    # --- index ---
    idx: pd.Index | None = None

    # date-like single column
    for cand in ["date", "time", "Date", "Time"]:
        if cand in df.columns:
            dt = pd.to_datetime(df[cand], errors="coerce")
            if dt.notna().sum() >= max(3, int(0.8 * len(dt))):
                idx = pd.DatetimeIndex(dt)
                break

    # (year, month)
    if idx is None:
        ycol = next((c for c in ["year", "Year"] if c in df.columns), None)
        mcol = next((c for c in ["month", "Month"] if c in df.columns), None)
        if ycol is not None and mcol is not None:
            year = pd.to_numeric(df[ycol], errors="coerce").astype("Int64")
            month = pd.to_numeric(df[mcol], errors="coerce").astype("Int64")
            ok = year.notna() & month.notna()
            if ok.sum() > 0:
                idx = pd.PeriodIndex(
                    year=year[ok].astype(int),
                    month=month[ok].astype(int),
                    freq="M",
                )
                vals = vals[ok].reset_index(drop=True)

    if idx is None:
        idx = pd.RangeIndex(len(vals), name="t")

    ser = pd.Series(vals.astype(float).to_numpy(), index=idx, name=csv_path.stem).dropna()

    # trim to full years (multiple of 12)
    n = len(ser) - (len(ser) % 12)
    if n <= 0:
        raise ValueError(f"Series in {csv_path} shorter than 12 points after cleaning.")
    ser = ser.iloc[:n]

    # nicer monthly index if possible
    ser.index = _to_monthly_index(ser.index)

    return ser


# ======================================================================
# Core runner
# ======================================================================
def run_one(series: str, y_ser: pd.Series, out_root: Path) -> None:
    y = y_ser.to_numpy(dtype=float)
    idx = y_ser.index[: len(y)]
    period = 12

    # ---- Priors (match your DLMGibbsConjugate priors) ----
    pri = Priors(
        a_sigma=2.0,
        b_sigma=1.0,
        m0_alpha=0.0,
        P0_alpha=10.0,
        m0_beta=0.0,
        P0_beta=10.0,
        m0_gamma=None,   # leave at 0 baseline; we’ll init gamma0 from data below
        P0_gamma=10.0,
        # lasso global:
        a_lambda=0.001,
        b_lambda=0.001,
    )

    # ---- Sampler config ----
    cfg = SamplerConfig(
        n_iter=20000,
        burn=10000,
        thin=1,
        random_seed=42,
        progress=True,
        progress_every=10,  # adjust
    )

    # ---- Initial values ----
    y_sd = float(np.std(y)) if len(y) > 1 else 1.0
    sigma2_init = float(np.var(y) * 0.2) if len(y) > 1 else 1.0

    alpha0_init = float(np.mean(y[:12])) if len(y) >= 12 else float(np.mean(y))
    beta0_init = 0.0
    gamma0_init = _gamma0_init_from_monthly_means(y, idx, period=period)

    # (Signed) process SD inits (lasso will shrink if needed)
    s_alpha_init = 0.05 * y_sd
    s_beta_init = 0.01 * y_sd
    s_gamma_init = 0.05 * y_sd

    sampler = DLMGibbsConjugate(
        y=y,
        period=period,
        level_mode="dynamic",
        trend_mode="dynamic",
        seasonal_mode="dynamic",
        alpha0=alpha0_init,
        beta0=beta0_init,
        gamma0=gamma0_init,
        sigma2_init=sigma2_init,
        s_alpha_init=s_alpha_init,
        s_beta_init=s_beta_init,
        s_gamma_init=s_gamma_init,
        priors=pri,
        cfg=cfg,
    )

    # ---- Output dir ----
    date_tag = _date_tag_from_index(idx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{series}_dynamic-dynamic-dynamic"
    outdir = out_root / f"{tag}_{date_tag}_{timestamp}"
    _ensure_dir(outdir)

    # ---- Run ----
    print(f"Running DLM (dummies + NCP + lasso) for {series} ...")
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"{series}: run time {elapsed:.1f} seconds")

    # ---- Save posterior (keep plotter-friendly name) ----
    out_npz = outdir / "posterior.npz"
    sampler.save_posterior(
        out_npz_path=str(out_npz),
        extra_meta={
            "series": series,
            "label": f"{series}_monthly",
            "data_path": str(DATA_DIR / f"{series}.csv"),
            "index_type": type(idx).__name__,
            "index_values_preview": [str(ix) for ix in idx[: min(10, len(idx))]],
            "date_tag": date_tag,
            "elapsed_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
            "description": (
                "Gaussian DLM with dummy seasonal component (newest-first rotation), "
                "non-centred parametrisation (FFBS), and hierarchical Bayesian lasso "
                "prior on signed process SDs (s_alpha, s_beta, s_gamma)."
            ),
        },
    )

    # ---- Quick fit plot ----
    mu_hat = post["mu"].mean(axis=0)
    t_index = idx

    plt.figure(figsize=(12, 4))
    plt.plot(t_index, y, label=series, lw=1)
    plt.plot(t_index, mu_hat, "-.", label="μ̂_t (post mean)")
    plt.title(f"{series}: DLM dummies + NCP lasso (period={period})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fit_name = "fit.png"
    plt.savefig(outdir / fit_name, dpi=150)
    plt.close()

    print(f"Saved posterior to {out_npz}")
    print(f"Saved fit plot to {outdir / fit_name}\n")


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    series = "Precm"
    csv_path = DATA_DIR / f"{series}.csv"
    y_ser = load_series(csv_path)

    print(f"{series} head:\n{y_ser.head()}\n")
    out_root = _series_out_root(series)
    run_one(series, y_ser, out_root)

    print("Saved results under:")
    print(f"  {out_root}/*")


if __name__ == "__main__":
    main()
