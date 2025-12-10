# run_uccle_dlm_double_gamma_monthly.py  — monthly summaries (period = 12)
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- import the non-centred DLM with double-gamma prior and dummy seasonality ----
from optimization.dlm_doublegamma_dummy_nc import (
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


def _series_out_root(series: str) -> Path:
    """
    Map series name to required output root directories:

      TXm → results/uccle/TX/TXm/Monthly/
      TNm → results/uccle/TN/TNm/Monthly/
    """
    base = Path("results/uccle")
    mapping = {
        "TXm": base / "TX" / "TXm" / "Monthly",
        "TNm": base / "TN" / "TNm" / "Monthly",
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


# ======================================================================
# Data loading
# ======================================================================
def load_series(csv_path: Path) -> pd.Series:
    """
    Load a univariate *monthly* time series from CSV and trim to a multiple of 12
    (whole number of years).

    Expected format (flexible):
      - One date-like column (e.g. 'date', 'Date', 'time', 'year' + 'month', ...).
      - One numeric column with the series values (e.g. 'TXm', 'TNm', 'value', ...).
    """
    df = pd.read_csv(csv_path)

    # pick 'value' if present, otherwise first numeric column
    if "value" in df.columns:
        s = pd.to_numeric(df["value"], errors="coerce")
    else:
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            # attempt coercion of non-numeric columns (skip the first, often an ID or date)
            for c in df.columns[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            raise ValueError(f"No numeric columns found in {csv_path}")
        s = df[numcols[0]]

    # try to parse a date/time column if present, otherwise integer index
    idx = None
    for cand in ["date", "time", "Date", "Time", "year", "Year"]:
        if cand in df.columns:
            idx = pd.to_datetime(df[cand], errors="coerce")
            break

    if idx is None:
        # fallback: simple integer index
        idx = pd.RangeIndex(len(s), name="t")

    ser = pd.Series(s.astype(float).to_numpy(), index=idx, name=csv_path.name).dropna()

    # trim length to a multiple of 12 (full years of monthly data)
    n = len(ser) - (len(ser) % 12)
    if n <= 0:
        raise ValueError(f"Series in {csv_path} is shorter than one full year (12 points).")
    return ser.iloc[:n]


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
    Run the non-centred DLM with dummy monthly seasonality and double-gamma
    global–local prior on process SDs for a single *monthly* series.

    Parameters
    ----------
    series : {"TXm", "TNm"}
        Short series code used in filenames ("TXm" or "TNm").
    y_ser : pd.Series
        Monthly time series.
    out_root : Path
        Base directory (e.g., results/uccle/TX/TXm/Monthly or results/uccle/TN/TNm/Monthly).
    level_mode, trend_mode, seasonal_mode : str
        Mode strings passed to DLMGibbsConjugate and used in filenames
        (currently enforced to be "dynamic"/"dynamic"/"dynamic").
    """
    y = y_ser.to_numpy(dtype=float)
    period = 12  # 12 months per year

    # ----- Priors: double-gamma on process SDs, weak normals on baselines ----- #
    # m0_gamma has length period-1, newest-first seasonal baseline parameters.
    m0_gamma_prior = [0.0] * (period - 1)

    pri = Priors(
        a_sigma=2.0,
        b_sigma=2.0,
        # m0 priors (weak, centred near 0; the data scale will dominate)
        m_m0_alpha=0.0,
        s_m0_alpha=10.0,
        m_m0_beta=0.0,
        s_m0_beta=10.0,
        m_m0_gamma=m0_gamma_prior,
        s_m0_gamma=5.0,
        # P0 priors (kept for compatibility / storage; not updated in NCP scheme)
        a_P0_alpha=5.0,
        b_P0_alpha=1.0,
        a_P0_beta=5.0,
        b_P0_beta=1.0,
        a_P0_gamma=5.0,
        b_P0_gamma=1.0,
        # Double-gamma hyperparameters: relatively mild shrinkage
        a_xi=1.0,
        b_xi=1.0,
        a_tau=1.0,
        b_tau=1.0,
    )

    # ----- Sampler configuration ----- #
    cfg = SamplerConfig(
        n_iter=10000,
        burn=5000,
        thin=1,
        random_seed=42,
        progress=True,
        progress_every=1,
    )

    # ----- Initial values ----- #
    # Observation variance initial guess: fraction of empirical variance
    sigma2_init = float(np.var(y) * 0.1) if len(y) > 1 else 1.0
    modes_tag = f"{level_mode}_{trend_mode}_{seasonal_mode}"

    # Rough initial level = mean of first year; trend = 0
    init_level = float(np.mean(y[: min(len(y), period)])) if len(y) >= period else float(np.mean(y))
    init_trend = 0.0

    # Process SD initial guesses: small random-walk noise
    s_alpha_init = 1e-2
    s_beta_init = 1e-3
    s_gamma_init = 1e-3

    sampler = DLMGibbsConjugate(
        y=y,
        period=period,
        level_mode=level_mode,
        trend_mode=trend_mode,
        seasonal_mode=seasonal_mode,
        # initial means/vars for dynamic level/trend
        m0_alpha_init=init_level,
        P0_alpha_init=0.25,
        m0_beta_init=init_trend,
        P0_beta_init=0.05,
        # seasonal baseline initialisation: let the sampler start at zeros
        m0_gamma_init=None,   # defaults to zeros of length period-1
        P0_gamma_init=0.25,
        # observation variance + process SD inits
        sigma2_init=sigma2_init,
        s_alpha_init=s_alpha_init,
        s_beta_init=s_beta_init,
        s_gamma_init=s_gamma_init,
        priors=pri,
        cfg=cfg,
    )

    # ----- Construct run-specific directories and filenames ----- #
    idx = y_ser.index[: len(y)]
    date_tag = _date_tag_from_index(idx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{series}_{modes_tag}"

    outdir = out_root / f"{tag}_{timestamp}"
    _ensure_dir(outdir)

    # ----- Run sampler ----- #
    print(f"Running monthly non-centred DLM (double-gamma, dummies) for {series} with modes={modes_tag} ...")
    t0 = datetime.now().timestamp()
    post = sampler.run()
    elapsed = datetime.now().timestamp() - t0
    print(f"{series}: run time {elapsed:.2f} seconds")

    # ----- Save posterior + metadata ----- #
    npz_name = f"posterior_{series}_{date_tag}_{modes_tag}.npz"
    out_npz = outdir / npz_name

    sampler.save_posterior(
        out_npz_path=str(out_npz),
        extra_meta={
            "series": series,
            "label": f"{series}_monthly",
            "index_type": type(idx).__name__,
            "index_values_preview": [str(ix) for ix in idx[: min(10, len(idx))]],
            "modes": {
                "level_mode": level_mode,
                "trend_mode": trend_mode,
                "seasonal_mode": seasonal_mode,
            },
            "description": (
                "Gaussian DLM with dummy monthly seasonality, "
                "dynamic level/trend/season, non-centred parametrisation, "
                "double-gamma global-local prior on process SDs "
                "s_k | ξ_k, τ, σ² ~ N(0, σ² / (ξ_k τ))."
            ),
            "period": period,
            "date_tag": date_tag,
            "elapsed_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
            "double_gamma_priors": {
                "a_xi": pri.a_xi,
                "b_xi": pri.b_xi,
                "a_tau": pri.a_tau,
                "b_tau": pri.b_tau,
            },
        },
    )

    # ----- Quick fit plot ----- #
    mu_hat = post["mu"].mean(axis=0)
    t_index = idx

    plt.figure(figsize=(12, 4))
    plt.plot(t_index, y, label=f"{series}")
    plt.plot(t_index, mu_hat, "-.", label="μ̂_t")
    plt.title(
        f"{series}: monthly DLM with dummy seasonality "
        f"(L/T/S={modes_tag}, period={period})"
    )
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
    # Monthly mean max / min temperatures
    tx = load_series(DATA_DIR / "TXm.csv")
    tn = load_series(DATA_DIR / "TNm.csv")

    print("TXm head:\n", tx.head(), "\n")
    print("TNm head:\n", tn.head(), "\n")

    # Base roots (with 'results/uccle' prefix)
    tx_root = _series_out_root("TXm")  # results/uccle/TX/TXm/Monthly
    tn_root = _series_out_root("TNm")  # results/uccle/TN/TNm/Monthly

    # Modes (for now, dynamic/dynamic/dynamic only; enforced inside DLMGibbsConjugate)
    level_mode = "dynamic"
    trend_mode = "dynamic"
    seasonal_mode = "dynamic"

    run_one("TXm", tx, tx_root, level_mode, trend_mode, seasonal_mode)
    run_one("TNm", tn, tn_root, level_mode, trend_mode, seasonal_mode)

    print("Saved monthly results under:")
    print(f"  {tx_root}/*")
    print(f"  {tn_root}/*")


if __name__ == "__main__":
    main()
