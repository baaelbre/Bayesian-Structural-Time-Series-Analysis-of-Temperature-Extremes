# run_uccle_harmonic.py
import os
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- import the harmonic DLM with log-normal process SDs ----
from optimization.dlm_lognormal_harmonic import (
    DLMGibbsHarmonic,
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

      TXm → results/TX/TXm/Seasonal/
      TNm → results/TN/TNm/Seasonal/
    """
    base = Path("results/uccle")
    mapping = {
        "TXm": base / "TX" / "TXm" / "Seasonal",
        "TNm": base / "TN" / "TNm" / "Seasonal",
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
    Load a univariate time series from CSV and trim to a multiple of 4
    (DJF/MAM/JJA/SON seasonal means).
    """
    df = pd.read_csv(csv_path)

    # pick 'value' if present, otherwise first numeric column
    if "value" in df.columns:
        s = pd.to_numeric(df["value"], errors="coerce")
    else:
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            # attempt coercion of non-numeric columns
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
        idx = pd.RangeIndex(len(s), name="t")

    ser = pd.Series(s.astype(float).to_numpy(), index=idx, name=csv_path.name).dropna()

    # trim length to a multiple of 4 (seasonal cycle DJF/MAM/JJA/SON)
    n = len(ser) - (len(ser) % 4)
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
    Run the harmonic DLM (cos/sin + optional Nyquist) on a single seasonal series.

    Parameters
    ----------
    series : {"TXm", "TNm"}
        Short series code used in filenames ("TXm" or "TNm").
    y_ser : pd.Series
        Time series of seasonal means.
    out_root : Path
        Base directory (e.g., results/TX/TXm/Seasonal or results/TN/TNm/Seasonal).
    level_mode, trend_mode, seasonal_mode : str
        Mode strings passed to DLMGibbsHarmonic and used in filenames.
    """
    y = y_ser.to_numpy(float)
    period = 4  # four seasons per year (DJF/MAM/JJA/SON)

    # ----- Priors (log-normal on process SDs + weak-normal on m0) ----- #
    pri = Priors(
        a_sigma=2.0,
        b_sigma=1.0,
        # m0 priors (weak)
        m_m0_alpha=0.0,
        s_m0_alpha=10.0,
        m_m0_beta=0.0,
        s_m0_beta=10.0,
        # harmonic m0 priors (cos/sin/nyq); keep mean 0, moderately vague SD
        m_m0_nyq=0.0,
        s_m0_harm=5.0,
        m_m0_cos=None,  # default: zeros
        m_m0_sin=None,  # default: zeros
        # P0 priors (Inv-Gamma)
        a_P0_alpha=5.0,
        b_P0_alpha=1.0,
        a_P0_beta=5.0,
        b_P0_beta=1.0,
        a_P0_harm=5.0,
        b_P0_harm=1.0,
        # log-normal priors for process SDs (ln s ~ N(mu, sd^2))
        ln_s_alpha_mu=-3.0,
        ln_s_alpha_sd=1.0,
        ln_s_beta_mu=-3.0,
        ln_s_beta_sd=1.0,
        ln_s_gamma_mu=-3.0,
        ln_s_gamma_sd=1.0,
    )

    # ----- Sampler configuration ----- #
    cfg = SamplerConfig(
        n_iter=100,
        burn=10,
        thin=1,
        random_seed=42,
        progress=True,
        progress_every=10,
        print_dummies_every=0,  # set >0 if you want reconstructed seasonal dummies printed
        slice_w=3.0,
        slice_m=50,
    )

    # ----- Initial values ----- #
    sigma2_init = float(np.var(y) * 0.1) if len(y) > 1 else 1.0
    modes_tag = f"{level_mode}_{trend_mode}_{seasonal_mode}"

    # For s=4, default harmonics=None ⇒ K_full = (4-1)//2 = 1, Nyquist auto enabled.
    sampler = DLMGibbsHarmonic(
        y=y,
        period=period,
        harmonics=None,      # use full harmonic basis for s=4
        use_nyquist=None,    # let the sampler decide (True for s even & K large enough)
        level_mode=level_mode,
        trend_mode=trend_mode,
        seasonal_mode=seasonal_mode,
        # initial means/vars for dynamic level/trend
        m0_alpha_init=float(np.mean(y[: min(len(y), 8)])),
        P0_alpha_init=0.25,
        m0_beta_init=0.0,
        P0_beta_init=0.05,
        # seasonal initialisation: default zero means in harmonic space
        m0_cos_init=None,
        m0_sin_init=None,
        m0_nyq_init=0.0,
        P0_harm_init=0.25,
        # observation variance + process SD inits
        sigma2_init=sigma2_init,
        s_alpha_init=1e-1,
        s_beta_init=1e-1,
        s_gamma_init=1e-1,
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
    print(f"Running harmonic DLM for {series} with modes={modes_tag} ...")
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
            "label": f"{series}_seasonal",
            "index_type": type(idx).__name__,
            "index_values_preview": [str(ix) for ix in idx[: min(10, len(idx))]],
            "modes": {
                "level_mode": level_mode,
                "trend_mode": trend_mode,
                "seasonal_mode": seasonal_mode,
            },
            "description": (
                "Gaussian DLM with harmonic seasonality (cos/sin + Nyquist), "
                "log-normal priors on process SDs."
            ),
            "period": period,
            "date_tag": date_tag,
            "elapsed_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
        },
    )

    # ----- Quick fit plot ----- #
    mu_hat = post["mu"].mean(axis=0)
    t_index = idx

    plt.figure(figsize=(10, 4))
    plt.plot(t_index, y, label=f"{series}")
    plt.plot(t_index, mu_hat, "-.", label="μ̂_t")
    plt.title(
        f"{series}: harmonic DLM (L/T/S={modes_tag}, s={period}, "
        f"K={sampler.K}, nyq={sampler.use_nyq})"
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
    # Load TXm and TNm seasonal series
    tx = load_series(DATA_DIR / "TXm_seasonal.csv")
    tn = load_series(DATA_DIR / "TNm_seasonal.csv")

    print("TXm head:\n", tx.head(), "\n")
    print("TNm head:\n", tn.head(), "\n")

    # Base roots (now with 'results/' prefix)
    tx_root = _series_out_root("TXm")  # results/TX/TXm/Seasonal
    tn_root = _series_out_root("TNm")  # results/TN/TNm/Seasonal

    # Modes (can edit here if you want to experiment later)
    level_mode = "dynamic"
    trend_mode = "dynamic"
    seasonal_mode = "dynamic"

    run_one("TXm", tx, tx_root, level_mode, trend_mode, seasonal_mode)
    run_one("TNm", tn, tn_root, level_mode, trend_mode, seasonal_mode)

    print("Saved results under:")
    print(f"  {tx_root}/*")
    print(f"  {tn_root}/*")


if __name__ == "__main__":
    main()
