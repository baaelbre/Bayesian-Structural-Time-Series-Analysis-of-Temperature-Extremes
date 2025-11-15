# run_uccle_harmonic.py  — monthly summaries (period = 12)
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- import the harmonic DLM with log-normal process SDs ----
# Adjust module name/path if needed.
from optimization.dlm_lognormal_harmonic import (
    DLMGibbsHarmonic,
    Priors,
    SamplerConfig,
)

DATA_DIR = Path("data")


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


def run_one(label: str, y_ser: pd.Series, outdir: Path) -> None:
    """
    Run the harmonic DLM (cos/sin + optional Nyquist) on a single *monthly* series.
    """
    y = y_ser.to_numpy(dtype=float)
    period = 12  # 12 months per year

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
        ln_s_alpha_mu=-2.0,
        ln_s_alpha_sd=1.0,
        ln_s_beta_mu=-5.0,
        ln_s_beta_sd=1.0,
        ln_s_gamma_mu=-5.0,
        ln_s_gamma_sd=1.0,
    )

    # ----- Sampler configuration ----- #
    cfg = SamplerConfig(
        n_iter=50_000,
        burn=10_000,
        thin=1,
        random_seed=42,
        progress=True,
        progress_every=10,
        print_dummies_every=0,  # set >0 if you want reconstructed monthly dummies printed
        slice_w=3.0,
        slice_m=50,
    )

    # ----- Initial values ----- #
    sigma2_init = float(np.var(y) * 0.1) if len(y) > 1 else 1.0

    # For period=12, harmonics=None ⇒ use the sampler’s full harmonic basis.
    sampler = DLMGibbsHarmonic(
        y=y,
        period=period,
        harmonics=None,          # use full harmonic basis for monthly cycle
        use_nyquist=None,        # let the sampler decide (True for even s when appropriate)
        level_mode="dynamic",
        trend_mode="dynamic",
        seasonal_mode="dynamic",
        # initial means/vars for dynamic level/trend
        m0_alpha_init=float(np.mean(y[: min(len(y), period)])),  # use roughly first year
        P0_alpha_init=0.25,
        m0_beta_init=0.0,
        P0_beta_init=0.05,
        # seasonal initialisation: defaults to zero means for harmonics
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

    # ----- Run sampler ----- #
    post = sampler.run()

    # ----- Save posterior + metadata ----- #
    outdir.mkdir(parents=True, exist_ok=True)
    out_npz = outdir / "posterior.npz"
    sampler.save_posterior(
        out_npz_path=str(out_npz),
        extra_meta={
            "label": label,
            "index_type": type(y_ser.index).__name__,
            "index_values": [str(ix) for ix in y_ser.index[: len(y)]],
            "description": (
                "Gaussian DLM with harmonic monthly seasonality (cos/sin + Nyquist), "
                "dynamic level/trend/season, log-normal priors on process SDs."
            ),
        },
    )

    # ----- Quick fit plot ----- #
    mu_hat = post["mu"].mean(axis=0)
    t_index = y_ser.index[: len(y)]

    plt.figure(figsize=(12, 4))
    plt.plot(t_index, y, label=f"{label}")
    plt.plot(t_index, mu_hat, "-.", label="μ̂_t")
    plt.title(
        f"{label}: harmonic DLM (dyn level/trend/season, "
        f"s={period}, K={sampler.K}, nyq={sampler.use_nyq})"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit.png", dpi=150)
    plt.close()


def main() -> None:
    # Monthly mean max / min temperatures
    tx = load_series(DATA_DIR / "TXm.csv")
    tn = load_series(DATA_DIR / "TNm.csv")

    print("TXm head:\n", tx.head(), "\n")
    print("TNm head:\n", tn.head(), "\n")

    run_one("TXm_monthly", tx, Path("results/uccle/TX_harm_monthly/"))
    run_one("TNm_monthly", tn, Path("results/uccle/TN_harm_monthly/"))

    print("Saved results under results/uccle/{TX_harm_monthly,TN_harm_monthly}")


if __name__ == "__main__":
    main()
