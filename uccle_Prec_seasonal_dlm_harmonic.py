# run_uccle_precip_harmonic_seasonal.py
import os
import math
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
    Load a univariate time series from CSV and trim to a multiple of 4
    (DJF/MAM/JJA/SON seasonal means).

    Expected format:
      - 'date' column (or similar) for the seasonal time index.
      - One numeric column with the series values (e.g. 'Prec', 'value', ...).
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
    if n <= 0:
        raise ValueError(f"Series in {csv_path} is shorter than one full seasonal cycle (4 points).")
    return ser.iloc[:n]


def run_one(label: str, y_ser: pd.Series, outdir: Path) -> None:
    """
    Run the harmonic DLM (cos/sin + optional Nyquist) on a single seasonal
    precipitation series (DJF/MAM/JJA/SON).
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
        n_iter=50000,
        burn=10000,
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

    # For s=4, default harmonics=None ⇒ K_full = (4-1)//2 = 1, Nyquist auto enabled.
    sampler = DLMGibbsHarmonic(
        y=y,
        period=period,
        harmonics=None,          # use full harmonic basis for s=4
        use_nyquist=None,        # let the sampler decide (True for s even & K large enough)
        level_mode="dynamic",
        trend_mode="dynamic",
        seasonal_mode="dynamic",
        # initial means/vars for dynamic level/trend
        m0_alpha_init=float(np.mean(y[: min(len(y), 8)])),
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
                "Gaussian DLM with harmonic DJF/MAM/JJA/SON seasonality "
                "(cos/sin + Nyquist), dynamic level/trend/season, "
                "log-normal priors on process SDs, applied to seasonal mean precipitation."
            ),
        },
    )

    # ----- Quick fit plot ----- #
    mu_hat = post["mu"].mean(axis=0)
    t_index = y_ser.index[: len(y)]

    plt.figure(figsize=(10, 4))
    plt.plot(t_index, y, label=f"{label}")
    plt.plot(t_index, mu_hat, "-.", label="μ̂_t")
    plt.title(
        f"{label}: harmonic DLM (dyn level/trend/season, s={period}, "
        f"K={sampler.K}, nyq={sampler.use_nyq})"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit.png", dpi=150)
    plt.close()


def main() -> None:
    # Seasonal mean precipitation (DJF/MAM/JJA/SON)
    precm_seasonal = load_series(DATA_DIR / "Precm_seasonal.csv")

    print("Precm_seasonal head:\n", precm_seasonal.head(), "\n")

    run_one("Precm_seasonal", precm_seasonal, Path("results/uccle/Precm_harm_seasonal/"))

    print("Saved results under results/uccle/Precm_harm_seasonal")


if __name__ == "__main__":
    main()
