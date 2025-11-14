# run_uccle_dummy_ncp.py
import os, math, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- import your new NCP dummy-rotation DLM ----
# Adjust module name/path to match where you saved the class you pasted above.
from optimization.dlm_lognormal_nc import (
    DLMDisturbanceNCP,
    Priors,
    SamplerConfig,
)

DATA_DIR = Path("data")


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


def run_one(label: str, y_ser: pd.Series, outdir: Path) -> None:
    """
    Run the disturbance-NCP dummy-rotation DLM on a single series.
    """
    y = y_ser.to_numpy(float)
    period = 4  # four seasons per year

    # ----- Priors (log-normal on process SDs + weak-normal on m0) -----
    pri = Priors(
        a_sigma=2.0,
        b_sigma=1.0,
        # m0 priors (leave quite weak)
        m_m0_alpha=0.0,
        s_m0_alpha=10.0,
        m_m0_beta=0.0,
        s_m0_beta=10.0,
        m_m0_gamma=[0.0] * (period - 1),
        s_m0_gamma=5.0,
        # P0 priors (Inv-Gamma)
        a_P0_alpha=5.0,
        b_P0_alpha=1.0,
        a_P0_beta=5.0,
        b_P0_beta=1.0,
        a_P0_gamma=5.0,
        b_P0_gamma=1.0,
        # log-normal priors for process SDs (roughly matching your old ln_s_* settings)
        mu_log_s_alpha=-5.0; sd_log_s_alpha=1  # level almost deterministic
        mu_log_s_beta =-5.0; sd_log_s_beta =1   # trend almost deterministic
        mu_log_s_gamma=-5.0; sd_log_s_gamma=1   # seasonal almost fixed

    )

    # ----- Sampler configuration -----
    cfg = SamplerConfig(
        n_iter=50000,
        burn=10000,
        thin=1,
        random_seed=42,
        progress=True,
        progress_every=10,
        slice_w=3.0,
        slice_m=50,
        sigma_update="eps",  # use ε-draws from DK smoother
    )

    # ----- Initial values -----
    sigma2_init = float(np.var(y) * 0.1) if len(y) > 1 else 1.0

    sampler = DLMDisturbanceNCP(
        y=y,
        period=period,
        level_mode="dynamic",
        trend_mode="dynamic",
        seasonal_mode="dynamic",
        # initial variance + process SDs
        sigma2_init=sigma2_init,
        s_alpha_init=1e-5,
        s_beta_init=1e-5,
        s_gamma_init=1e-5,
        # initial means for level / trend
        m0_alpha_init=float(np.mean(y[: min(len(y), 8)])),
        m0_beta_init=0.0,
        # initial P0 (roughly as in the __main__ of the new file)
        P0_alpha_init=0.25,
        P0_beta_init=0.05,
        P0_gamma_init=0.25,
        m0_gamma_init=None,  # let sampler start gamma at zeros
        priors=pri,
        cfg=cfg,
    )

    # ----- Run sampler -----
    post = sampler.run()

    # ----- Save posterior + metadata -----
    outdir.mkdir(parents=True, exist_ok=True)
    out_npz = outdir / "posterior.npz"
    sampler.save_posterior(
        out_npz_path=str(out_npz),
        extra_meta={
            "label": label,
            "index_type": type(y_ser.index).__name__,
            "index_values": [str(ix) for ix in y_ser.index[: len(y)]],
            "description": (
                "Disturbance-NCP DLM with dummy-rotation seasonality "
                "(period=4, dynamic level/trend/season)."
            ),
        },
    )

    # ----- Quick fit plot -----
    mu_hat = post["mu"].mean(axis=0)
    t_index = y_ser.index[: len(y)]

    plt.figure(figsize=(10, 4))
    plt.plot(t_index, y, label=f"{label}")
    plt.plot(t_index, mu_hat, "-.", label="μ̂_t")
    plt.title(f"{label}: DLM-NCP (dummy-rotation, dyn level/trend/season), s=4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit.png", dpi=150)
    plt.close()


def main() -> None:
    tx = load_series(DATA_DIR / "TXm_seasonal.csv")
    tn = load_series(DATA_DIR / "TNm_seasonal.csv")

    print("TX head:\n", tx.head(), "\n")
    print("TN head:\n", tn.head(), "\n")

    run_one("TXm_seasonal", tx, Path("results/uccle/TX/"))
    run_one("TNm_seasonal", tn, Path("results/uccle/TN/"))

    print("Saved results under results/uccle/{TX,TN}")

if __name__ == "__main__":
    main()
