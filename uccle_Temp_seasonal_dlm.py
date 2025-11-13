# run_uccle_harm.py
import os, math, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# ---- import your class ----
from optimization.dlm_lognormal_harmonic import DLMGibbsHarmonic, Priors, SamplerConfig  # adjust module name

DATA_DIR = Path("data")

def load_series(csv_path: Path):
    df = pd.read_csv(csv_path)
    # pick first numeric col as the series if 'value' not present
    if "value" in df.columns:
        s = pd.to_numeric(df["value"], errors="coerce")
    else:
        # find a numeric column
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            for c in df.columns[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        s = df[numcols[0]]
    # try to parse a date/time column if present, otherwise integer index
    idx = None
    for cand in ["date","time","Date","Time","year","Year"]:
        if cand in df.columns:
            idx = pd.to_datetime(df[cand], errors="coerce")
            break
    if idx is None:
        idx = pd.RangeIndex(len(s), name="t")
    ser = pd.Series(s.astype(float).to_numpy(), index=idx, name=csv_path.name).dropna()
    # trim to multiple of 4 (seasonal means)
    n = len(ser) - (len(ser) % 4)
    return ser.iloc[:n]

def run_one(label: str, y_ser: pd.Series, outdir: Path):
    y = y_ser.to_numpy(float)
    period = 4
    pri = Priors(
        a_sigma=2.0, b_sigma=1.0,
        ln_s_alpha_mu=-3, ln_s_alpha_sd=3.0,
        ln_s_beta_mu=-6,  ln_s_beta_sd=3.0,
        ln_s_gamma_mu=-6, ln_s_gamma_sd=3.0,
        s_m0_harm=2.0,
    )
    cfg = SamplerConfig(
        n_iter=50000, burn=10000, thin=1, random_seed=42,
        progress=True, progress_every=10, slice_w=3.0, slice_m=50,
        print_dummies_every=0
    )
    sampler = DLMGibbsHarmonic(
        y=y, period=period, harmonics=None, use_nyquist=None,
        level_mode="dynamic", trend_mode="dynamic", seasonal_mode="dynamic",
        sigma2_init=np.var(y)*0.1,
        s_alpha_init=1e-3, s_beta_init=1e-3, s_gamma_init=1e-3,
        m0_alpha_init=float(np.mean(y[:8])),
        priors=pri, cfg=cfg,
    )
    post = sampler.run()
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "posterior.npz", **post, index=np.arange(len(y)))
    # quick plot
    mu_hat = post["mu"].mean(axis=0)
    plt.figure(figsize=(10,4))
    plt.plot(y_ser.index[:len(y)], y, label=f"{label}")
    plt.plot(y_ser.index[:len(y)], mu_hat, "-.", label="μ̂_t")
    plt.title(f"{label}: harmonic DLM (dyn level/trend/season), s=4")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "fit.png", dpi=150); plt.close()

def main():
    tx = load_series(DATA_DIR / "TXm_seasonal.csv")
    tn = load_series(DATA_DIR / "TNm_seasonal.csv")
    print("TX head:\n", tx.head(), "\n")
    print("TN head:\n", tn.head(), "\n")

    run_one("TXm_seasonal", tx, Path("results/uccle/TX"))
    run_one("TNm_seasonal", tn, Path("results/uccle/TN"))
    print("Saved results under results/uccle/{TX,TN}")

if __name__ == "__main__":
    main()
