# uccle_Temp_monthly_dgev_dummies_laplace_ncp_lasso.py
# ---------------------------------------------------
# Run monthly Uccle extremes (TXx, TXn, TNx, TNn) with:
#   DGEV Laplace (pseudo-obs) + NCP FFBS + FS regression update
#   + hierarchical Bayesian lasso on signed process SDs.
#
# Example:
#   for s in TXx TXn TNx TNn; do python -u uccle_Temp_monthly_dgev_dummies_laplace_ncp_lasso.py --series "$s"; done

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Import sampler (robust import style)
# ---------------------------------------------------------------------
from optimization.dgev_laplace_2 import DGEVLaplaceNCP, Priors, SamplerConfig  # type: ignore

SERIES_FILES_DEFAULT: Dict[str, str] = {
    "TXx": "TXx.csv",
    "TXn": "TXn.csv",
    "TNx": "TNx.csv",
    "TNn": "TNn.csv",
}

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _series_out_root(series: str) -> Path:
    """
    Map series name to required output root:

      TXx → results/uccle/TX/TXx/Monthly/Laplace/
      TXn → results/uccle/TX/TXn/Monthly/Laplace/
      TNx → results/uccle/TN/TNx/Monthly/Laplace/
      TNn → results/uccle/TN/TNn/Monthly/Laplace/
    """
    base = Path("results") / "uccle"
    mapping = {
        "TXx": base / "TX" / "TXx" / "Monthly" / "Laplace",
        "TXn": base / "TX" / "TXn" / "Monthly" / "Laplace",
        "TNx": base / "TN" / "TNx" / "Monthly" / "Laplace",
        "TNn": base / "TN" / "TNn" / "Monthly" / "Laplace",
    }
    if series not in mapping:
        raise ValueError(f"Unknown series '{series}' for output mapping.")
    return mapping[series]


def _date_tag_from_index(idx: pd.Index) -> str:
    if len(idx) == 0:
        return "NA-NA"
    if isinstance(idx, pd.PeriodIndex):
        start = str(idx[0])
        end = str(idx[-1])
        return f"{start}-{end}"
    if isinstance(idx, pd.DatetimeIndex):
        return f"{idx[0].strftime('%Y-%m-%d')}-{idx[-1].strftime('%Y-%m-%d')}"
    return f"{idx[0]}-{idx[-1]}"


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "":
        return None
    return [float(tok) for tok in ss.split(",") if tok.strip()]


def _default_seasonal_pattern_full(period: int) -> np.ndarray:
    """
    Default smooth seasonal pattern (length=period), mean-centered to sum-to-zero.
    """
    p = int(period)
    g = np.cos(2.0 * np.pi * np.arange(p) / p)
    g = g - g.mean()
    # enforce exact sum-to-zero numerically
    g = g - (g.sum() / p)
    return g.astype(float)


def _to_gamma0_pminus1(vals: List[float], period: int) -> np.ndarray:
    """
    Convert a user-provided seasonal pattern to gamma0 of length (p-1)
    for the sum-to-zero parametrisation:
      seasonal_full = [gamma0...,  -sum(gamma0)]
    Accepted input lengths:
      - p-1: interpreted directly as gamma0 (newest-first)
      - p  : mean-center to sum-to-zero, then take first p-1 as gamma0
    """
    p = int(period)
    arr = np.asarray(vals, float).ravel()
    if arr.size == p - 1:
        return arr.copy()
    if arr.size == p:
        arr = arr - arr.mean()
        arr = arr - (arr.sum() / p)
        return arr[: p - 1].copy()
    raise ValueError(f"Season pattern must have length {p-1} or {p}, got {arr.size}.")


def load_monthly_series(
    csv_path: Path,
    *,
    start_year: int,
    end_year: int,
    force_start_january: bool = True,
    trim_full_years: bool = True,
) -> pd.Series:
    """
    Load a *monthly* series from CSV into a pandas Series with PeriodIndex(freq='M').

    Expected formats:
      - Index column is date-like (common for your Uccle CSVs).
      - Value column: if 'value' exists use it, else first numeric column.
    """
    df = pd.read_csv(csv_path, index_col=0)

    # values
    if "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce")
    else:
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            # try coercion once
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            raise ValueError(f"No numeric column found in {csv_path}")
        vals = pd.to_numeric(df[numcols[0]], errors="coerce")

    # index -> PeriodIndex('M')
    dt = pd.to_datetime(df.index, errors="coerce")
    ok = dt.notna() & vals.notna()
    dt = dt[ok]
    vals = vals[ok]
    idx = dt.to_period("M")

    ser = pd.Series(np.asarray(vals, float), index=idx, name=csv_path.stem).sort_index()
    ser = ser[(ser.index.year >= int(start_year)) & (ser.index.year <= int(end_year))]

    if force_start_january and len(ser) > 0:
        m0 = int(ser.index[0].month)
        if m0 != 1:
            mask = (ser.index.month == 1)
            if mask.any():
                first_jan_pos = int(np.argmax(mask.to_numpy()))
                ser = ser.iloc[first_jan_pos:]

    if trim_full_years and len(ser) >= 12:
        n = len(ser) - (len(ser) % 12)
        ser = ser.iloc[:n]

    if len(ser) < 20:
        raise ValueError(f"Series {csv_path.name} too short after filtering (T={len(ser)}).")

    return ser


# =====================================================================
# Core runner (mirrors your DLM runner)
# =====================================================================
def run_one(
    *,
    series: str,
    y_ser: pd.Series,
    out_root: Path,
    period: int,
    priors: Priors,
    cfg: SamplerConfig,
    # sampler initial values
    alpha0_init: float,
    beta0_init: float,
    gamma0_init: Optional[np.ndarray],
    sigma_init: float,
    xi_init: float,
    s_alpha_init: float,
    s_beta_init: float,
    s_gamma_init: float,
    # knobs (kept aligned with dgev_laplace.py)
    ffbs_C0_scale: float,
    ffbs_C0_A: float,
    sigma2_eff: float,
    # plotting
    plot: bool,
) -> None:
    y = y_ser.to_numpy(dtype=float)
    idx = y_ser.index

    sampler = DGEVLaplaceNCP(
        y=y,
        period=int(period),
        alpha0=float(alpha0_init),
        beta0=float(beta0_init),
        gamma0=None if gamma0_init is None else np.asarray(gamma0_init, float),
        sigma_init=float(sigma_init),
        xi_init=float(xi_init),
        s_alpha_init=float(s_alpha_init),
        s_beta_init=float(s_beta_init),
        s_gamma_init=float(s_gamma_init),
        priors=priors,
        cfg=cfg,
        ffbs_C0_scale=float(ffbs_C0_scale),
        ffbs_C0_A=float(ffbs_C0_A),
        sigma2_eff=float(sigma2_eff),
    )

    date_tag = _date_tag_from_index(idx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes_tag = "dynamic_dynamic_dynamic"  # fixed by this sampler’s structural setup
    tag = f"{series}_{modes_tag}"

    outdir = out_root / f"{tag}_{date_tag}_{timestamp}"
    figdir = outdir / "figures"
    _ensure_dir(outdir)
    _ensure_dir(figdir)

    print(f"Running Uccle monthly DGEV Laplace (NCP + lasso) for {series} ...")
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"{series}: run time {elapsed:.2f} seconds")

    # save posterior
    out_npz = outdir / f"posterior_{series}_{date_tag}_{modes_tag}.npz"
    sampler.save_posterior(
        out_npz_path=str(out_npz),
        extra_meta={
            "series": series,
            "label": f"{series}_monthly",
            "period": int(period),
            "date_tag": date_tag,
            "elapsed_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
            "index_type": type(idx).__name__,
            "index_values_preview": [str(ix) for ix in idx[: min(10, len(idx))]],
            "model": "DGEV_LAPLACE_NCP_LASSO_DUMMIES",
            "cfg": asdict(cfg),
            "priors": asdict(priors),
            "knobs": {
                "ffbs_C0_scale": float(ffbs_C0_scale),
                "ffbs_C0_A": float(ffbs_C0_A),
                "sigma2_eff": float(sigma2_eff),
            },
        },
    )

    # quick summaries
    if "sigma" in post:
        print(f"Posterior mean sigma: {float(np.mean(post['sigma'])):.4f}")
    if "xi" in post:
        print(f"Posterior mean xi:    {float(np.mean(post['xi'])):.4f}")
    for k in ["alpha", "beta", "gamma"]:
        kk = f"Q_{k}"
        if kk in post:
            mQ = float(np.mean(post[kk]))
            print(f"Posterior mean {kk}: {mQ:.4g} (sqrt≈{math.sqrt(max(mQ, 0.0)):.4g})")
    if "lambda2" in post:
        print(f"Posterior mean lambda2: {float(np.mean(post['lambda2'])):.4g}")
    print(f"Saved posterior to {out_npz}")

    # plot fit (post mean + 90% band for mu_t)
    if plot:
        mu_draws = post["mu"]
        mu_hat = mu_draws.mean(axis=0)
        lo, hi = np.quantile(mu_draws, [0.05, 0.95], axis=0)

        # matplotlib is happier with timestamps than PeriodIndex
        x = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else idx

        plt.figure(figsize=(12, 4))
        plt.plot(x, y, lw=1, label=f"{series}")
        plt.plot(x, mu_hat, "-.", lw=1.5, label="μ̂_t (post mean)")
        plt.fill_between(x, lo, hi, alpha=0.2, label="90% CI (μ_t)")
        plt.title(f"{series}: monthly DGEV Laplace (NCP + Bayesian lasso) — period={period}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fit_path = figdir / f"fit_mu_{series}_{date_tag}_{modes_tag}.png"
        plt.savefig(fit_path, dpi=180)
        plt.close()

        print(f"Saved fit plot to {fit_path}\n")


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Uccle MONTHLY TX/TN extremes: DGEV Laplace + FFBS in NCP with monthly seasonal dummies "
            "and hierarchical Bayesian lasso prior on signed process SDs."
        )
    )

    # --- data ---
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--start-year", type=int, default=1892)
    p.add_argument("--end-year", type=int, default=2022)
    p.add_argument("--series", choices=["TXx", "TXn", "TNx", "TNn"], default="TNx")
    p.add_argument("--period", type=int, default=12)

    # --- files (overrideable) ---
    p.add_argument("--txx-file", type=str, default=SERIES_FILES_DEFAULT["TXx"])
    p.add_argument("--txn-file", type=str, default=SERIES_FILES_DEFAULT["TXn"])
    p.add_argument("--tnx-file", type=str, default=SERIES_FILES_DEFAULT["TNx"])
    p.add_argument("--tnn-file", type=str, default=SERIES_FILES_DEFAULT["TNn"])

    # --- initial values ---
    p.add_argument("--alpha0-init", type=float, default=float("nan"))  # if nan -> median(y)
    p.add_argument("--beta0-init", type=float, default=0.0)
    p.add_argument("--gamma0-init", type=str, default=None, help="CSV length p-1 or p.")
    p.add_argument("--sigma-init", type=float, default=float("nan"))   # if nan -> std(y)
    p.add_argument("--xi-init", type=float, default=-0.1)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-3)

    # --- priors (obs) ---
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=2.0)
    p.add_argument("--prior-xi-lower", type=float, default=-0.5)
    p.add_argument("--prior-xi-upper", type=float, default=0.5)

    # --- priors (baselines) ---
    p.add_argument("--prior-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-P0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-P0-beta", type=float, default=10.0)
    p.add_argument(
        "--prior-m0-gamma",
        type=str,
        default=None,
        help="CSV length p-1 (gamma0) or p (full seasonal pattern). If omitted: cosine pattern.",
    )
    p.add_argument("--prior-P0-gamma", type=float, default=5.0)

    # --- priors (lasso hyperprior on lambda^2) ---
    p.add_argument("--prior-a-lambda", type=float, default=0.001)
    p.add_argument("--prior-b-lambda", type=float, default=0.001)

    # --- sampler config ---
    p.add_argument("--n-iter", type=int, default=8000)
    p.add_argument("--burn", type=int, default=4000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", type=_str2bool, default=True)
    p.add_argument("--progress-every", type=int, default=10, help="0=auto (~2%).")

    # --- knobs (match dgev_laplace.py) ---
    p.add_argument("--ffbs-C0-scale", type=float, default=1e-6)
    p.add_argument("--ffbs-C0-A", type=float, default=1e-6)
    p.add_argument("--sigma2-eff", type=float, default=1.0)

    # --- output / plotting ---
    p.add_argument("--out-dir", type=str, default=None, help="Override output run directory root (folder).")
    p.add_argument("--plot", type=_str2bool, default=True)

    args = p.parse_args()
    rng = np.random.default_rng(int(args.seed))

    series = str(args.series)
    period = int(args.period)
    K = period - 1

    # choose file for selected series
    file_map = {
        "TXx": args.txx_file,
        "TXn": args.txn_file,
        "TNx": args.tnx_file,
        "TNn": args.tnn_file,
    }
    csv_path = Path(args.data_dir) / str(file_map[series])

    # load data
    y_ser = load_monthly_series(
        csv_path,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        force_start_january=True,
        trim_full_years=True,
    )

    # --- prior mean for gamma0 (length p-1) ---
    m0_gamma_vals = _parse_csv_floats(args.prior_m0_gamma)
    if m0_gamma_vals is None:
        full = _default_seasonal_pattern_full(period)
        m0_gamma = full[:K].copy()
    else:
        m0_gamma = _to_gamma0_pminus1(m0_gamma_vals, period)

    # --- init gamma0 ---
    gamma0_init_vals = _parse_csv_floats(args.gamma0_init)
    gamma0_init = None if gamma0_init_vals is None else _to_gamma0_pminus1(gamma0_init_vals, period)

    # --- data-driven alpha0/sigma init ---
    y = y_ser.to_numpy(float)

    alpha0_init = float(args.alpha0_init)
    if not np.isfinite(alpha0_init):
        alpha0_init = float(np.median(y))

    sigma_init = float(args.sigma_init)
    if (not np.isfinite(sigma_init)) or sigma_init <= 0.0:
        sigma_init = float(max(1e-3, np.std(y, ddof=1)))

    # clip xi init into prior support
    xi_lb = float(args.prior_xi_lower)
    xi_ub = float(args.prior_xi_upper)
    xi_init = float(np.clip(float(args.xi_init), xi_lb, xi_ub))

    # --- priors + cfg ---
    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=xi_lb,
        xi_upper=xi_ub,
        m0_alpha=float(args.prior_m0_alpha),
        P0_alpha=float(args.prior_P0_alpha),
        m0_beta=float(args.prior_m0_beta),
        P0_beta=float(args.prior_P0_beta),
        m0_gamma=m0_gamma.tolist(),
        P0_gamma=float(args.prior_P0_gamma),
        a_lambda=float(args.prior_a_lambda),
        b_lambda=float(args.prior_b_lambda),
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
    )

    # --- output root ---
    if args.out_dir is not None:
        out_root = Path(args.out_dir)
    else:
        out_root = _series_out_root(series)

    _ensure_dir(out_root)

    # --- run ---
    run_one(
        series=series,
        y_ser=y_ser,
        out_root=out_root,
        period=period,
        priors=priors,
        cfg=cfg,
        alpha0_init=alpha0_init,
        beta0_init=float(args.beta0_init),
        gamma0_init=gamma0_init,
        sigma_init=sigma_init,
        xi_init=xi_init,
        s_alpha_init=float(args.s_alpha_init),
        s_beta_init=float(args.s_beta_init),
        s_gamma_init=float(args.s_gamma_init),
        ffbs_C0_scale=float(args.ffbs_C0_scale),
        ffbs_C0_A=float(args.ffbs_C0_A),
        sigma2_eff=float(args.sigma2_eff),
        plot=bool(args.plot),
    )


if __name__ == "__main__":
    main()
