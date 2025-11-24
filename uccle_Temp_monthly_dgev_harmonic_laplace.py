# uccle_Temp_monthly_dgev_harmonic_laplace.py
# for s in TXx TXn TNx TNn; do python -u uccle_Temp_monthly_dgev_harmonic_laplace.py --series "$s"; done
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root on path
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(THIS_DIR, "..")
sys.path.append(PROJECT_ROOT)

from optimization.dgev_lognormal_harmonic_laplace import (  # type: ignore
    DGEVApproxGibbs,
    Priors,
    SamplerConfig,
)
from optimization.harmonic_helpers import (  # type: ignore
    center_and_report_dummies_full,
    dummies_full_to_harmonics_fft,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_csv_floats(val: str | None):
    """Parse '1,2,3' → [1.0, 2.0, 3.0] (or None if empty)."""
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    return [float(tok) for tok in val.split(",") if tok.strip()]


def _default_seasonal_dummies(period: int) -> np.ndarray:
    """
    Default smooth seasonal pattern over a full period, mean-centered.
    Used when no explicit seasonal prior is provided.
    """
    g = np.cos(2.0 * np.pi * np.arange(period) / period)
    g -= g.mean()
    return g.astype(float)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    return v in {"true", "1", "yes", "y"}


def _jsonify_dict(d: dict) -> dict:
    """Convert numpy types to JSON-safe Python objects."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _series_out_root(series: str) -> str:
    """
    Map series name to required output root:

      TXx → results/uccle/TX/TXx/Monthly/Laplace/
      TXn → results/uccle/TX/TXn/Monthly/Laplace/
      TNx → results/uccle/TN/TNx/Monthly/Laplace/
      TNn → results/uccle/TN/TNn/Monthly/Laplace/
    """
    base = "results/uccle"
    mapping = {
        "TXx": os.path.join(base, "TX", "TXx", "Monthly", "Laplace"),
        "TXn": os.path.join(base, "TX", "TXn", "Monthly", "Laplace"),
        "TNx": os.path.join(base, "TN", "TNx", "Monthly", "Laplace"),
        "TNn": os.path.join(base, "TN", "TNn", "Monthly", "Laplace"),
    }
    if series not in mapping:
        raise ValueError(f"Unknown series '{series}' for output mapping.")
    return mapping[series]


# ---------------------------------------------------------------------------
# Data loading (monthly extremes)
# ---------------------------------------------------------------------------
def load_monthlies(
    start_year: int = 1892,
    end_year: int = 2022,
    data_dir: str = "data",
    txx_file: str = "TXx.csv",
    txn_file: str = "TXn.csv",
    tnx_file: str = "TNx.csv",
    tnn_file: str = "TNn.csv",
) -> dict:
    """
    Load four *monthly* CSVs:

      TXx: monthly max of daily TX
      TXn: monthly min of daily TX
      TNx: monthly max of daily TN
      TNn: monthly min of daily TN

    The index is converted to PeriodIndex('M'). All series are aligned on the
    common monthly index and trimmed to [start_year, end_year].
    """

    def read_one(path: str) -> pd.Series:
        df = pd.read_csv(path, index_col=0)

        # Select numeric data column
        if df.shape[1] == 1:
            col = df.columns[0]
        else:
            col = next(
                (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])),
                None,
            )
        if col is None:
            raise ValueError(f"No numeric column found in {path}.")

        ts = pd.to_datetime(df.index)
        s = pd.Series(
            df[col].to_numpy(dtype=float),
            index=ts.to_period("M"),
        ).sort_index()

        s = s[(s.index.year >= start_year) & (s.index.year <= end_year)]
        return s

    series_dict = {
        "TXx": read_one(os.path.join(data_dir, txx_file)),
        "TXn": read_one(os.path.join(data_dir, txn_file)),
        "TNx": read_one(os.path.join(data_dir, tnx_file)),
        "TNn": read_one(os.path.join(data_dir, tnn_file)),
    }

    # Align to common PeriodIndex
    common_idx = None
    for s in series_dict.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    for k in series_dict:
        series_dict[k] = series_dict[k].reindex(common_idx).sort_index()

    return series_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Structural DGEV with harmonic seasonality (Laplace + FFBS) on Uccle "
            "MONTHLY TX/TN extremes (Jan–Dec)."
        )
    )

    # ---------------- Data & selection ----------------
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--txx-file", type=str, default="TXx.csv")
    parser.add_argument("--txn-file", type=str, default="TXn.csv")
    parser.add_argument("--tnx-file", type=str, default="TNx.csv")
    parser.add_argument("--tnn-file", type=str, default="TNn.csv")

    parser.add_argument("--start-year", type=int, default=1892)
    parser.add_argument("--end-year", type=int, default=2022)

    parser.add_argument(
        "--series",
        choices=["TXx", "TXn", "TNx", "TNn"],
        default="TNx",
        help="Which monthly extreme series to fit.",
    )

    # ---------------- Structural model modes ----------------
    parser.add_argument(
        "--level-mode",
        choices=["dynamic", "deterministic"],
        default="dynamic",
    )
    parser.add_argument(
        "--trend-mode",
        choices=["dynamic", "deterministic", "none"],
        default="dynamic",
    )
    parser.add_argument(
        "--seasonal-mode",
        choices=["dynamic", "deterministic", "none"],
        default="dynamic",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=12,
        help="Seasonal period (12 for monthly Jan–Dec).",
    )

    # ---------------- Initial structural values ----------------
    parser.add_argument("--level-init", type=float, default=0.0)
    parser.add_argument("--slope-init", type=float, default=0.0)

    # Observation parameter inits
    parser.add_argument("--init-sigma", type=float, default=1.0)
    parser.add_argument("--init-xi", type=float, default=-0.1)

    # ---------------- Priors ----------------
    # σ² ~ InvGamma(a_sigma, b_sigma)
    parser.add_argument("--prior-a-sigma", type=float, default=2.0)
    parser.add_argument("--prior-b-sigma", type=float, default=2.0)

    # ξ ~ Uniform[xi_lower, xi_upper]
    parser.add_argument("--prior-xi-lower", type=float, default=-0.5)
    parser.add_argument("--prior-xi-upper", type=float, default=0.5)

    # Deterministic / initial level & slope
    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s-level", type=float, default=10.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s-slope", type=float, default=10.0)

    # Seasonal prior in dummy space; converted to harmonics
    parser.add_argument(
        "--prior-m-season",
        type=str,
        default=None,
        help=(
            "Comma-separated values giving a seasonal dummy pattern. "
            "If length = period-1, last is implied by sum-to-zero; "
            "if length = period, used as-is. Converted to harmonic prior means."
        ),
    )
    parser.add_argument("--prior-s-season", type=float, default=5.0)

    # Log-normal priors on process SDs (ln s_α, ln s_β, ln s_γ)
    parser.add_argument("--prior-ln-s-alpha-m", type=float, default=-3)
    parser.add_argument("--prior-ln-s-alpha-sd", type=float, default=1)
    parser.add_argument("--prior-ln-s-beta-m", type=float, default=-4)
    parser.add_argument("--prior-ln-s-beta-sd", type=float, default=1)
    parser.add_argument("--prior-ln-s-gamma-m", type=float, default=-4.0)
    parser.add_argument("--prior-ln-s-gamma-sd", type=float, default=1)

    # ---------------- Sampler config ----------------
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--burn", type=int, default=10)
    parser.add_argument("--thin", type=int, default=1)

    # RW–MH step sizes
    parser.add_argument("--step-logsigma", type=float, default=0.1)
    parser.add_argument("--step-xi", type=float, default=0.1)
    parser.add_argument("--step-level", type=float, default=0.05)
    parser.add_argument("--step-slope", type=float, default=0.001)
    parser.add_argument("--step-season", type=float, default=0.05)

    # Slice sampler for ln s
    parser.add_argument("--slice-w", type=float, default=1.0)
    parser.add_argument("--slice-m", type=int, default=20)

    # Adaptive RW–MH
    parser.add_argument(
        "--adapt-steps",
        type=_str2bool,
        default=True,
        help="Adapt RW–MH step sizes during burn-in.",
    )
    parser.add_argument("--adapt-every", type=int, default=25)
    parser.add_argument("--adapt-until", choices=["burn", "all"], default="burn")
    parser.add_argument("--adapt-eta0", type=float, default=0.05)
    parser.add_argument("--adapt-decay", type=float, default=0.75)
    parser.add_argument("--adapt-target-1d", type=float, default=0.44)
    parser.add_argument("--step-min", type=float, default=1e-5)
    parser.add_argument("--step-max", type=float, default=1.0)

    # ---------------- Output & reproducibility ----------------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Override output directory. "
            "By default, uses results/uccle/TX/TXx/Monthly/Laplace/, "
            "results/uccle/TX/TXn/Monthly/Laplace/, "
            "results/uccle/TN/TNx/Monthly/Laplace/ or "
            "results/uccle/TN/TNn/Monthly/Laplace/ depending on series."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    parser.add_argument(
        "--progress",
        type=_str2bool,
        default=True,
        help="Print per-iteration progress info.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Progress line every k iterations (0 = auto ~2% of n_iter).",
    )
    parser.add_argument(
        "--print-dummies-every",
        type=int,
        default=0,
        help="Reconstruct & print full seasonal dummies every N iters (0=off).",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # ---------------- Load data ----------------
    all_series = load_monthlies(
        start_year=args.start_year,
        end_year=args.end_year,
        data_dir=args.data_dir,
        txx_file=args.txx_file,
        txn_file=args.txn_file,
        tnx_file=args.tnx_file,
        tnn_file=args.tnn_file,
    )
    s = all_series[args.series].dropna().sort_index()
    y = s.to_numpy(dtype=float)
    T = y.size

    if T < 5:
        raise ValueError(
            f"Not enough observations after filtering months/years; got T={T}."
        )

    # ---------------- Harmonic specification ----------------
    K_full = (args.period - 1) // 2  # max number of harmonics
    use_nyq = bool((args.period % 2 == 0) and (K_full > 0))

    # ---------------- Seasonal prior in harmonic form ----------------
    m_season_dummies = _parse_csv_floats(args.prior_m_season)

    if m_season_dummies is None and args.seasonal_mode != "none":
        # Use default smooth pattern if seasonality is present and no prior is given
        m_season_dummies = _default_seasonal_dummies(args.period).tolist()

    cos_prior = None
    sin_prior = None
    nyq_prior = 0.0

    if m_season_dummies is not None and args.seasonal_mode != "none":
        vals = list(m_season_dummies)
        if len(vals) == args.period - 1:
            vals.append(-sum(vals))
        elif len(vals) != args.period:
            raise ValueError(
                f"--prior-m-season must have length {args.period} "
                f"or {args.period - 1} (with last implied)."
            )

        full = np.asarray(vals, float)
        centered = center_and_report_dummies_full(full, tol=1e-12)

        if K_full > 0:
            cos_arr, sin_arr, nyq_val = dummies_full_to_harmonics_fft(
                centered,
                K=K_full,
                use_nyquist=use_nyq,
            )
            cos_prior = cos_arr.tolist()
            sin_prior = sin_arr.tolist()
            nyq_prior = 0.0 if nyq_val is None else float(nyq_val)

    # ---------------- Priors object ----------------
    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=float(args.prior_xi_lower),
        xi_upper=float(args.prior_xi_upper),
        # Level / slope
        m_m0_alpha=float(args.prior_m_level),
        s_m0_alpha=float(args.prior_s_level),
        m_m0_beta=float(args.prior_m_slope),
        s_m0_beta=float(args.prior_s_slope),
        # Harmonic means
        m_m0_cos=None if cos_prior is None else cos_prior,
        m_m0_sin=None if sin_prior is None else sin_prior,
        m_m0_nyq=float(nyq_prior),
        s_m0_harm=float(args.prior_s_season),
        # Log-normal on process SDs
        mu_log_s_alpha=float(args.prior_ln_s_alpha_m),
        sd_log_s_alpha=float(args.prior_ln_s_alpha_sd),
        mu_log_s_beta=float(args.prior_ln_s_beta_m),
        sd_log_s_beta=float(args.prior_ln_s_beta_sd),
        mu_log_s_gamma=float(args.prior_ln_s_gamma_m),
        sd_log_s_gamma=float(args.prior_ln_s_gamma_sd),
    )

    # If level is deterministic, center its prior around the data for better mixing
    if args.level_mode == "deterministic":
        priors.m_m0_alpha = float(np.median(y))
        priors.s_m0_alpha = max(2.0, 0.5 * y.std(ddof=1))

    # ---------------- Sampler configuration ----------------
    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=int(args.progress_every if args.progress_every > 0 else 0),
        step_logsigma=float(args.step_logsigma),
        step_xi=float(args.step_xi),
        step_level=float(args.step_level),
        step_slope=float(args.step_slope),
        step_season=float(args.step_season),
        adapt_steps=bool(args.adapt_steps),
        adapt_every=int(args.adapt_every),
        adapt_until=str(args.adapt_until),
        adapt_eta0=float(args.adapt_eta0),
        adapt_eta_decay=float(args.adapt_decay),
        adapt_target_1d=float(args.adapt_target_1d),
        step_min=float(args.step_min),
        step_max=float(args.step_max),
        slice_w=float(args.slice_w),
        slice_m=int(args.slice_m),
        print_dummies_every=int(args.print_dummies_every),
    )

    # ---------------- Initial state ----------------
    init_m0_level = float(args.level_init)
    init_P0_level = 0.2

    init_m0_trend = float(args.slope_init if args.trend_mode != "none" else 0.0)
    init_P0_trend = 0.05

    if cos_prior is not None:
        m0_cos_init = np.asarray(cos_prior, float)
        m0_sin_init = np.asarray(sin_prior, float)
        m0_nyq_init = float(nyq_prior)
    else:
        m0_cos_init = np.zeros(K_full, float)
        m0_sin_init = np.zeros(K_full, float)
        m0_nyq_init = 0.0
    P0_harm_init = 0.25

    # Initial σ and ξ
    if args.init_sigma is not None:
        init_sigma = float(args.init_sigma)
    else:
        init_sigma = float(max(1e-3, np.std(y, ddof=1)))

    xi_lb = float(args.prior_xi_lower)
    xi_ub = float(args.prior_xi_upper)
    if args.init_xi is not None:
        xi_raw = float(args.init_xi)
    else:
        xi_raw = 0.0
    init_xi = float(np.clip(xi_raw, xi_lb, xi_ub))

    # ---------------- Construct sampler ----------------
    sampler = DGEVApproxGibbs(
        y=y,
        period=int(args.period),
        harmonics=K_full,
        use_nyquist=use_nyq,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        # x0 priors (dynamic coords)
        m0_alpha_init=init_m0_level,
        P0_alpha_init=init_P0_level,
        m0_beta_init=init_m0_trend,
        P0_beta_init=init_P0_trend,
        m0_cos_init=m0_cos_init,
        m0_sin_init=m0_sin_init,
        m0_nyq_init=m0_nyq_init,
        P0_harm_init=P0_harm_init,
        # observation init
        sigma_init=init_sigma,
        xi_init=init_xi,
        # process SD init (small → smooth states)
        s_alpha_init=1e-2,
        s_beta_init=1e-3,
        s_gamma_init=1e-3,
        priors=priors,
        cfg=cfg,
    )

    # ---------------- Output directories ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes_tag = f"{args.level_mode}_{args.trend_mode}_{args.seasonal_mode}"
    tag = f"{args.series}_{modes_tag}"

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        series_root = _series_out_root(args.series)
        out_dir = os.path.join(series_root, f"{tag}_{timestamp}")

    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # ---------------- Run sampler ----------------
    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f} seconds")

    # ---------------- Save posterior + metadata ----------------
    date_tag = f"{int(args.start_year)}-{int(args.end_year)}"
    npz_filename = f"posterior_{args.series}_{date_tag}_{modes_tag}.npz"
    npz_path = os.path.join(out_dir, npz_filename)

    meta = {
        "series": args.series,
        "T": int(T),
        "period": int(args.period),
        "index_type": "M (monthly), Jan labeled by calendar year",
        "modes": {
            "level_mode": args.level_mode,
            "trend_mode": args.trend_mode,
            "seasonal_mode": args.seasonal_mode,
        },
        "cfg": _jsonify_dict(asdict(cfg)),
        "priors": _jsonify_dict(asdict(priors)),
        "elapsed_seconds": float(elapsed),
        "timestamp": datetime.now().isoformat(),
        "years": {"start": int(args.start_year), "end": int(args.end_year)},
        "tag": tag,
    }
    sampler.save_posterior(out_npz_path=npz_path, extra_meta=meta)

    # ---------------- Quick summaries ----------------
    if "sigma" in posterior and posterior["sigma"].size:
        print(f"Posterior mean sigma: {np.mean(posterior['sigma']):.3f}")
    if "xi" in posterior and posterior["xi"].size:
        print(f"Posterior mean xi:    {np.mean(posterior['xi']):.3f}")
    if "loglike" in posterior and posterior["loglike"].size:
        ll = posterior["loglike"]
        print(
            "log p(y|θ) (approx GEV): "
            f"mean={np.nanmean(ll):.3f}, "
            f"median={np.nanmedian(ll):.3f}, "
            f"best={np.nanmax(ll):.3f}"
        )
