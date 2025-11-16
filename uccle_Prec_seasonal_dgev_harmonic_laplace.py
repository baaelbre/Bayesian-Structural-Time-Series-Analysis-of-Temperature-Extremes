# uccle_Precx_seasonal_dgev_harmonic_laplace.py
# Example:
#   python -u uccle_Precx_seasonal_dgev_harmonic_laplace.py

import os, sys, time, json
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

# Import from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.dgev_lognormal_harmonic_laplace import (
    DGEVApproxGibbs,
    Priors,
    SamplerConfig,
)
from optimization.harmonic_helpers import (
    center_and_report_dummies_full,
    dummies_full_to_harmonics_fft,
)
from simulator.dgev_plotter import DGEVPlotter  # optional


# =========================
# I/O + small helpers
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _series_out_root() -> str:
    """
    Output root for seasonal precipitation extremes:

      Precx → results/uccle/Prec/Precx/Seasonal/Laplace/
    """
    base = "results/uccle"
    return os.path.join(base, "Prec", "Precx", "Seasonal", "Laplace")


def load_precx_seasonal(
    start_year: int = 1892,
    end_year: int = 2022,
    data_dir: str = "data",
    precx_file: str = "Precx_seasonal.csv",
) -> pd.Series:
    """
    Read a SEASONAL CSV for max precipitation:

      Precx_seasonal.csv: seasonal max of daily precipitation,
      1 row per meteorological season (DJF/MAM/JJA/SON).

    The CSV is expected to have:
      - a date-like first column (any date within the season, e.g. date of max),
      - one numeric column with the precipitation maximum (e.g. 'Prec').

    We convert the index to PeriodIndex('Q-FEB'):
      Q1 = DJF (Dec–Jan–Feb, labeled by Jan/Feb year),
      Q2 = MAM, Q3 = JJA, Q4 = SON.
    """

    path = os.path.join(data_dir, precx_file)
    df = pd.read_csv(path, index_col=0)

    # pick numeric column
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
        index=ts.to_period("Q-FEB"),
    ).sort_index()

    s = s[(s.index.year >= start_year) & (s.index.year <= end_year)]
    return s


def _parse_csv_floats(val: str | None):
    if val is None:
        return None
    val = val.strip()
    return None if not val else [float(tok) for tok in val.split(",") if tok.strip() != ""]


def _default_seasonal_dummies(period: int) -> np.ndarray:
    """
    Default smooth seasonal pattern as full-length dummies that sum to zero.
    (Cosine over the period, mean-centered.)
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
    """
    Make a dict JSON-safe: convert any np.ndarray to list, np.generic to Python scalars.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Structural DGEV with harmonic seasonality (Laplace + FFBS) on Uccle "
            "SEASONAL precipitation extremes (Precx_seasonal, meteorological seasons)."
        )
    )

    # ---------------- Data & selection ----------------
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--precx-file", type=str, default="Precx_seasonal.csv")
    parser.add_argument("--start-year", type=int, default=1892, help="Season label year (e.g., DJF 1892).")
    parser.add_argument("--end-year", type=int, default=2022, help="Season label year (e.g., DJF 2022).")

    # ---------------- Structural model modes ----------------
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
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
        default=4,
        help="Seasonal period (meteorological seasons = 4).",
    )

    # ---------------- Initial structural values ----------------
    parser.add_argument("--level-init", type=float, default=0.0, help="Initial level (for m0 or deterministic).")
    parser.add_argument("--slope-init", type=float, default=0.0, help="Initial slope (for m0 or deterministic).")

    # Observation parameter inits
    parser.add_argument(
        "--init-sigma",
        type=float,
        default=1,
        help="Initial σ for GEV (default: sample sd of y).",
    )
    parser.add_argument(
        "--init-xi",
        type=float,
        default=0.1,
        help="Initial ξ for GEV (default: 0.0, clipped to [xi_lower, xi_upper]).",
    )

    # ---------------- Priors (match DGEVApproxGibbs Priors) ----------------
    # σ² ~ InvGamma(a_sigma, b_sigma)
    parser.add_argument("--prior-a-sigma", type=float, default=2.0)
    parser.add_argument("--prior-b-sigma", type=float, default=2.0)

    # ξ ~ Uniform[xi_lower, xi_upper]
    parser.add_argument("--prior-xi-lower", type=float, default=-0.5)
    parser.add_argument("--prior-xi-upper", type=float, default=0.5)

    # Deterministic / initial level & slope (Gaussian)
    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s-level", type=float, default=10.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s-slope", type=float, default=10.0)

    # Seasonal prior: full dummy pattern (length = period or period-1; last implied)
    parser.add_argument(
        "--prior-m-season",
        type=str,
        default=None,
        help=(
            "Comma-separated seasonal dummy pattern. "
            "If length = period-1, last entry is implied by sum-to-zero; "
            "if length = period, used as-is. Converted to harmonic prior means."
        ),
    )
    parser.add_argument("--prior-s-season", type=float, default=5.0)

    # Log-normal priors for process SDs ln s_α, ln s_β, ln s_γ
    parser.add_argument("--prior-ln-s-alpha-m", type=float, default=-3)
    parser.add_argument("--prior-ln-s-alpha-sd", type=float, default=1)
    parser.add_argument("--prior-ln-s-beta-m", type=float, default=-5)
    parser.add_argument("--prior-ln-s-beta-sd", type=float, default=1)
    parser.add_argument("--prior-ln-s-gamma-m", type=float, default=-5)
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
        help="Whether to adapt RW–MH step sizes (True/False).",
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
            "By default: results/uccle/Prec/Precx/Seasonal/Laplace/"
            "with subfolder Precx_<modes>_<timestamp>/"
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument(
        "--progress",
        type=_str2bool,
        default=True,
        help="Print per-iteration progress info (True/False).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Progress line every k iterations (0 = auto ≈ 2% of n_iter).",
    )
    parser.add_argument(
        "--print-dummies-every",
        type=int,
        default=0,
        help="Reconstruct & print full seasonal dummies every N iters (0=off).",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # ---------------- Load SEASONAL Precx series ----------------
    s = (
        load_precx_seasonal(
            start_year=args.start_year,
            end_year=args.end_year,
            data_dir=args.data_dir,
            precx_file=args.precx_file,
        )
        .dropna()
        .sort_index()
    )  # PeriodIndex('Q-FEB')

    y = s.to_numpy(dtype=float)
    T = y.size
    if T < 5:
        raise ValueError(f"Not enough observations after filtering by seasons/years; got T={T}.")

    # ---------------- Harmonic spec ----------------
    K_full = (args.period - 1) // 2  # maximum number of harmonics
    use_nyq = bool((args.period % 2 == 0) and (K_full > 0))

    # ---------------- Seasonal prior in harmonic form ----------------
    m_season_dummies = _parse_csv_floats(args.prior_m_season)

    # If we have seasonality and no explicit prior, use a smooth cos pattern
    if m_season_dummies is None and args.seasonal_mode != "none":
        m_season_dummies = _default_seasonal_dummies(args.period).tolist()

    cos_prior = None
    sin_prior = None
    nyq_prior = 0.0

    if m_season_dummies is not None and args.seasonal_mode != "none":
        vals = list(m_season_dummies)
        if len(vals) == args.period - 1:
            last = -sum(vals)
            vals.append(last)
        elif len(vals) != args.period:
            raise ValueError(
                f"--prior-m-season must have length {args.period} "
                f"(or {args.period - 1} with last implied)."
            )
        full = np.asarray(vals, float)
        centered = center_and_report_dummies_full(full, tol=1e-12)

        if K_full > 0:
            cos_prior_arr, sin_prior_arr, nyq_val = dummies_full_to_harmonics_fft(
                centered, K=K_full, use_nyquist=use_nyq
            )
            cos_prior = cos_prior_arr.tolist()
            sin_prior = sin_prior_arr.tolist()
            nyq_prior = 0.0 if nyq_val is None else float(nyq_val)

    # ---------------- Priors for DGEVApproxGibbs ----------------
    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=float(args.prior_xi_lower),
        xi_upper=float(args.prior_xi_upper),
        # level / trend priors
        m_m0_alpha=float(args.prior_m_level),
        s_m0_alpha=float(args.prior_s_level),
        m_m0_beta=float(args.prior_m_slope),
        s_m0_beta=float(args.prior_s_slope),
        # harmonic priors (lists or None → handled inside sampler)
        m_m0_cos=None if cos_prior is None else cos_prior,
        m_m0_sin=None if sin_prior is None else sin_prior,
        m_m0_nyq=float(nyq_prior),
        s_m0_harm=float(args.prior_s_season),
        # log-normal priors on process SDs ln s
        mu_log_s_alpha=float(args.prior_ln_s_alpha_m),
        sd_log_s_alpha=float(args.prior_ln_s_alpha_sd),
        mu_log_s_beta=float(args.prior_ln_s_beta_m),
        sd_log_s_beta=float(args.prior_ln_s_beta_sd),
        mu_log_s_gamma=float(args.prior_ln_s_gamma_m),
        sd_log_s_gamma=float(args.prior_ln_s_gamma_sd),
    )

    # For deterministic level: re-center its prior around the data (helps mixing)
    if args.level_mode == "deterministic":
        priors.m_m0_alpha = float(np.median(y))
        priors.s_m0_alpha = max(2.0, 0.5 * y.std(ddof=1))

    # ---------------- Sampler config ----------------
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

    # ---------------- Initial values for structural state ----------------
    init_m0_level = float(args.level_init)
    init_P0_level = 0.2

    init_m0_trend = float(args.slope_init if args.trend_mode != "none" else 0.0)
    init_P0_trend = 0.05

    # Initial harmonic seasonal m0
    if cos_prior is not None:
        m0_cos_init = np.asarray(cos_prior, float)
        m0_sin_init = np.asarray(sin_prior, float)
        m0_nyq_init = float(nyq_prior)
    else:
        m0_cos_init = np.zeros(K_full, float)
        m0_sin_init = np.zeros(K_full, float)
        m0_nyq_init = 0.0
    P0_harm_init = 0.25

    # ---------------- Initial σ and ξ ----------------
    if args.init_sigma is not None:
        init_sigma = float(args.init_sigma)
    else:
        init_sigma = float(max(1e-3, np.std(y, ddof=1)))

    xi_lb = float(args.prior_xi_lower)
    xi_ub = float(args.prior_xi_upper)
    if args.init_xi is not None:
        init_xi_raw = float(args.init_xi)
    else:
        init_xi_raw = 0.0
    init_xi = float(np.clip(init_xi_raw, xi_lb, xi_ub))

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
    series = "Precx"
    tag = f"{series}_{modes_tag}"

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        series_root = _series_out_root()
        out_dir = os.path.join(series_root, f"{tag}_{timestamp}")

    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # ---------------- Run sampler ----------------
    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    # ---------------- Save posterior + metadata ----------------
    date_tag = f"{int(args.start_year)}-{int(args.end_year)}"
    npz_filename = f"posterior_{series}_{date_tag}_{modes_tag}.npz"
    npz_path = os.path.join(out_dir, npz_filename)

    extra_meta = {
        "series": "Precx_seasonal",
        "T": int(T),
        "period": int(args.period),
        "index_type": "Q-FEB (DJF/MAM/JJA/SON), DJF labeled by Jan/Feb year",
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
    sampler.save_posterior(out_npz_path=npz_path, extra_meta=extra_meta)

    # ---------------- Quick summaries ----------------
    if "sigma" in posterior and posterior["sigma"].size:
        print(f"Posterior mean sigma: {np.mean(posterior['sigma']):.3f}")
    if "xi" in posterior and posterior["xi"].size:
        print(f"Posterior mean xi:    {np.mean(posterior['xi']):.3f}")
    if "loglike" in posterior and posterior["loglike"].size:
        le = posterior["loglike"]
        print(
            f"log p(y|θ) (approx GEV): mean={np.nanmean(le):.3f}, "
            f"median={np.nanmedian(le):.3f}, best={np.nanmax(le):.3f}"
        )

    print("Saved results under:")
    print(f"  {out_dir}/*")
