#!/usr/bin/env python3
# compare_dgev_variants.py
# Run + compare 4 DGEV PGAS variants on the same simulated extremal time series,
# with optional *comparable* priors (PC vs IG) via a shared tail statement.
# Forces progress output by setting progress_every (or closest synonym) in cfg.
# Adds unified MCMC/SMC settings and a prior preview/plot BEFORE running samplers.

import argparse, importlib, json, os, sys, time
from datetime import datetime
import numpy as np
import math
import matplotlib.pyplot as plt

from simulator.extremal_time_series import Extremal_Time_Series as ETS

# ----------------------------------------------------------------------------- #
# Utilities
# ----------------------------------------------------------------------------- #

def import_class(spec: str):
    mod_path, _, cls_name = spec.partition(":")
    if not mod_path or not cls_name:
        raise ValueError(f"Bad import spec '{spec}'. Expected 'pkg.module:ClassName'.")
    mod = importlib.import_module(mod_path)
    try:
        return getattr(mod, cls_name)
    except AttributeError:
        raise ValueError(f"Class '{cls_name}' not found in module '{mod_path}'.")

def dataclass_from_dict(dc_type, maybe_dict):
    if not maybe_dict:
        return dc_type()
    fields = set(getattr(dc_type, "__dataclass_fields__", {}).keys())
    filtered = {k: v for k, v in dict(maybe_dict).items() if k in fields}
    return dc_type(**filtered)

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def mean_or_nan(x):
    return float(np.mean(x)) if (x is not None and len(x) > 0) else float("nan")

def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape or a.size == 0: return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))

def robust_sd(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) / 1.4826)

def enable_progress(cfg, every: int):
    for name in ("progress_every", "print_every", "report_every", "prog_every"):
        if hasattr(cfg, name):
            cur = getattr(cfg, name)
            if not cur or int(cur) <= 0:
                try: setattr(cfg, name, int(every))
                except Exception: pass
    for name in ("progress", "print_progress", "show_progress", "verbose"):
        if hasattr(cfg, name):
            try: setattr(cfg, name, True)
            except Exception: pass

def set_if_has(obj, name, value):
    if value is None: return
    if hasattr(obj, name):
        try: setattr(obj, name, value)
        except Exception: pass

def apply_mcmc_settings(cfg, args):
    # Core MCMC
    set_if_has(cfg, "n_iter", args.n_iter)
    set_if_has(cfg, "burn", args.burn)
    set_if_has(cfg, "thin", args.thin)
    # RW–MH steps
    set_if_has(cfg, "step_logsigma", args.step_logsigma)
    set_if_has(cfg, "step_xi", args.step_xi)
    set_if_has(cfg, "step_level", args.step_level)
    set_if_has(cfg, "step_slope", args.step_slope)
    set_if_has(cfg, "step_season", args.step_season)
    set_if_has(cfg, "step_log_s_alpha", args.step_log_s_alpha)
    set_if_has(cfg, "step_log_s_beta", args.step_log_s_beta)
    set_if_has(cfg, "step_log_s_gamma", args.step_log_s_gamma)
    # Adaptation
    set_if_has(cfg, "adapt_steps", bool(args.adapt_steps))
    set_if_has(cfg, "adapt_every", args.adapt_every)
    set_if_has(cfg, "adapt_until", args.adapt_until)
    set_if_has(cfg, "adapt_target_1d", args.adapt_target_1d)
    set_if_has(cfg, "adapt_eta0", args.adapt_eta0)
    set_if_has(cfg, "adapt_eta_decay", args.adapt_decay)
    set_if_has(cfg, "step_min", args.step_min)
    set_if_has(cfg, "step_max", args.step_max)
    # SMC / PF
    set_if_has(cfg, "n_particles", args.particles)
    set_if_has(cfg, "trans_eps", args.trans_eps)
    set_if_has(cfg, "ess_threshold_frac", args.ess_threshold_frac)
    # Randomness + progress
    set_if_has(cfg, "random_seed", args.seed)
    enable_progress(cfg, args.progress_every)

# ----------------------------------------------------------------------------- #
# Comparable prior calibration (shared tail statement on s = sqrt(Q))
# ----------------------------------------------------------------------------- #

def prior_scales_from_y(y, frac_alpha=0.10, frac_beta=0.10, frac_gamma=0.10):
    dy  = np.diff(y)
    d2y = np.diff(y, n=2)
    u_alpha = max(1e-12, frac_alpha * robust_sd(dy))
    u_beta  = max(1e-12, frac_beta  * (robust_sd(d2y) if d2y.size else robust_sd(dy)))
    u_gamma = max(1e-12, frac_gamma * robust_sd(dy))
    return dict(alpha=u_alpha, beta=u_beta, gamma=u_gamma)

def pc_lambda_from_u(u, alpha_tail=0.05):
    return float(-np.log(max(1e-12, alpha_tail)) / max(1e-12, u))

def ig_from_pc_u(u, alpha_tail=0.05, how="moment", a_fixed=2.2):
    lam = pc_lambda_from_u(u, alpha_tail)
    if how == "moment":
        a = 2.2
        b = 2.4 / (lam**2)
        return a, b, lam
    elif how == "quantile":
        try:
            from scipy.special import gammainccinv
        except Exception as e:
            raise RuntimeError("how='quantile' requires SciPy (scipy.special.gammainccinv)") from e
        a = float(a_fixed)
        t = float(gammainccinv(a, 1.0 - float(alpha_tail)))
        b = t * (u**2)
        return a, b, lam
    else:
        raise ValueError("how must be 'moment' or 'quantile'.")

def build_priors_dict_for_model(model_key, base_priors_dict, y, args):
    pri = dict(base_priors_dict or {})
    if not args.calibrate_priors:
        return pri
    U = prior_scales_from_y(
        y,
        frac_alpha=args.scale_frac_alpha,
        frac_beta=args.scale_frac_beta,
        frac_gamma=args.scale_frac_gamma,
    )
    a_a, b_a, lam_a = ig_from_pc_u(U["alpha"], alpha_tail=args.tail_alpha, how=args.ig_map, a_fixed=args.ig_a)
    a_b, b_b, lam_b = ig_from_pc_u(U["beta"],  alpha_tail=args.tail_alpha, how=args.ig_map, a_fixed=args.ig_a)
    a_g, b_g, lam_g = ig_from_pc_u(U["gamma"], alpha_tail=args.tail_alpha, how=args.ig_map, a_fixed=args.ig_a)

    if model_key in ("ig", "ig_epf"):
        pri.update({
            "a_q_alpha": float(a_a), "b_q_alpha": float(b_a),
            "a_q_beta":  float(a_b), "b_q_beta":  float(b_b),
            "a_q_gamma": float(a_g), "b_q_gamma": float(b_g),
        })
    elif model_key in ("pc", "pc_epf"):
        pri_alpha = dict(pri.get("pc_alpha", {}))
        pri_beta  = dict(pri.get("pc_beta",  {}))
        pri_gamma = dict(pri.get("pc_gamma", {}))
        pri_alpha.update({"lambda_s": lam_a, "frac": args.scale_frac_alpha, "alpha_prob": args.tail_alpha})
        pri_beta.update( {"lambda_s": lam_b, "frac": args.scale_frac_beta,  "alpha_prob": args.tail_alpha})
        pri_gamma.update({"lambda_s": lam_g, "frac": args.scale_frac_gamma, "alpha_prob": args.tail_alpha})
        pri.update({"pc_alpha": pri_alpha, "pc_beta": pri_beta, "pc_gamma": pri_gamma})
    return pri

# ----------------------- Prior preview / plotting --------------------------- #

def _ensure_pc_lambda(pc_dict, u_default, tail_alpha):
    lam = None
    if isinstance(pc_dict, dict):
        lam = pc_dict.get("lambda_s", None)
    if lam is None:
        lam = pc_lambda_from_u(u_default, tail_alpha)
    return float(lam)

def _ensure_ig_params(pri_dict, key, u_default, tail_alpha, ig_map="moment", ig_a=2.2):
    a = pri_dict.get(f"a_q_{key}", None)
    b = pri_dict.get(f"b_q_{key}", None)
    if (a is None) or (b is None):
        a, b, _ = ig_from_pc_u(u_default, alpha_tail=tail_alpha, how=ig_map, a_fixed=ig_a)
    return float(a), float(b)

def _pc_pdf_s(s, lam):
    s = np.asarray(s, float)
    out = lam * np.exp(-lam * s)
    out[s < 0] = 0.0
    return out

def _ig_pdf_q(q, a, b):
    # IG(a, b): f(q) = b^a / Γ(a) * q^{-a-1} * exp(-b/q), q>0
    q = np.asarray(q, float)
    out = np.zeros_like(q)
    mask = q > 0
    if not np.any(mask): return out
    from math import gamma, log, exp
    c = (b**a) / gamma(a)
    out[mask] = c * (q[mask] ** (-a - 1.0)) * np.exp(-b / q[mask])
    return out

def _ig_pdf_s_from_q(s, a, b):
    # s = sqrt(q) => q = s^2, f_s(s) = 2s f_Q(s^2)
    s = np.asarray(s, float)
    q = np.square(s)
    return 2.0 * s * _ig_pdf_q(q, a, b)

def _ig_quantile_s(a, b, alpha_tail):
    # solve P(Q > q)=alpha => gammaincc(a, b/q)=alpha => t=gammainccinv(a, alpha), q=b/t, u=sqrt(q)
    try:
        from scipy.special import gammainccinv
        t = float(gammainccinv(float(a), float(alpha_tail)))
        if t <= 0: return float("nan")
        q = b / t
        return math.sqrt(q) if q > 0 else float("nan")
    except Exception:
        return None  # SciPy not available

def preview_process_noise_priors(y, args, runs):
    # Data-calibrated u's for fallback/defaults
    U = prior_scales_from_y(
        y,
        frac_alpha=args.scale_frac_alpha,
        frac_beta=args.scale_frac_beta,
        frac_gamma=args.scale_frac_gamma,
    )

    # For each run, build the priors dict AS USED and extract params
    per_model = {}
    for label, model_key, _, _, _, priors_json, _ in runs:
        base = json.loads(priors_json) if priors_json.strip() else {}
        pri = build_priors_dict_for_model(model_key, base, y, args)

        if model_key in ("pc","pc_epf"):
            # PC priors over s
            lamA = _ensure_pc_lambda(pri.get("pc_alpha", {}), U["alpha"], args.tail_alpha)
            lamB = _ensure_pc_lambda(pri.get("pc_beta",  {}), U["beta"],  args.tail_alpha)
            lamG = _ensure_pc_lambda(pri.get("pc_gamma", {}), U["gamma"], args.tail_alpha)
            per_model[label] = {"type":"pc", "alpha":lamA, "beta":lamB, "gamma":lamG}
        else:
            # IG priors over Q, will plot on s via transform
            aA,bA = _ensure_ig_params(pri, "alpha", U["alpha"], args.tail_alpha, args.ig_map, args.ig_a)
            aB,bB = _ensure_ig_params(pri, "beta",  U["beta"],  args.tail_alpha, args.ig_map, args.ig_a)
            aG,bG = _ensure_ig_params(pri, "gamma", U["gamma"], args.tail_alpha, args.ig_map, args.ig_a)
            per_model[label] = {"type":"ig", "alpha":(aA,bA), "beta":(aB,bB), "gamma":(aG,bG)}

    # Build s-grids per component using broad upper limits across models
    grids = {}
    for comp, u in (("alpha",U["alpha"]), ("beta",U["beta"]), ("gamma",U["gamma"])):
        s_max_candidates = [10*u]  # default wide
        for label, spec in per_model.items():
            if spec["type"]=="pc":
                lam = spec[comp]
                s_max_candidates.append(7.0/lam)  # ~cover several e-folds
            else:
                a,b = spec[comp]
                # heuristic: 6 * sqrt(E[Q]) when defined
                if a>1:
                    s_max_candidates.append(6.0*np.sqrt(b/(a-1)))
                else:
                    s_max_candidates.append(10.0*np.sqrt(b/max(a,1e-6)))
        s_max = float(min(max(s_max_candidates), 1e6))
        s_grid = np.linspace(0.0, s_max, 800)
        grids[comp] = s_grid

    # Plot (3 rows, one per component)
    if args.plot_priors:
        nrows=3; ncols=1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8, 9), sharex=False)
        comps = ["alpha","beta","gamma"]
        for ax, comp in zip(axes, comps):
            s = grids[comp]
            any_plotted = False
            for label, spec in per_model.items():
                if spec["type"]=="pc":
                    lam = spec[comp]
                    ypdf = _pc_pdf_s(s, lam)
                    ax.plot(s, ypdf, label=f"{label} (PC λ={lam:.3g})")
                    any_plotted = True
                else:
                    a,b = spec[comp]
                    ypdf = _ig_pdf_s_from_q(s, a, b)
                    ax.plot(s, ypdf, label=f"{label} (IG a={a:.2f}, b={b:.3g})")
                    any_plotted = True
            ax.set_title(f"Prior over s = sqrt(Q) for {comp}")
            ax.set_xlabel("s")
            ax.set_ylabel("density")
            if any_plotted: ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.25)

        fig.tight_layout()
        out_path = args.priors_out
        if out_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = args.out_dir or "."
            out_path = os.path.join(base, f"prior_preview_{ts}.png")
        ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=160)
        print(f"[priors] Saved prior preview plot -> {out_path}")


    def pc_u(lam): return -math.log(args.tail_alpha)/lam
    try:
        from scipy.special import gammainccinv
        have_scipy=True
    except Exception:
        have_scipy=False
    for label, spec in per_model.items():
        row = [label]
        for comp in ("alpha","beta","gamma"):
            if spec["type"]=="pc":
                u = pc_u(spec[comp])
                row.append(f"{comp}:{u:.4g}")
            else:
                a,b = spec[comp]
                if have_scipy:
                    u = _ig_quantile_s(a,b,args.tail_alpha)
                    row.append(f"{comp}:{(u if (u is not None) else float('nan')):.4g}")
                else:
                    row.append(f"{comp}:n/a")
        print("  - " + " | ".join(row))

    return per_model

# ----------------------------------------------------------------------------- #
# One model run
# ----------------------------------------------------------------------------- #

def run_one(label, model_key, mod_spec, priors_cls_spec, cfg_cls_spec,
            priors_json, cfg_json, y, truths, args):
    DGEV   = import_class(mod_spec)
    Priors = import_class(priors_cls_spec)
    Cfg    = import_class(cfg_cls_spec)

    base_priors_dict = json.loads(priors_json) if priors_json.strip() else {}
    priors_dict = build_priors_dict_for_model(model_key, base_priors_dict, y, args)

    priors = dataclass_from_dict(Priors, priors_dict)
    cfg    = dataclass_from_dict(Cfg, json.loads(cfg_json) if cfg_json.strip() else {})

    if hasattr(cfg, "random_seed") and getattr(cfg, "random_seed") is None:
        cfg.random_seed = args.seed
    apply_mcmc_settings(cfg, args)

    # Deterministic seasonal init if needed
    seasonal_init_pminus1 = None
    if args.season_mode == "deterministic":
        g = np.cos(2 * np.pi * np.arange(args.period) / args.period)
        g -= np.mean(g)
        seasonal_init_pminus1 = g[: args.period - 1].astype(float)

    # Dynamic seasonal initial priors for constructor
    if args.season_mode == "dynamic":
        m0_season_first = np.zeros(args.period - 1, float)
        v0_season_first = np.full(args.period - 1, args.v0_season, float)
    else:
        m0_season_first = None
        v0_season_first = None

    sampler = DGEV(
        y=np.asarray(y, float),
        period=args.period,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.season_mode,
        m0_level=args.level_init, v0_level=args.v0_level,
        m0_trend=(args.slope_init if args.trend_mode != "none" else 0.0), v0_trend=args.v0_trend,
        m0_season=m0_season_first, v0_season=v0_season_first,
        priors=priors,
        cfg=cfg,
        level_value_init=args.level_init,
        slope_value_init=args.slope_init,
        seasonal_vector_init=seasonal_init_pminus1,
    )

    mu_T, alpha_T, beta_T, gamma_T = truths
    true_Q = []
    if args.level_mode  == "dynamic": true_Q.append(args.q_alpha)
    if args.trend_mode  == "dynamic": true_Q.append(args.q_beta)
    if args.season_mode == "dynamic": true_Q += [args.q_gamma] + [0.0] * (args.period - 2)
    sampler.set_truth(sigma=args.true_sigma, xi=args.true_xi,
                      Q=(np.asarray(true_Q, float) if len(true_Q) else None))
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0

    pm_sigma = mean_or_nan(post.get("sigma"))
    pm_xi    = mean_or_nan(post.get("xi"))
    le       = post.get("log_evidence", np.array([]))
    le_mean  = mean_or_nan(le)
    le_med   = float(np.nanmedian(le)) if le.size else float("nan")
    le_max   = float(np.nanmax(le)) if le.size else float("nan")
    mu_hat   = (np.nanmean(post.get("mu", np.empty((0, len(y)))), axis=0)
                if post.get("mu", None) is not None and post["mu"].size else None)
    mu_rmse  = rmse(mu_hat, mu_T) if (mu_hat is not None and mu_T is not None) else float("nan")

    acc  = getattr(sampler, "accept", {}) or {}
    prop = getattr(sampler, "proposals", {}) or {}
    def pct(k):
        a, p = acc.get(k, 0), prop.get(k, 0)
        return (100.0 * a / p) if p else 0.0
    acc_logs, acc_xi = pct("logsigma"), pct("xi")

    pf = getattr(sampler, "last_pf_diag", {}) or {}
    ess_mean = pf.get("ess_mean", float("nan"))
    resample_rate = pf.get("resample_rate", float("nan"))

    if args.save and args.out_dir:
        out_dir = os.path.join(args.out_dir, label)
        ensure_dir(out_dir)
        sampler.save_posterior(
            os.path.join(out_dir, "posterior.npz"),
            extra_meta={"model": label, "elapsed_seconds": float(elapsed)},
        )
        out_meta = {
            "pm_sigma": pm_sigma, "pm_xi": pm_xi,
            "le_mean": le_mean, "le_med": le_med, "le_max": le_max,
            "mu_rmse": mu_rmse, "elapsed_sec": elapsed,
            "acc_logsigma_pct": acc_logs, "acc_xi_pct": acc_xi,
            "ess_mean": ess_mean, "resample_rate": resample_rate,
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(out_meta, f, indent=2)

    return {
        "label": label, "elapsed": elapsed,
        "pm_sigma": pm_sigma, "pm_xi": pm_xi,
        "le_mean": le_mean, "le_med": le_med, "le_max": le_max,
        "mu_rmse": mu_rmse,
        "acc_logsigma_pct": acc_logs, "acc_xi_pct": acc_xi,
        "ess_mean": ess_mean, "resample_rate": resample_rate,
    }

# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description="Compare 4 DGEV PGAS variants on the same extremal time series, "
                    "with optional comparable PC/IG priors and unified MCMC settings."
    )

    # Import specs
    p.add_argument("--mod-ig",            default="optimization.dgev_pgas_ig:DGEVParticleGibbs")
    p.add_argument("--mod-ig-priors",     default="optimization.dgev_pgas_ig:Priors")
    p.add_argument("--mod-ig-cfg",        default="optimization.dgev_pgas_ig:SamplerConfig")
    p.add_argument("--mod-ig-epf",        default="optimization.dgev_pgas_ig_epf:DGEVParticleGibbs")
    p.add_argument("--mod-ig-epf-priors", default="optimization.dgev_pgas_ig_epf:Priors")
    p.add_argument("--mod-ig-epf-cfg",    default="optimization.dgev_pgas_ig_epf:SamplerConfig")
    p.add_argument("--mod-pc",            default="optimization.dgev_pgas_pc:DGEVParticleGibbs")
    p.add_argument("--mod-pc-priors",     default="optimization.dgev_pgas_pc:Priors")
    p.add_argument("--mod-pc-cfg",        default="optimization.dgev_pgas_pc:SamplerConfig")
    p.add_argument("--mod-pc-epf",        default="optimization.dgev_pgas_pc_epf:DGEVParticleGibbs")
    p.add_argument("--mod-pc-epf-priors", default="optimization.dgev_pgas_pc_epf:Priors")
    p.add_argument("--mod-pc-epf-cfg",    default="optimization.dgev_pgas_pc_epf:SamplerConfig")

    # Which models to run
    p.add_argument("--which", nargs="+", default=["ig", "ig_epf", "pc", "pc_epf"],
                   help="Choose among: ig ig_epf pc pc_epf")

    # Simulator knobs
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--level-mode",  choices=["dynamic","deterministic"], default="dynamic")
    p.add_argument("--trend-mode",  choices=["dynamic","deterministic","none"], default="none")
    p.add_argument("--season-mode", choices=["dynamic","deterministic","none"], default="none")
    p.add_argument("--true-sigma", type=float, default=2.0)
    p.add_argument("--true-xi",    type=float, default=0.1)
    p.add_argument("--q-alpha",    type=float, default=1e-3)
    p.add_argument("--q-beta",     type=float, default=1e-9)
    p.add_argument("--q-gamma",    type=float, default=1e-7)

    # Initial state priors for simulation
    p.add_argument("--level-init", type=float, default=5.0)
    p.add_argument("--v0-level",   type=float, default=0.2)
    p.add_argument("--slope-init", type=float, default=0.02)
    p.add_argument("--v0-trend",   type=float, default=0.05)
    p.add_argument("--v0-season",  type=float, default=0.5)

    # Per-model knobs via JSON
    p.add_argument("--priors-ig",        default="{}", help="JSON for Priors (IG/Bootstrap).")
    p.add_argument("--cfg-ig",           default="{}", help="JSON for SamplerConfig (IG/Bootstrap).")
    p.add_argument("--priors-ig-epf",    default="{}", help="JSON for Priors (IG/Laplace).")
    p.add_argument("--cfg-ig-epf",       default="{}", help="JSON for SamplerConfig (IG/Laplace).")
    p.add_argument("--priors-pc",        default="{}", help="JSON for Priors (PC/Bootstrap).")
    p.add_argument("--cfg-pc",           default="{}", help="JSON for SamplerConfig (PC/Bootstrap).")
    p.add_argument("--priors-pc-epf",    default="{}", help="JSON for Priors (PC/Laplace).")
    p.add_argument("--cfg-pc-epf",       default="{}", help="JSON for SamplerConfig (PC/Laplace).")

    # Comparable-priors calibration
    p.add_argument("--calibrate-priors", action="store_true",
                   help="Override process-noise priors so PC and IG share the same tail statement.")
    p.add_argument("--tail-alpha", type=float, default=0.05,
                   help="α in P(s > u) = α used for PC and IG calibration.")
    p.add_argument("--scale-frac-alpha", type=float, default=0.10,
                   help="u_alpha = frac * robust_sd(Δy).")
    p.add_argument("--scale-frac-beta",  type=float, default=0.10,
                   help="u_beta  = frac * robust_sd(Δ²y).")
    p.add_argument("--scale-frac-gamma", type=float, default=0.10,
                   help="u_gamma = frac * robust_sd(Δy).")
    p.add_argument("--ig-map", choices=["moment","quantile"], default="moment",
                   help="Map PC tail statement to IG: moment (closed form) or quantile (SciPy).")
    p.add_argument("--ig-a", type=float, default=2.2,
                   help="IG shape a used when ig-map=quantile (reported for completeness).")

    # Progress
    p.add_argument("--progress-every", type=int, default=10,
                   help="Force progress printing every N iterations if supported by the sampler config.")

    # -------------------------- MCMC / SMC settings -------------------------- #
    p.add_argument("--n-iter", type=int, default=2000)
    p.add_argument("--burn", type=int, default=100)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--step-logsigma", type=float, default=0.2)
    p.add_argument("--step-xi",       type=float, default=0.2)
    p.add_argument("--step-level",    type=float, default=0.02)
    p.add_argument("--step-slope",    type=float, default=0.001)
    p.add_argument("--step-season",   type=float, default=0.02)
    p.add_argument("--step-log-s-alpha", type=float, default=0.22)
    p.add_argument("--step-log-s-beta",  type=float, default=0.10)
    p.add_argument("--step-log-s-gamma", type=float, default=0.10)
    p.add_argument("--adapt-steps", action="store_true", help="Enable Robbins–Monro step-size adaptation.")
    p.add_argument("--adapt-every", type=int, default=25)
    p.add_argument("--adapt-until", choices=["burn","all"], default="all")
    p.add_argument("--adapt-target-1d", type=float, default=0.44)
    p.add_argument("--adapt-eta0", type=float, default=0.2)
    p.add_argument("--adapt-decay", type=float, default=0.75)
    p.add_argument("--step-min", type=float, default=1e-5)
    p.add_argument("--step-max", type=float, default=1.0)
    p.add_argument("--particles", type=int, default=250)
    p.add_argument("--trans-eps", type=float, default=1e-8)
    p.add_argument("--ess-threshold-frac", type=float, default=0.5,
                   help="Resample non-reference particles when ESS/N < this fraction.")

    # Prior plotting
    p.add_argument("--plot-priors", action="store_true", default=True,
                   help="If set, draw and save prior densities for s = sqrt(Q) before running.")
    p.add_argument("--priors-out", type=str, default=None,
                   help="PNG path to save the prior preview (defaults into --out-dir).")

    # Misc
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--save", action="store_true",
                   help="Save per-model posterior npz + meta to out-dir/<model>/")

    args = p.parse_args()
    np.random.seed(args.seed)

    # Seasonal arrays for simulator
    if args.season_mode == "deterministic":
        g = np.cos(2 * np.pi * np.arange(args.period) / args.period)
        g -= np.mean(g)
        m0_season_first = g[: args.period - 1].astype(float)
        v0_season_first = np.full(args.period - 1, args.v0_season, float)
    elif args.season_mode == "dynamic":
        m0_season_first = np.zeros(args.period - 1, float)
        v0_season_first = np.full(args.period - 1, args.v0_season, float)
    else:
        m0_season_first = None
        v0_season_first = None

    # Simulate data
    ts = ETS(
        parameters=(args.true_sigma, args.true_xi),
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.season_mode,
        period=args.period,
        q_level=args.q_alpha,
        q_trend=args.q_beta,
        q_season=args.q_gamma,
        m0_level=args.level_init, v0_level=args.v0_level,
        m0_trend=(args.slope_init if args.trend_mode != "none" else 0.0), v0_trend=args.v0_trend,
        m0_season=m0_season_first, v0_season=v0_season_first,
        start_date=datetime(1980, 1, 1),
    )
    y = []
    for _ in range(args.T):
        ts.move(); y.append(ts.measure())
    y = np.asarray(y, float)

    # Truth paths
    truths_raw = ts.get_truth_paths(as_numpy=False)
    mu_T    = np.asarray(truths_raw["mu"][1:1 + args.T], float)
    alpha_T = np.asarray(truths_raw["alpha"][1:1 + args.T], float) if args.level_mode  == "dynamic" else None
    beta_T  = np.asarray(truths_raw["beta"][1:1 + args.T],  float) if args.trend_mode  == "dynamic" else None
    gamma_T = np.asarray(truths_raw["gamma_last"][1:1 + args.T], float) if args.season_mode == "dynamic" else None
    truths = (mu_T, alpha_T, beta_T, gamma_T)

    # Build run list
    runs = []
    if "ig" in args.which:
        runs.append(("IG_bootstrap", "ig",
                     args.mod_ig, args.mod_ig_priors, args.mod_ig_cfg,
                     args.priors_ig, args.cfg_ig))
    if "ig_epf" in args.which:
        runs.append(("IG_Laplace", "ig_epf",
                     args.mod_ig_epf, args.mod_ig_epf_priors, args.mod_ig_epf_cfg,
                     args.priors_ig_epf, args.cfg_ig_epf))
    if "pc" in args.which:
        runs.append(("PC_bootstrap", "pc",
                     args.mod_pc, args.mod_pc_priors, args.mod_pc_cfg,
                     args.priors_pc, args.cfg_pc))
    if "pc_epf" in args.which:
        runs.append(("PC_Laplace", "pc_epf",
                     args.mod_pc_epf, args.mod_pc_epf_priors, args.mod_pc_epf_cfg,
                     args.priors_pc_epf, args.cfg_pc_epf))

    if args.out_dir:
        ensure_dir(args.out_dir)

    # -------- Prior preview BEFORE running samplers -------- #
    _ = preview_process_noise_priors(y, args, runs)

    # Execute
    rows = []
    for label, model_key, mod_spec, priors_spec, cfg_spec, priors_json, cfg_json in runs:
        print(f"\n=== Running {label} (progress_every≈{args.progress_every}) ===")
        try:
            row = run_one(label, model_key, mod_spec, priors_spec, cfg_spec,
                          priors_json, cfg_json, y, truths, args)
            rows.append(row)
            print(f"[{label}] time={row['elapsed']:.2f}s | "
                  f"σ̄={row['pm_sigma']:.3f} ξ̄={row['pm_xi']:.3f} | "
                  f"logZ(mean/med/max)={row['le_mean']:.2f}/{row['le_med']:.2f}/{row['le_max']:.2f} | "
                  f"RMSE(μ̂, μ*)={row['mu_rmse']:.3f} | "
                  f"acc(logσ)={row['acc_logsigma_pct']:.1f}% acc(ξ)={row['acc_xi_pct']:.1f}% | "
                  f"ESS̄={row['ess_mean']:.1f} RS%={row['resample_rate']:.2f}")
        except Exception as e:
            print(f"[{label}] ERROR: {e}", file=sys.stderr)

    if rows:
        print("\n=== Summary ===")
        head = f"{'model':14} {'time(s)':>8} {'sigmā':>8} {'xī':>7} {'logZ_mean':>10} {'logZ_max':>9} {'RMSE_μ':>8}"
        print(head)
        print("-" * len(head))
        for r in rows:
            print(f"{r['label']:14} {r['elapsed']:8.2f} {r['pm_sigma']:8.3f} {r['pm_xi']:7.3f} "
                  f"{r['le_mean']:10.2f} {r['le_max']:9.2f} {r['mu_rmse']:8.3f}")

if __name__ == "__main__":
    main()
