# %% simulator/dlm_crossval.py
from __future__ import annotations
"""
Rolling-origin cross-validation for the Gaussian DLM (Bayesian lasso sampler)

- Robust run discovery like simulator/dlm_plotter.py:
    load_posterior/find_latest_run (+ fallback to newest posterior*.npz)
- Re-fits the DLM on training prefixes and produces fine-scale forecast fan charts

Plot conventions (Uccle-style)
-----------------------------
- Observations (train + held-out): black
- Forecast median + credible band: TX* red, TN* blue (else C0)
- No title, no legend

Reusable entry point:
    DLMCrossValidator

CLI is provided at the bottom and simply instantiates the class.

Expected bundle content
-----------------------
- draws['y'] : (T,) the observation series
- meta['period'] optional (default 12)
- meta['start_date'] optional (or pass --start-date) for date-like splits
- meta['series'] optional (e.g. "TXm"/"TNm") to control forecast colors

Outputs (default)
-----------------
<run>/crossval/
  cv_split_<label>_t<idx>_h<H>/
      forecast_fine.png
      metrics.json
      forecast_payload.npz   (includes y_train, y_test, y_future_draws, etc.)
  crossval_summary.csv
  crossval_summary.json
"""

import os
import sys
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make optimization/ visible (mirrors dlm_plotter/dlm_forecast pattern)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------
# I/O helpers: posterior loader + robust latest-run discovery
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e

try:
    from simulator.utils import _ensure_dir, _parse_date_ymd, find_latest_posterior_npz  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import simulator.utils (need _ensure_dir, _parse_date_ymd, find_latest_posterior_npz).\n"
        "Make sure simulator/utils.py is on PYTHONPATH."
    ) from e

# Sampler (re-fit per split)
try:
    from optimization.dlm_3 import DLMGibbsConjugate, Priors, SamplerConfig  # type: ignore
except Exception:
    from dlm_3 import DLMGibbsConjugate, Priors, SamplerConfig  # type: ignore

# Forecast simulator (posterior predictive)
try:
    from simulator.dlm_forecast import simulate_dlm_forecast  # type: ignore
except Exception:
    from dlm_forecast import simulate_dlm_forecast  # type: ignore


TimeSpec = Union[int, float, str]


# =============================================================================
# Color logic (mirrors uccle_dlm_plotter.py)
# =============================================================================
def series_color(series: str) -> str:
    s = str(series).upper()
    if s.startswith("TX"):
        return "red"
    if s.startswith("TN"):
        return "blue"
    return "C0"


def infer_series_code(meta: Dict[str, Any], path_hint: Optional[str] = None) -> str:
    # Prefer explicit metadata
    for k in ("series", "target", "name"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Fallback: infer from path
    if path_hint:
        m = re.search(r"\b(TX[a-zA-Z0-9]*|TN[a-zA-Z0-9]*|PREC[a-zA-Z0-9]*|PRECX)\b", path_hint.upper())
        if m:
            return m.group(1)

    return ""


# =============================================================================
# Data containers
# =============================================================================
@dataclass(frozen=True)
class SplitResult:
    split_label: str
    split_idx: int
    T_train: int
    H: int
    level: float
    coverage: float
    rmse: float
    mae: float
    avg_width: float
    split_dir: str


# =============================================================================
# Cross-validator class
# =============================================================================
class DLMCrossValidator:
    def __init__(
        self,
        y: np.ndarray,
        meta: Dict[str, Any],
        *,
        priors: Optional[Priors] = None,
        cfg: Optional[SamplerConfig] = None,
        out_dir: Optional[str] = None,
        level: float = 0.90,
        start_date_override: Optional[datetime] = None,
        seed_forecast: int = 123,
        window: int = 240,
        show: bool = False,
        ylabel: str = "y ",
        path_hint: Optional[str] = None,
    ):
        self.y = np.asarray(y, float).ravel()
        if self.y.size < 5:
            raise ValueError("Need at least 5 observations.")

        self.meta = dict(meta)
        self.level = float(level)
        if not (0.0 < self.level < 1.0):
            raise ValueError("level must be in (0,1).")

        self.period = int(self.meta.get("period", 12) or 12)
        self.layout = self._default_layout(self.meta, self.period)

        self.priors = priors if priors is not None else self._priors_from_meta(self.meta)
        self.cfg = cfg if cfg is not None else SamplerConfig()

        self.out_dir = out_dir
        self.start_date_override = start_date_override or self._start_date_from_meta(self.meta)

        self.seed_forecast = int(seed_forecast)
        self.window = int(window)
        self.show = bool(show)
        self.ylabel = str(ylabel)

        # series inference for coloring
        self.path_hint = path_hint
        self.series_code = infer_series_code(self.meta, self.path_hint)

        # ensure key meta fields
        self.meta["period"] = int(self.period)
        self.meta["layout"] = list(self.layout)
        if self.start_date_override is not None:
            self.meta["start_date"] = self.start_date_override.strftime("%Y-%m-%d")
        if self.series_code and not isinstance(self.meta.get("series", None), str):
            self.meta["series"] = str(self.series_code)

    # ----------------------------- constructors ----------------------------- #
    @classmethod
    def from_posterior_bundle(
        cls,
        run_path: str,
        *,
        out_dir: Optional[str] = None,
        level: float = 0.90,
        start_date: Optional[str] = None,
        seed_forecast: int = 123,
        window: int = 240,
        show: bool = False,
        ylabel: str = "y ",
        priors: Optional[Priors] = None,
        cfg: Optional[SamplerConfig] = None,
    ) -> "DLMCrossValidator":
        bundle = load_posterior(run_path)
        draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path
        if "y" not in draws:
            raise ValueError("This crossval expects draws['y'] in the posterior bundle.")
        y = np.asarray(draws["y"], float).ravel()

        sd_override = _parse_date_optional(start_date)
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(npz_path), "crossval")

        return cls(
            y=y,
            meta=meta,
            priors=priors,
            cfg=cfg,
            out_dir=out_dir,
            level=level,
            start_date_override=sd_override,
            seed_forecast=seed_forecast,
            window=window,
            show=show,
            ylabel=ylabel,
            path_hint=str(npz_path),
        )

    @classmethod
    def from_latest(
        cls,
        *,
        root: str,
        target: Optional[str] = None,
        out_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "DLMCrossValidator":
        run_path = cls.resolve_run_path(target=target, root=root)
        return cls.from_posterior_bundle(run_path, out_dir=out_dir, **kwargs)

    @staticmethod
    def resolve_run_path(*, target: Optional[str], root: str) -> str:
        # Mirrors dlm_plotter.py
        run_path = target
        if run_path is None:
            print(f"[info] --target not provided; searching for the latest posterior under --root={root!r} ...")
            run_path = find_latest_run(root=root)
            if run_path is None:
                npz_path = find_latest_posterior_npz(root)
                if npz_path is None:
                    raise FileNotFoundError(f"No posterior runs found under {root!r}. Provide --target or change --root.")
                run_path = npz_path
                print(f"[info] find_latest_run found nothing; using latest npz: {run_path}")
            else:
                print(f"[info] Using latest run: {run_path}")
        return str(run_path)

    # ----------------------------- meta / parsing ----------------------------- #
    @staticmethod
    def _default_layout(meta: Dict[str, Any], period: int) -> List[str]:
        layout = meta.get("layout", None)
        if isinstance(layout, (list, tuple)) and len(layout) >= 2:
            return list(layout)
        # default for DLM lasso model: alpha, beta, g1..g_{p-1}
        return ["alpha", "beta"] + [f"g{k}" for k in range(1, int(period))]

    @staticmethod
    def _priors_from_meta(meta: Dict[str, Any]) -> Priors:
        pri = Priors()
        if isinstance(meta.get("priors", None), dict):
            try:
                pri = Priors(**dict(meta["priors"]))
            except Exception:
                pri = Priors()
        return pri

    @staticmethod
    def _start_date_from_meta(meta: Dict[str, Any]) -> Optional[datetime]:
        sd = meta.get("start_date", None) or meta.get("start", None) or meta.get("t0", None)
        if sd is None:
            return None
        return _parse_date_optional(str(sd))

    def _time_to_index(self, spec: TimeSpec, *, T: int) -> Tuple[int, str]:
        """
        Accepts:
          - int index (0..T-1)
          - float in (0,1): fraction of sample length
          - 'YYYY'/'YYYY-MM'/'YYYY-MM-DD' (needs start_date and period dividing 12)
        """
        if isinstance(spec, (int, np.integer)):
            i = int(spec)
            if not (0 <= i <= T - 1):
                raise ValueError(f"Split index {i} out of range [0, {T-1}].")
            return i, f"idx{i}"

        if isinstance(spec, (float, np.floating)):
            x = float(spec)
            if not (0.0 < x < 1.0):
                raise ValueError("Float split specs must be in (0,1).")
            i = int(round(x * (T - 1)))
            i = max(0, min(T - 1, i))
            return i, f"p{x:.3g}".replace(".", "p")

        s = str(spec).strip()
        if s.isdigit():
            return self._time_to_index(int(s), T=T)

        # try fraction
        try:
            xf = float(s)
            if 0.0 < xf < 1.0:
                return self._time_to_index(xf, T=T)
        except Exception:
            pass

        # date-like
        dt0 = self.start_date_override
        if dt0 is None:
            raise ValueError(
                f"Got date-like split {s!r} but no start_date is available "
                f"(provide --start-date or store meta['start_date'])."
            )

        period = int(self.period)
        if period <= 0:
            period = 12
        if period not in (12, 6, 4, 3, 2, 1) or (12 % period != 0):
            raise ValueError(f"Date-like splits require period dividing 12. Got period={period}.")

        dt = _parse_date_optional(s)
        if dt is None:
            raise ValueError(f"Could not parse split date {s!r}.")

        step_months = 12 // period
        months0 = dt0.year * 12 + (dt0.month - 1)
        months = dt.year * 12 + (dt.month - 1)
        diff = months - months0
        i = int(round(diff / step_months))
        if not (0 <= i <= T - 1):
            raise ValueError(f"Split date {s!r} maps to index {i}, out of range [0, {T-1}].")
        label = s.replace("-", "")
        return i, label

    # ----------------------------- model fit / forecast ----------------------------- #
    def fit_on_prefix(self, y_train: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fit the Bayesian lasso DLM on y_train and return the sampler keep-draws dict.
        """
        y_train = np.asarray(y_train, float).ravel()
        sig0 = float(np.nanstd(y_train))
        if not np.isfinite(sig0) or sig0 <= 0:
            sig0 = 1.0

        K = int(self.period) - 1
        gamma0_init = np.zeros((K,), float) if K > 0 else None

        sampler = DLMGibbsConjugate(
            y=y_train,
            period=int(self.period),
            sigma2_init=sig0 * sig0,
            s_alpha_init=1e-2,
            s_beta_init=1e-3,
            s_gamma_init=1e-3,
            alpha0=float(self.priors.m0_alpha),
            beta0=float(self.priors.m0_beta),
            gamma0=gamma0_init,
            priors=self.priors,
            cfg=self.cfg,
        )
        post = sampler.run()
        post = dict(post)
        post["y"] = y_train.copy()
        return post

    def forecast_from_posterior(self, post_train: Dict[str, np.ndarray], horizon: int):
        meta_train = dict(self.meta)
        meta_train["period"] = int(self.period)
        meta_train["layout"] = list(self.layout)
        if self.start_date_override is not None:
            meta_train["start_date"] = self.start_date_override.strftime("%Y-%m-%d")
        if self.series_code:
            meta_train["series"] = str(self.series_code)

        return simulate_dlm_forecast(
            draws=post_train,
            meta=meta_train,
            horizon=int(horizon),
            seed=int(self.seed_forecast),
            start_date=self.start_date_override,
        )

    # ----------------------------- metrics / plotting ----------------------------- #
    @staticmethod
    def summarize_ribbon(draws_2d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        loq = (1.0 - float(level)) / 2.0
        hiq = 1.0 - loq
        med = np.quantile(draws_2d, 0.5, axis=0)
        lo = np.quantile(draws_2d, loq, axis=0)
        hi = np.quantile(draws_2d, hiq, axis=0)
        return med, lo, hi

    def compute_metrics(
        self,
        y_future_draws: np.ndarray,  # (S, H)
        y_future_actual: np.ndarray,  # (H,)
    ) -> Dict[str, float]:
        y_true = np.asarray(y_future_actual, float).ravel()
        S, H = np.asarray(y_future_draws).shape
        if y_true.size != H:
            raise ValueError("y_future_actual must have length equal to forecast horizon H.")

        med, lo, hi = self.summarize_ribbon(y_future_draws, level=self.level)
        cover = (y_true >= lo) & (y_true <= hi)

        coverage = float(np.mean(cover))
        rmse = float(np.sqrt(np.mean((med - y_true) ** 2)))
        mae = float(np.mean(np.abs(med - y_true)))
        avg_width = float(np.mean(hi - lo))
        return {"coverage": coverage, "rmse": rmse, "mae": mae, "avg_width": avg_width}

    def plot_forecast_with_actual(
        self,
        *,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        x_future: np.ndarray,
        y_future_draws: np.ndarray,
        y_future_actual: np.ndarray,
        split_x: Any,
        title: str,   # kept for API compatibility; ignored
        ylabel: str,
        save_path: str,
    ) -> None:
        med, lo, hi = self.summarize_ribbon(y_future_draws, level=self.level)
        y_true = np.asarray(y_future_actual, float).ravel()

        col = series_color(self.series_code or infer_series_code(self.meta, self.path_hint))

        fig, ax = plt.subplots(1, 1, figsize=(12, 3.9))

        # observations: black
        ax.plot(x_obs, y_obs, lw=1.2, color="black")
        ax.plot(x_future, y_true, lw=1.3, linestyle="--", color="black")

        # forecast: colored median + colored band
        ax.fill_between(x_future, lo, hi, alpha=0.25, color=col, linewidth=0)
        ax.plot(x_future, med, lw=1.8, color=col)

        ax.axvline(split_x, lw=1.0, alpha=0.8, color="black")

        # no title / no legend
        ax.set_title("")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        _ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if self.show:
            plt.show()
        else:
            plt.close(fig)

    # ----------------------------- main runner ----------------------------- #
    def run(
        self,
        *,
        splits: Sequence[TimeSpec],
        horizon: int,
    ) -> List[SplitResult]:
        T_full = int(self.y.size)
        H_req = int(horizon)
        if H_req <= 0:
            raise ValueError("horizon must be > 0")

        if self.out_dir is None:
            raise ValueError("out_dir is None. Provide out_dir when constructing the cross-validator.")
        _ensure_dir(self.out_dir)

        # map splits -> indices; de-duplicate; sort
        mapped: Dict[int, str] = {}
        for s in splits:
            idx, lab = self._time_to_index(s, T=T_full)
            mapped[int(idx)] = str(lab)
        split_list = sorted(mapped.items(), key=lambda z: z[0])

        results: List[SplitResult] = []

        for split_idx, split_lab in split_list:
            T_train = int(split_idx) + 1
            if T_train < 5:
                print(f"[warn] split {split_lab} -> T_train={T_train} too small; skipping.")
                continue

            max_h = (T_full - T_train)
            if max_h <= 0:
                print(f"[warn] split {split_lab} has no future observations; skipping.")
                continue
            H = min(H_req, max_h)

            y_train = self.y[:T_train]
            y_test = self.y[T_train : T_train + H]

            split_dir = os.path.join(self.out_dir, f"cv_split_{split_lab}_t{split_idx}_h{H}")
            _ensure_dir(split_dir)

            print(f"\n[cv] split={split_lab} idx={split_idx} -> train={T_train} test={H}")

            # fit
            post_train = self.fit_on_prefix(y_train)

            # forecast
            fr = self.forecast_from_posterior(post_train, horizon=H)
            # fr expected fields: y_obs, x_axis_full, y_future, split_x

            # plotting window
            Tt = int(np.asarray(fr.y_obs).size)
            x_full = np.asarray(fr.x_axis_full)
            i0 = max(0, Tt - int(self.window))

            x_obs = x_full[i0:Tt]
            y_obs = np.asarray(fr.y_obs)[i0:Tt]

            x_future = x_full[Tt : Tt + H]
            y_future_draws = np.asarray(fr.y_future)[:, :H]

            # plot
            self.plot_forecast_with_actual(
                x_obs=x_obs,
                y_obs=y_obs,
                x_future=x_future,
                y_future_draws=y_future_draws,
                y_future_actual=y_test,
                split_x=fr.split_x,
                title="",                 # ignored
                ylabel=self.ylabel,
                save_path=os.path.join(split_dir, "forecast_fine.png"),
            )

            # metrics
            m = self.compute_metrics(y_future_draws, y_test)

            # write per-split artifacts
            with open(os.path.join(split_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "split": {
                            "split_label": split_lab,
                            "split_idx": int(split_idx),
                            "T_train": int(T_train),
                            "H": int(H),
                            "level": float(self.level),
                            **m,
                        },
                        "cfg": asdict(self.cfg),
                        "priors": asdict(self.priors),
                        "seed_forecast": int(self.seed_forecast),
                        "window": int(self.window),
                        "series": str(self.series_code),
                    },
                    f,
                    indent=2,
                )

            print(
                f"[cv] coverage={m['coverage']:.3f}  rmse={m['rmse']:.3f}  "
                f"mae={m['mae']:.3f}  avg_width={m['avg_width']:.3f}"
            )

            results.append(
                SplitResult(
                    split_label=str(split_lab),
                    split_idx=int(split_idx),
                    T_train=int(T_train),
                    H=int(H),
                    level=float(self.level),
                    coverage=float(m["coverage"]),
                    rmse=float(m["rmse"]),
                    mae=float(m["mae"]),
                    avg_width=float(m["avg_width"]),
                    split_dir=str(split_dir),
                )
            )

            # optional payload
            try:
                np.savez_compressed(
                    os.path.join(split_dir, "forecast_payload.npz"),
                    y_train=y_train,
                    y_test=y_test,
                    y_future_draws=y_future_draws,
                    x_obs=x_obs,
                    y_obs=y_obs,
                    x_future=x_future,
                )
            except Exception as e:
                print(f"[warn] could not write forecast_payload.npz: {e}")

        # combined summary
        if results:
            self.write_summary(results, out_dir=self.out_dir)
        else:
            print("[done] No splits were evaluated (nothing to write).")

        return results

    @staticmethod
    def write_summary(results: Sequence[SplitResult], *, out_dir: str) -> None:
        _ensure_dir(out_dir)

        # CSV
        csv_path = os.path.join(out_dir, "crossval_summary.csv")
        cols = ["split_label", "split_idx", "T_train", "H", "level", "coverage", "rmse", "mae", "avg_width"]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in results:
                row = asdict(r)
                f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")

        # JSON
        json_path = os.path.join(out_dir, "crossval_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        print(f"[save] {csv_path}")
        print(f"[save] {json_path}")


# =============================================================================
# Small helpers
# =============================================================================
def _parse_date_optional(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "":
        return None
    return _parse_date_ymd(ss)


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Rolling-origin cross-validation for Bayesian lasso DLM.\n"
            "Loads y from an existing posterior bundle, then re-fits on prefixes and forecasts into held-out data.\n"
            "Run discovery mirrors dlm_plotter.py.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run discovery
    p.add_argument("--target", type=str, default=None, help="Run dir or posterior.npz. If omitted, uses latest under --root.")
    p.add_argument("--root", type=str, default="results/simulations/DLM", help="Search root when --target is omitted.")

    # Splits
    p.add_argument(
        "--splits",
        type=str,
        default="0.6",
        help=(
            "Comma-separated split specs. Each item can be:\n"
            "  - an integer index (0-based), e.g. 900\n"
            "  - a fraction in (0,1), e.g. 0.8\n"
            "  - a date 'YYYY'/'YYYY-MM'/'YYYY-MM-DD' (needs start_date and period dividing 12)\n"
        ),
    )
    p.add_argument("--horizon", type=int, default=120, help="Forecast horizon (fine-scale steps).")

    # Plots / outputs
    p.add_argument("--level", type=float, default=0.90, help="Forecast band level.")
    p.add_argument("--seed-forecast", type=int, default=123, help="RNG seed for posterior predictive simulation.")
    p.add_argument("--window", type=int, default=240, help="Plot window: last N training points to show before split.")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--ylabel", type=str, default="y", help="Y-axis label for forecast plots.")
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override meta['start_date'] (YYYY / YYYY-MM / YYYY-MM-DD). Also enables date-like splits.",
    )

    # MCMC config per split
    p.add_argument("--n-iter", type=int, default=4000, help="MCMC iterations per split fit.")
    p.add_argument("--burn", type=int, default=1000, help="Burn-in per split fit.")
    p.add_argument("--thin", type=int, default=1, help="Thinning per split fit.")
    p.add_argument("--seed-mcmc", type=int, default=40, help="MCMC RNG seed (same seed used for each split).")
    p.add_argument("--no-progress", action="store_true", default=False, help="Disable sampler progress prints.")
    p.add_argument("--progress-every", type=int, default=0, help="Sampler progress frequency (0 => ~2%).")

    # Prior overrides (optional)
    p.add_argument("--prior-a-sigma", type=float, default=None)
    p.add_argument("--prior-b-sigma", type=float, default=None)
    p.add_argument("--prior-m0-alpha", type=float, default=None)
    p.add_argument("--prior-P0-alpha", type=float, default=None)
    p.add_argument("--prior-m0-beta", type=float, default=None)
    p.add_argument("--prior-P0-beta", type=float, default=None)
    p.add_argument("--prior-P0-gamma", type=float, default=None)
    p.add_argument("--prior-a-lambda", type=float, default=None)
    p.add_argument("--prior-b-lambda", type=float, default=None)

    # Output
    p.add_argument("--out", type=str, default=None, help="Output directory. Default: <run>/crossval")

    args = p.parse_args()

    # Resolve run path like dlm_plotter
    run_path = DLMCrossValidator.resolve_run_path(target=args.target, root=args.root)
    bundle = load_posterior(run_path)
    npz_path = bundle.npz_path

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "crossval")
    _ensure_dir(out_dir)

    # Build priors (meta first, then CLI overrides)
    meta_full = bundle.meta
    pri = Priors()
    if isinstance(meta_full.get("priors", None), dict):
        try:
            pri = Priors(**dict(meta_full["priors"]))
        except Exception:
            pri = Priors()

    for k_cli, k_pri in [
        ("prior_a_sigma", "a_sigma"),
        ("prior_b_sigma", "b_sigma"),
        ("prior_m0_alpha", "m0_alpha"),
        ("prior_P0_alpha", "P0_alpha"),
        ("prior_m0_beta", "m0_beta"),
        ("prior_P0_beta", "P0_beta"),
        ("prior_P0_gamma", "P0_gamma"),
        ("prior_a_lambda", "a_lambda"),
        ("prior_b_lambda", "b_lambda"),
    ]:
        v = getattr(args, k_cli)
        if v is not None:
            setattr(pri, k_pri, float(v))

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed_mcmc),
        progress=(not bool(args.no_progress)),
        progress_every=int(args.progress_every),
    )

    # Parse splits list
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip() != ""]
    if not splits:
        raise SystemExit("[error] --splits parsed to an empty list.")

    cv = DLMCrossValidator.from_posterior_bundle(
        run_path,
        out_dir=out_dir,
        level=float(args.level),
        start_date=args.start_date,
        seed_forecast=int(args.seed_forecast),
        window=int(args.window),
        show=bool(args.show),
        ylabel=str(args.ylabel),
        priors=pri,
        cfg=cfg,
    )

    print(f"[info] using posterior: {bundle.npz_path}")
    print(f"[info] writing crossval outputs to: {out_dir}")
    print(f"[info] T={cv.y.size}, period={cv.period}, series={cv.series_code!r}")

    cv.run(splits=splits, horizon=int(args.horizon))
    print("[done] cross-validation finished.")
