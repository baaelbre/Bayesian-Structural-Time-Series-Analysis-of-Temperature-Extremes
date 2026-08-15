#!/usr/bin/env python3
"""Fixed-seed univariate compatibility validation for bucex 2.1.5."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


PROFILES = (
    "manuscript_lasso",
    "regularized_lasso",
    "regularized_horseshoe",
    "triple_gamma",
    "regularized_triple_gamma",
    "pc",
    "normal",
    "ssvs",
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def _fit_record(fit):
    return {
        "plan": fit.plan.to_dict(),
        "state_shape": list(fit.state_draws.shape),
        "parameters": sorted(fit.parameter_draws),
        "prior_profile": fit.meta["prior_profile"],
        "finite_states": bool(np.all(np.isfinite(fit.state_draws))),
        "finite_log_posterior": bool(np.all(np.isfinite(fit.log_posterior))),
    }


def run(data_dir: str | Path, particles: int) -> dict[str, object]:
    started_total = time.perf_counter()
    record: dict[str, object] = {
        "bucex_version": bx.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "single_result_type": bx.PosteriorBundle is bx.FitResult,
    }
    if bx.__version__ != "2.1.5":
        raise RuntimeError(f"Expected bucex 2.1.5, found {bx.__version__}.")

    data_table = bx.validate_uccle_data(data_dir, check_daily=True)
    record["uccle_data"] = data_table.reset_index().to_dict(orient="records")

    rng = np.random.default_rng(20)
    exog = rng.normal(size=(36, 1))
    regression = bx.fit(
        1.5 * exog[:, 0] + rng.normal(scale=0.25, size=36),
        bx.Model(
            bx.Gaussian(),
            [bx.LocalLevel(), bx.Regression(1, dynamic=True, name="x")],
        ),
        exog=exog,
        parameterization="disturbance",
        priors="normal",
        asis=True,
        mcmc=bx.MCMC(draws=2, warmup=2, chains=2, seed=21),
    )
    record["dynamic_regression"] = _fit_record(regression)

    sample = bx.load_uccle_series("TXm", data_dir).iloc[:48]
    parameterizations = {}
    for index, parameterization in enumerate(
        ("centered", "disturbance", "fruehwirth_schnatter")
    ):
        result = bx.fit(
            sample,
            family="gaussian",
            period=12,
            parameterization=parameterization,
            priors="normal",
            asis=True,
            mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=30 + index),
        )
        parameterizations[parameterization] = _fit_record(result)
    record["gaussian_parameterizations"] = parameterizations

    profile_results = {}
    for index, profile in enumerate(PROFILES):
        result = bx.fit(
            sample,
            family="gaussian",
            period=12,
            parameterization="fruehwirth_schnatter",
            priors=profile,
            mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=40 + index),
        )
        profile_results[profile] = _fit_record(result)
    record["fs_prior_profiles"] = profile_results

    tail_sample = bx.load_uccle_series("TXx", data_dir).iloc[:60]
    laplace = bx.fit(
        tail_sample,
        family="gev",
        period=12,
        parameterization="fruehwirth_schnatter",
        engine="laplace",
        priors="regularized_lasso",
        asis=True,
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=50),
        laplace=bx.Laplace(max_iterations=15),
    )
    record["gev_laplace"] = {
        **_fit_record(laplace),
        "engine_diagnostics": laplace.diagnostics()["engine"],
        "restored_iterations": laplace.meta["restored_iterations"],
    }

    full_series = {}
    for index, name in enumerate(bx.UCCLE_SERIES):
        started = time.perf_counter()
        result = bx.fit_uccle_series(
            name,
            data_dir,
            priors="normal",
            parameterization="fruehwirth_schnatter",
            engine="auto",
            asis=False,
            mcmc=bx.MCMC(draws=1, warmup=0, chains=1, seed=70 + index),
        )
        full_series[name] = {
            **_fit_record(result),
            "seconds": time.perf_counter() - started,
            "engine_diagnostics": result.diagnostics()["engine"],
        }
    record["full_uccle_smoke"] = full_series

    started = time.perf_counter()
    full_txm = bx.fit_uccle_series(
        "TXm",
        data_dir,
        priors="normal",
        parameterization="centered",
        asis=True,
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=60),
    )
    record["txm_full_centered"] = {
        **_fit_record(full_txm),
        "seconds": time.perf_counter() - started,
    }

    started = time.perf_counter()
    full_txx = bx.fit_uccle_series(
        "TXx",
        data_dir,
        priors="regularized_horseshoe",
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        asis=True,
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=61),
        particles=bx.Particles(n=particles, proposal="guided"),
    )
    record["txx_full_pgas"] = {
        **_fit_record(full_txx),
        "particles": particles,
        "seconds": time.perf_counter() - started,
        "engine_diagnostics": full_txx.diagnostics()["engine"],
        "restored_iterations": full_txx.meta["restored_iterations"],
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fit.bucex"
        full_txx.save(path)
        restored = bx.FitResult.load(path)
        forecast = restored.forecast(3, draws=1, seed=62)
        record["archive_and_forecast"] = {
            "roundtrip_equal": bool(
                np.array_equal(restored.state_draws, full_txx.state_draws)
            ),
            "model_type": type(restored.model).__name__,
            "prior_type": type(restored.priors).__name__,
            "archive_bytes": path.stat().st_size,
            "forecast_shape": list(forecast.observations.shape),
            "forecast_finite": bool(np.all(np.isfinite(forecast.observations))),
        }

    record["total_seconds"] = time.perf_counter() - started_total
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--particles", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/univariate_validation_2.1.5.json"),
    )
    args = parser.parse_args()
    result = run(args.data_dir, args.particles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_value),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
