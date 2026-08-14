"""Reproducible v2 validation for shared-factor compilation and inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.stats import multivariate_normal

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx
from bucex.core.numerics import gaussian_support


def gaussian_likelihood_check() -> dict[str, object]:
    model = bx.FactorModel(
        channels=(bx.Channel("a", bx.Gaussian()), bx.Channel("b", bx.Gaussian())),
        factors=(
            bx.Factor(
                "shared",
                (bx.LocalLevel(initial_mean=0.0, initial_sd=1.1),),
                {"a": 1.0, "b": 0.7},
            ),
        ),
    )
    y = np.asarray([[0.2, 0.0], [-0.1, 0.1], [0.4, 0.2], [0.3, 0.4]])
    compiled = bx.compile_model(model, y)
    params = {
        "sd.factor.shared.level": 0.25,
        "sigma.a": 0.4,
        "sigma.b": 0.6,
    }
    actual = bx.kalman_filter(y, compiled, params).log_likelihood
    n_time = y.shape[0]
    covariance = np.zeros((2 * n_time, 2 * n_time))
    loadings = np.asarray([1.0, 0.7])
    sigmas = np.asarray([0.4, 0.6])
    for t in range(n_time):
        for s in range(n_time):
            latent = 1.1**2 + (min(t, s) + 1) * 0.25**2
            for i in range(2):
                for j in range(2):
                    covariance[2 * t + i, 2 * s + j] = loadings[i] * loadings[j] * latent
    covariance += np.diag(np.tile(sigmas**2, n_time))
    expected = float(
        multivariate_normal.logpdf(y.reshape(-1), mean=np.zeros(y.size), cov=covariance)
    )
    error = abs(actual - expected)
    return {
        "kalman_log_likelihood": float(actual),
        "joint_normal_log_likelihood": expected,
        "absolute_error": float(error),
        "passed": bool(error < 1e-9),
    }


def loading_recovery_check() -> dict[str, object]:
    model = bx.FactorModel(
        channels=(bx.Channel("a", bx.Gaussian()), bx.Channel("b", bx.Gaussian())),
        factors=(
            bx.Factor(
                "shared",
                (bx.LocalLevel(initial_mean=0.0, initial_sd=1.0),),
                {
                    "a": 1.0,
                    "b": bx.Loading.estimated(0.2, sd=2.0),
                },
            ),
        ),
    )
    truth = {
        "sd.factor.shared.level": 0.20,
        "sigma.a": 0.12,
        "sigma.b": 0.15,
        "loading.shared.b": 0.65,
    }
    simulation = bx.simulate(model, 60, truth, seed=101)
    priors = bx.FactorPriors(
        process={"factor.shared.level": bx.FixedSD(0.20)},
        observation_sd={"a": bx.FixedSD(0.12), "b": bx.FixedSD(0.15)},
        shape={},
        profile="fixed-nuisance recovery",
    )
    started = time.perf_counter()
    fit = bx.fit(
        simulation.y,
        model,
        priors=priors,
        parameterization="centered",
        mcmc=bx.MCMC(draws=100, warmup=60, chains=1, seed=102),
    )
    elapsed = time.perf_counter() - started
    draws = fit.loading_draws("shared", "b")
    interval = np.quantile(draws, [0.05, 0.50, 0.95])
    recovered = bool(interval[0] <= truth["loading.shared.b"] <= interval[2])
    return {
        "truth": truth,
        "loading_mean": float(np.mean(draws)),
        "loading_q05_q50_q95": interval.tolist(),
        "truth_in_90_interval": recovered,
        "loading_acceptance": float(fit.acceptance["loading.shared.b"]),
        "state_shape": list(fit.state_draws.shape),
        "all_finite": bool(np.all(np.isfinite(fit.state_draws))),
        "elapsed_seconds": float(elapsed),
        "passed": bool(recovered and np.all(np.isfinite(fit.state_draws))),
    }


def mixed_boundary_check() -> tuple[dict[str, object], bx.FitResult]:
    model = bx.FactorModel(
        channels=(bx.Channel("mean", bx.Gaussian()), bx.Channel("tail", bx.GEV())),
        factors=(
            bx.Factor(
                "climate",
                (
                    bx.LocalLinearTrend(
                        initial_level=0.0,
                        initial_slope=0.0,
                        initial_level_sd=1.0,
                        initial_slope_sd=0.01,
                    ),
                ),
                {"mean": 1.0, "tail": 0.8},
            ),
        ),
        name="near-boundary mixed factor",
    )
    truth = {
        "sd.factor.climate.level": 0.03,
        "sd.factor.climate.slope": 2e-7,
        "sigma.mean": 0.20,
        "sigma.tail": 0.30,
        "xi.tail": -0.05,
    }
    simulation = bx.simulate(model, 24, truth, seed=201)
    compiled = bx.compile_model(model, simulation.y)
    support = gaussian_support(compiled.transition_cov(truth))
    priors = bx.FactorPriors(
        process={
            "factor.climate.level": bx.FixedSD(0.03),
            "factor.climate.slope": bx.FixedSD(2e-7),
        },
        observation_sd={
            "mean": bx.FixedSD(0.20),
            "tail": bx.FixedSD(0.30),
        },
        shape={"tail": bx.TruncatedNormalPrior(0.0, 0.2, -0.5, 0.5)},
        profile="near-boundary fixed-scale",
    )
    started = time.perf_counter()
    fit = bx.fit(
        simulation.y,
        model,
        priors=priors,
        init={"xi.tail": -0.05},
        engine="pgas",
        parameterization="centered",
        mcmc=bx.MCMC(draws=50, warmup=25, chains=1, seed=202),
        particles=bx.Particles(n=32, proposal="guided"),
        laplace=bx.Laplace(max_iterations=12),
    )
    elapsed = time.perf_counter() - started
    engine = fit.diagnostics()["engine"]
    slope_terminal_sd = float(np.std(fit.state_draws[:, :, -1, 1]))
    passed = bool(
        support.active_variances.size == 2
        and np.all(np.isfinite(fit.state_draws))
        and engine["path_change_rate"] > 0.5
        and slope_terminal_sd > 0.0
    )
    laplace_plan = bx.plan(model, simulation.y, engine="laplace")
    return (
        {
            "truth": truth,
            "active_transition_variances": support.active_variances.tolist(),
            "active_rank": int(support.active_variances.size),
            "state_shape": list(fit.state_draws.shape),
            "all_finite": bool(np.all(np.isfinite(fit.state_draws))),
            "terminal_slope_draw_sd": slope_terminal_sd,
            "pgas_plan": fit.plan.to_dict(),
            "laplace_plan": laplace_plan.to_dict(),
            "particle_diagnostics": engine,
            "elapsed_seconds": float(elapsed),
            "passed": passed,
        },
        fit,
    )


def persistence_and_forecast_check(fit: bx.FitResult) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "factor.bucex"
        fit.save(path)
        restored = bx.FitResult.load(path)
        state_error = float(np.max(np.abs(restored.state_draws - fit.state_draws)))
        eta_error = float(np.max(np.abs(restored.eta_draws() - fit.eta_draws())))
    forecast = fit.forecast(4, draws=10, seed=301)
    finite = bool(
        np.all(np.isfinite(forecast.observations))
        and np.all(np.isfinite(forecast.eta))
    )
    return {
        "archive_state_max_abs_error": state_error,
        "archive_eta_max_abs_error": eta_error,
        "forecast_shape": list(forecast.observations.shape),
        "forecast_all_finite": finite,
        "passed": bool(state_error == 0.0 and eta_error == 0.0 and finite),
    }


def uccle_constructor_check(data_dir: str | Path) -> dict[str, object]:
    data = bx.load_uccle_factor_data(
        data_dir,
        start="2000-01-01",
        end="2001-12-01",
    )
    model = bx.make_uccle_factor_model(structure="contrasts")
    compiled = bx.compile_model(
        model,
        data.to_numpy() * model.transform_signs[None, :],
    )
    return {
        "channel_names": list(model.channel_names),
        "factor_names": list(model.factor_names),
        "data_shape": list(data.shape),
        "state_dim": int(compiled.state_dim),
        "noise_dim": int(compiled.noise_dim),
        "design_shape": list(compiled.design().shape),
        "passed": bool(
            tuple(data.columns) == model.channel_names
            and compiled.design().shape[:2] == data.shape
        ),
    }


def run(data_dir: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    likelihood = gaussian_likelihood_check()
    recovery = loading_recovery_check()
    boundary, fit = mixed_boundary_check()
    persistence = persistence_and_forecast_check(fit)
    uccle = uccle_constructor_check(data_dir)
    sections = {
        "gaussian_likelihood": likelihood,
        "loading_recovery": recovery,
        "mixed_near_boundary": boundary,
        "persistence_and_forecast": persistence,
        "uccle_constructor": uccle,
    }
    return {
        "release": "bucex 2.0.0",
        "version": bx.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
        "all_passed": bool(all(section["passed"] for section in sections.values())),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output",
        default="validation/factor_validation_2.0.0.json",
    )
    args = parser.parse_args()
    result = run(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
