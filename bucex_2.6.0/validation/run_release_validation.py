#!/usr/bin/env python3
"""Fixed-seed numerical release validation for bucex 2.6.0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


def _dates(n_time: int) -> np.ndarray:
    return np.arange(
        np.datetime64("2000-01"),
        np.datetime64("2000-01") + np.timedelta64(n_time, "M"),
        dtype="datetime64[M]",
    )


def _model(*, mixed: bool = False) -> bx.MultiSeriesModel:
    return bx.MultiSeriesModel(
        channels=(
            bx.Channel(
                "mean",
                bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
            ),
            bx.Channel(
                "maximum",
                bx.GEV() if mixed else bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
            ),
        ),
        name="release validation",
    )


def _finite_fit(fit: bx.FitResult) -> dict[str, object]:
    return {
        "backend": fit.plan.backend,
        "engine": fit.plan.engine,
        "finite_states": bool(np.all(np.isfinite(fit.state_draws))),
        "finite_log_posterior": bool(np.all(np.isfinite(fit.log_posterior))),
        "restored_iterations": int(fit.meta.get("restored_iterations", 0)),
        "initial_parameters": sorted(
            name for name in fit.parameter_draws if name.startswith("initial.")
        ),
    }


def run() -> dict[str, object]:
    started = time.perf_counter()
    if bx.__version__ != "2.6.0":
        raise RuntimeError(f"Expected bucex 2.6.0, found {bx.__version__}.")
    default_hierarchy = bx.HierarchicalPrior()
    if default_hierarchy.model_space != "componentwise":
        raise RuntimeError("The hierarchy must default to componentwise SSVS.")
    rng = np.random.default_rng(2500)
    record: dict[str, object] = {
        "bucex_version": bx.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # Long, nearly deterministic Gaussian path: regression for stable Joseph
    # covariance handling and estimated initial level/slope.
    n_long = 1000
    time_index = np.arange(n_long)
    long_values = (
        0.002 * time_index
        + 0.3 * np.sin(2.0 * np.pi * time_index / 12.0)
        + rng.normal(scale=0.4, size=n_long)
    )
    long_fit = bx.fit(
        long_values,
        family="gaussian",
        period=12,
        priors="normal",
        parameterization="fs",
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2401),
    )
    record["long_gaussian"] = {
        **_finite_fit(long_fit),
        "sign_invariance_max_error": float(
            np.nanmax(long_fit.draws_aux["sign_invariance_error"])
        ),
        "initial_level_finite": bool(
            np.all(np.isfinite(long_fit.parameter("alpha0")))
        ),
        "initial_slope_finite": bool(
            np.all(np.isfinite(long_fit.parameter("beta0")))
        ),
    }

    n_time = 24
    season = 0.15 * np.sin(2.0 * np.pi * np.arange(n_time) / 4.0)
    gaussian_values = np.column_stack(
        (
            0.02 * np.arange(n_time) + season + rng.normal(scale=0.15, size=n_time),
            0.01 * np.arange(n_time) + 0.8 * season + rng.normal(scale=0.2, size=n_time),
        )
    )
    pooling = {}
    for index, mode in enumerate(("selection", "slab", "both")):
        fit = bx.fit(
            gaussian_values,
            _model(),
            priors=bx.HierarchicalPrior(pool=mode),
            mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=2410 + index),
            dates=_dates(n_time),
        )
        probabilities = fit.component_probabilities()
        pooling[mode] = {
            **_finite_fit(fit),
            "sign_invariance_max_error": float(
                np.nanmax(fit.draws_aux["sign_invariance_error"])
            ),
            "season_zero_probability": probabilities.xs(
                "seasonal", level="process"
            )["zero"].tolist(),
            "hierarchy_probability_rows": int(
                fit.hierarchical_probabilities().shape[0]
            ),
            "hierarchy_slab_rows": int(fit.hierarchical_slab_summary().shape[0]),
            "model_space": fit.meta["trend_model_space"],
            "joint_structure_rows": int(fit.structural_model_probabilities().shape[0]),
        }
    record["hierarchical_gaussian"] = pooling

    joint_prior = bx.HierarchicalPrior(
        pool="selection",
        model_space="joint_trend",
        trend_states=("fixed", "dynamic"),
        trend_concentration=(1.0, 1.0),
    )
    joint_fit = bx.fit(
        gaussian_values,
        _model(),
        priors=joint_prior,
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=2519),
        dates=_dates(n_time),
    )
    trend_models = joint_fit.hierarchical_trend_model_probabilities()
    record["optional_joint_trend_space"] = {
        **_finite_fit(joint_fit),
        "rows": int(trend_models.shape[0]),
        "probability_sum": float(trend_models["mean"].sum()),
    }

    mixed_values = np.column_stack(
        (
            rng.normal(scale=0.2, size=16),
            rng.gumbel(scale=0.3, size=16),
        )
    )
    laplace_screen = bx.fit(
        mixed_values,
        _model(mixed=True),
        priors=bx.HierarchicalPrior(pool="selection"),
        engine="laplace",
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2420),
        dates=_dates(16),
    )
    record["mixed_hierarchical_laplace"] = {
        **_finite_fit(laplace_screen),
        "targets_exact_posterior": bool(
            laplace_screen.plan.targets_exact_posterior
        ),
        "warm_start_keys": sorted(laplace_screen.warm_start()),
    }

    mixed_fit = bx.fit(
        mixed_values,
        _model(mixed=True),
        priors=bx.HierarchicalPrior(pool="selection"),
        engine="pgas",
        init=laplace_screen,
        particles=bx.Particles(n=20, proposal="guided"),
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2421),
        dates=_dates(16),
    )
    record["mixed_guided_pgas"] = {
        **_finite_fit(mixed_fit),
        "targets_exact_posterior": bool(mixed_fit.plan.targets_exact_posterior),
        "external_warm_start": bool(mixed_fit.meta["external_warm_start"]),
        "path_update_fraction_finite": bool(
            np.all(
                np.isfinite(
                    mixed_fit.draws_aux["particle_path_update_fraction"]
                )
            )
        ),
    }

    disturbance_fit = bx.fit(
        rng.gumbel(size=18),
        family="gev",
        period=4,
        priors="normal",
        engine="pgas",
        parameterization="disturbance",
        asis=False,
        particles=bx.Particles(n=20, proposal="guided"),
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2422),
    )
    record["disturbance_guided_pgas"] = _finite_fit(disturbance_fit)
    record["total_seconds"] = time.perf_counter() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/release_validation_2.6.0.json"),
    )
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
