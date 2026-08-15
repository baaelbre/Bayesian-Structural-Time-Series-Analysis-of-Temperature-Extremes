#!/usr/bin/env python3
"""Fixed-seed validation for the bucex 2.1.5 one-factor release."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx
from bucex.inference.fit.factor_fs import (
    CompiledFactorFS,
    _initial_fs_parameters,
)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _two_channel_model(*, mixed: bool = True) -> bx.FactorModel:
    return bx.FactorModel(
        channels=(
            bx.Channel(
                "bulk",
                bx.Gaussian(),
                components=(bx.LocalLevel(), bx.DummySeasonal(period=4)),
            ),
            bx.Channel(
                "tail",
                bx.GEV() if mixed else bx.Gaussian(),
                components=(bx.LocalLevel(), bx.DummySeasonal(period=4)),
            ),
        ),
        factors=(
            bx.Factor(
                "common",
                (
                    bx.LocalLinearTrend(
                        initial_level=0.0,
                        initial_slope=0.0,
                        initial_level_sd=0.0,
                        initial_slope_sd=0.0,
                    ),
                ),
                {
                    "bulk": 1.0,
                    "tail": bx.Loading.estimated(0.9, sd=1.0),
                },
            ),
        ),
        name="v2.1 validation model",
    )


def uccle_graph_check(data_dir: str | Path) -> tuple[dict[str, object], object, object]:
    data = bx.load_uccle_factor_data(
        data_dir, start="2000-01-01", end="2001-12-01"
    )
    model = bx.make_uccle_factor_model()
    compiled = bx.compile_model(
        model, data.to_numpy() * model.transform_signs[None, :]
    )
    plan = bx.plan(model, data)
    priors = bx.default_factor_priors(compiled)
    expected_horseshoe = {
        f"channel.{channel}.level" for channel in model.channel_names
    }
    trend = model.factor("common").components[0]
    passed = bool(
        model.factor_names == ("common",)
        and model.supports_fs_parameterization
        and model.factor("common").loading_for("TXm").value == 1.0
        and compiled.state_dim == 74
        and compiled.noise_dim == 14
        and plan.parameterization == "fruehwirth_schnatter"
        and plan.engine == "pgas"
        and set(priors.horseshoe_processes) == expected_horseshoe
        and trend.initial_slope_sd == 0.0
    )
    return (
        {
            "channels": list(model.channel_names),
            "factors": list(model.factor_names),
            "data_shape": list(data.shape),
            "centered_state_dim": int(compiled.state_dim),
            "noise_dim": int(compiled.noise_dim),
            "plan": plan.to_dict(),
            "prior_profile": priors.profile,
            "horseshoe_processes": list(priors.horseshoe_processes),
            "initial_factor_slope_fixed": bool(trend.initial_slope_sd == 0.0),
            "passed": passed,
        },
        data,
        compiled,
    )


def fs_algebra_check(compiled) -> dict[str, object]:
    rng = np.random.default_rng(101)
    priors = bx.default_factor_priors(compiled)
    fs = CompiledFactorFS(compiled)
    params = _initial_fs_parameters(compiled, fs, priors, rng, None)
    path = np.zeros((compiled.n_time + 1, fs.state_dim))
    path[0] = fs.initial_mean
    for time_index in range(compiled.n_time):
        path[time_index + 1] = (
            fs.transition @ path[time_index]
            + fs.loading @ rng.normal(size=fs.noise_dim)
        )
    centered = fs.to_centered(path, params)
    eta_fs = fs.eta(path, params=params)
    eta_centered = compiled.eta(centered, params=params)
    restored = fs.from_centered(centered, params)
    eta_error = float(np.max(np.abs(eta_fs - eta_centered)))
    path_error = float(np.max(np.abs(path - restored)))

    # The NCP transition remains a fixed unit-innovation graph when process
    # coefficients approach zero; q only suppresses an observation contribution.
    idio = priors.horseshoe_processes[0]
    near_zero = dict(params)
    near_zero[f"signed_sd.{idio}"] = 1e-14
    near_zero[f"sd.{idio}"] = 1e-14
    finite_near_zero = bool(np.all(np.isfinite(fs.eta(path, params=near_zero))))
    transition_rank = int(np.linalg.matrix_rank(fs.transition_cov(near_zero)))
    passed = bool(
        eta_error < 1e-10
        and path_error < 1e-10
        and finite_near_zero
        and transition_rank == fs.noise_dim
    )
    return {
        "fs_state_dim": int(fs.state_dim),
        "unit_innovation_dim": int(fs.noise_dim),
        "unit_transition_rank_at_q_1e_minus_14": transition_rank,
        "eta_max_abs_error": eta_error,
        "centered_ncp_roundtrip_max_abs_error": path_error,
        "near_zero_eta_finite": finite_near_zero,
        "passed": passed,
    }


def six_channel_weight_check(data, compiled) -> dict[str, object]:
    rng = np.random.default_rng(102)
    priors = bx.default_factor_priors(compiled)
    fs = CompiledFactorFS(compiled)
    params = _initial_fs_parameters(compiled, fs, priors, rng, None)
    particles = rng.normal(size=(9, fs.state_dim))
    particles[:, 0] = 1.0
    design_t = fs.design(params=params)[0]
    y_t = data.to_numpy()[0] * compiled.model.transform_signs
    actual = fs.observation_logweights(y_t, particles, design_t, params)
    eta = particles @ design_t.T
    expected = np.zeros(particles.shape[0])
    family_count = {"gaussian": 0, "gev": 0}
    for index, channel in enumerate(compiled.model.channels):
        family_count[channel.family] += 1
        expected += channel.observation.logpdf(
            y_t[index],
            eta[:, index],
            sigma=params[f"sigma.{channel.name}"],
            xi=params.get(f"xi.{channel.name}"),
        )
    finite = np.isfinite(actual) & np.isfinite(expected)
    error = (
        float(np.max(np.abs(actual[finite] - expected[finite])))
        if np.any(finite)
        else np.inf
    )
    same_support = bool(np.array_equal(np.isfinite(actual), np.isfinite(expected)))
    return {
        "particles": int(particles.shape[0]),
        "family_count": family_count,
        "finite_particles": int(np.sum(finite)),
        "same_support": same_support,
        "max_abs_logweight_error": error,
        "passed": bool(
            family_count == {"gaussian": 2, "gev": 4}
            and same_support
            and error < 1e-10
        ),
    }


def uccle_fs_pgas_check(data, model) -> tuple[dict[str, object], bx.FitResult]:
    started = time.perf_counter()
    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        priors="regularized_horseshoe",
        asis=True,
        mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=103),
        # Twenty-four particles can leave this six-channel stress chain fixed
        # after harmless RNG-sequence changes in other sampler blocks. Sixty-
        # four is still intentionally small, but makes path movement a useful
        # deterministic release gate for this fixed seed.
        particles=bx.Particles(n=64, proposal="guided"),
    )
    metrics = fit.sampler_diagnostics["draw_metrics"]
    minimum_ess = np.asarray(metrics["particle_min_ess"], dtype=float)
    changed = np.asarray(metrics["particle_path_changed"], dtype=float)
    expected_local = {
        f"horseshoe.local.channel.{channel}.level"
        for channel in fit.channel_names
    }
    present_local = {
        name for name in fit.parameter_draws if name.startswith("horseshoe.local.")
    }
    passed = bool(
        fit.plan.targets_exact_posterior
        and fit.plan.parameterization == "fruehwirth_schnatter"
        and np.all(np.isfinite(fit.state_draws))
        and np.all(np.isfinite(fit.log_posterior))
        and np.all(np.isfinite(minimum_ess))
        and np.mean(changed) > 0.0
        and present_local == expected_local
        and fit.auxiliary_draws["fs_state"].shape == (1, 3, 25, 76)
        and np.allclose(fit.parameter("initial.factor.common.slope"), 0.0)
        and fit.meta["loading_kernels"]["loading.common.TNm"]
        == "collapsed_gaussian_ffbs"
        and fit.meta["loading_kernels"]["loading.common.TXx"].startswith(
            "predictor_preserving_interweave"
        )
    )
    return (
        {
            "plan": fit.plan.to_dict(),
            "semantic_state_shape": list(fit.state_draws.shape),
            "fs_state_shape": list(fit.auxiliary_draws["fs_state"].shape),
            "finite_log_posterior": bool(np.all(np.isfinite(fit.log_posterior))),
            "particle_min_ess": minimum_ess.reshape(-1).tolist(),
            "particle_path_change_rate": float(np.nanmean(changed)),
            "horseshoe_local_parameters": sorted(present_local),
            "loading_kernels": fit.meta["loading_kernels"],
            "fixed_initial_factor_slopes": fit.meta[
                "fixed_initial_factor_slopes"
            ],
            "elapsed_seconds": float(time.perf_counter() - started),
            "passed": passed,
        },
        fit,
    )


def disturbance_check() -> dict[str, object]:
    rng = np.random.default_rng(104)
    model = _two_channel_model(mixed=False)
    y = rng.normal(size=(16, 2))
    fit = bx.fit(
        y,
        model,
        parameterization="disturbance",
        priors="regularized_horseshoe",
        asis=True,
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=105),
    )
    passed = bool(
        fit.plan.parameterization == "disturbance"
        and fit.meta["regularized_horseshoe"]
        and set(fit.meta["horseshoe_processes"])
        == {"channel.bulk.level", "channel.tail.level"}
        and np.all(np.isfinite(fit.state_draws))
        and np.all(np.isfinite(fit.log_posterior))
    )
    return {
        "plan": fit.plan.to_dict(),
        "state_shape": list(fit.state_draws.shape),
        "horseshoe_processes": fit.meta["horseshoe_processes"],
        "finite": bool(
            np.all(np.isfinite(fit.state_draws))
            and np.all(np.isfinite(fit.log_posterior))
        ),
        "passed": passed,
    }


def results_and_archive_check(fit: bx.FitResult) -> dict[str, object]:
    factor = fit.factor("common")
    reconstructed = fit.reconstructed_state("TXx")
    rate = fit.factor_rate_summary("common")
    probabilities = fit.factor_probabilities("common")
    loading_probability = fit.loading_probability(
        "common", "TXx", threshold=1.0
    )
    normalized = fit.normalized_factor(slice(0, 12), "common")
    decomposition = fit.channel_decomposition(
        "TXx", "common", baseline=slice(0, 12)
    )
    decomposition_error = float(decomposition["reconstruction_error"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "factor-v2.1.bucex"
        fit.save(path)
        restored = bx.FitResult.load(path)
        state_error = float(
            np.max(np.abs(restored.state_draws - fit.state_draws))
        )
        fs_error = float(
            np.max(
                np.abs(
                    restored.auxiliary_draws["fs_state"]
                    - fit.auxiliary_draws["fs_state"]
                )
            )
        )
        schema = restored.schema_version
    passed = bool(
        factor.shape == (fit.n_draws, fit.n_time)
        and reconstructed.shape == (fit.n_draws, fit.n_time)
        and rate["factor"] == "common"
        and 0.0 <= loading_probability <= 1.0
        and set(probabilities["loading_positive"]) == set(fit.channel_names)
        and state_error == 0.0
        and fs_error == 0.0
        and schema == "2.1"
        and np.max(np.abs(np.mean(normalized[:, :12], axis=1))) < 1e-10
        and decomposition_error < 1e-10
    )
    return {
        "factor_shape": list(factor.shape),
        "reconstructed_TXx_shape": list(reconstructed.shape),
        "factor_rate_summary": rate,
        "TXx_loading_above_one_probability": loading_probability,
        "factor_probabilities": probabilities,
        "normalized_factor_baseline_max_abs_mean": float(
            np.max(np.abs(np.mean(normalized[:, :12], axis=1)))
        ),
        "decomposition_max_abs_error": decomposition_error,
        "archive_schema": schema,
        "archive_state_max_abs_error": state_error,
        "archive_fs_state_max_abs_error": fs_error,
        "passed": passed,
    }


def univariate_compatibility_check() -> dict[str, object]:
    rng = np.random.default_rng(106)
    y = rng.normal(size=20)
    fit = bx.fit(
        y,
        bx.Model(bx.Gaussian(), (bx.LocalLinearTrend(),)),
        parameterization="fruehwirth_schnatter",
        priors="normal",
        mcmc=bx.MCMC(draws=1, warmup=0, chains=1, seed=107),
    )
    return {
        "result_type": type(fit).__name__,
        "plan": fit.plan.to_dict(),
        "state_shape": list(fit.state_draws.shape),
        "passed": bool(
            type(fit) is bx.FitResult
            and fit.plan.parameterization == "fruehwirth_schnatter"
            and np.all(np.isfinite(fit.state_draws))
        ),
    }


def run(data_dir: str | Path) -> dict[str, object]:
    if bx.__version__ != "2.1.5":
        raise RuntimeError(f"Expected bucex 2.1.5, found {bx.__version__}.")
    started = time.perf_counter()
    graph, data, compiled = uccle_graph_check(data_dir)
    algebra = fs_algebra_check(compiled)
    weights = six_channel_weight_check(data, compiled)
    pgas, fit = uccle_fs_pgas_check(data, compiled.model)
    disturbance = disturbance_check()
    results = results_and_archive_check(fit)
    univariate = univariate_compatibility_check()
    sections = {
        "uccle_one_factor_graph": graph,
        "fs_centered_algebra": algebra,
        "six_channel_mixed_weight": weights,
        "uccle_fs_pgas": pgas,
        "disturbance_horseshoe": disturbance,
        "results_and_archive": results,
        "univariate_compatibility": univariate,
    }
    return {
        "release": "bucex 2.1.5",
        "version": bx.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
        "all_passed": bool(
            all(section["passed"] for section in sections.values())
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/factor_validation_2.1.5.json"),
    )
    args = parser.parse_args()
    result = run(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
