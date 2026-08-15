from __future__ import annotations

import numpy as np
import pytest

import bucex as bx
from bucex.inference.fit.factor_fs import (
    CompiledFactorFS,
    _initial_fs_parameters,
)
from bucex.inference.fit.factor_loading import (
    _RegressionRandomWalk,
    loading_deviation_interweave_sweep,
)
from bucex.inference.state.kalman import ffbs


def _factor_model(*, mixed: bool = False, fixed_slope: float = 0.0):
    return bx.FactorModel(
        channels=(
            bx.Channel("reference", bx.Gaussian(), (bx.LocalLevel(),)),
            bx.Channel(
                "response",
                bx.GEV() if mixed else bx.Gaussian(),
                (bx.LocalLevel(),),
            ),
        ),
        factors=(
            bx.Factor(
                "common",
                (
                    bx.LocalLinearTrend(
                        initial_level=0.0,
                        initial_level_sd=0.0,
                        initial_slope=fixed_slope,
                        initial_slope_sd=0.0,
                    ),
                ),
                {
                    "reference": 1.0,
                    "response": bx.Loading.estimated(
                        0.5, mean=0.0, sd=10.0
                    ),
                },
            ),
        ),
    )


def test_fixed_initial_factor_slope_is_not_an_mcmc_target():
    model = _factor_model(fixed_slope=0.025)
    y = np.random.default_rng(100).normal(size=(18, 2))
    fit = bx.fit(
        y,
        model,
        parameterization="fs",
        engine="ffbs",
        mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=101),
    )
    np.testing.assert_allclose(
        fit.parameter("initial.factor.common.slope"), 0.025
    )
    assert "initial.factor.common.slope" not in fit.sampler_diagnostics[
        "acceptance"
    ]
    assert fit.meta["fixed_initial_factor_slopes"] == {"common": 0.025}

    with pytest.raises(ValueError, match="is fixed at"):
        bx.fit(
            y,
            model,
            parameterization="fs",
            engine="ffbs",
            init={"initial.factor.common.slope": 0.0},
            mcmc=bx.MCMC(draws=1, warmup=0, chains=1, seed=102),
        )


def test_collapsed_regression_random_walk_recovers_static_coefficients():
    factor = np.linspace(-1.0, 1.0, 80)
    y = 1.25 + 1.8 * factor
    model = _RegressionRandomWalk(
        factor=factor,
        deviation_sd=0.0,
        intercept_mean=0.0,
        intercept_sd=10.0,
        loading_mean=0.0,
        loading_sd=10.0,
    )
    path, result = ffbs(
        y,
        model,
        {},
        np.random.default_rng(103),
        observation_variance=1e-6,
    )
    assert np.isfinite(result.log_likelihood)
    assert path[0, 0] == pytest.approx(1.25, abs=0.002)
    assert path[0, 1] == pytest.approx(1.8, abs=0.002)
    np.testing.assert_allclose(path[:, 0], path[0, 0], atol=1e-10)
    np.testing.assert_allclose(path[:, 1], path[0, 1], atol=1e-10)
    np.testing.assert_allclose(path[:, 2], 0.0)


def test_gaussian_factor_fit_uses_collapsed_loading_block():
    model = _factor_model()
    y = np.random.default_rng(104).normal(size=(20, 2))
    fit = bx.fit(
        y,
        model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",
        mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=105),
    )
    key = "loading.common.response"
    assert fit.sampler_diagnostics["loading_kernels"][key] == (
        "collapsed_gaussian_ffbs"
    )
    assert fit.meta["loading_kernels"][key] == "collapsed_gaussian_ffbs"
    assert fit.sampler_diagnostics["collapsed_gaussian_processes"] == [
        "channel.response.level"
    ]
    assert fit.sampler_diagnostics["acceptance"][key][0] == 1.0
    assert np.isfinite(
        fit.sampler_diagnostics["acceptance"][
            "signed_sd.channel.response.level"
        ][0]
    )
    assert "intercept.response" not in fit.sampler_diagnostics["acceptance"]
    assert np.all(np.isfinite(fit.reconstructed_state("response")))
    normalized = fit.normalized_factor(slice(0, 5))
    np.testing.assert_allclose(np.mean(normalized[:, :5], axis=1), 0.0, atol=1e-12)
    decomposition = fit.channel_decomposition(
        "response", baseline=slice(0, 5)
    )
    np.testing.assert_allclose(
        decomposition["baseline"][:, None]
        + decomposition["shared"]
        + decomposition["deviation"]
        + decomposition["seasonal"],
        decomposition["predictor"],
        atol=1e-12,
    )
    assert float(decomposition["reconstruction_error"]) < 1e-12


class _DeterministicRng:
    def normal(self):
        return 1.0

    def random(self):
        return 0.5


def test_loading_deviation_interweave_preserves_mixed_predictor():
    model = _factor_model(mixed=True)
    y = np.column_stack(
        (
            np.linspace(0.0, 1.0, 12),
            2.0 + np.linspace(0.0, 1.0, 12),
        )
    )
    compiled = bx.compile_model(model, y)
    priors = bx.default_factor_priors(compiled)
    fs = CompiledFactorFS(compiled)
    params = _initial_fs_parameters(
        compiled, fs, priors, np.random.default_rng(106), None
    )
    key = "loading.common.response"
    params[key] = 0.0
    params["signed_sd.factor.common.level"] = 1.0
    params["sd.factor.common.level"] = 1.0
    params["signed_sd.factor.common.slope"] = 0.0
    params["sd.factor.common.slope"] = 0.0
    params["signed_sd.channel.response.level"] = 1.0
    params["sd.channel.response.level"] = 1.0

    ncp = np.zeros((compiled.n_time + 1, fs.state_dim))
    ncp[:, 0] = 1.0
    factor_block = next(item for item in fs.blocks if item.kind == "factor")
    ncp[:, factor_block.ncp_slice.start] = np.arange(compiled.n_time + 1)
    centered = fs.to_centered(ncp, params)
    channel_block = next(
        item
        for item in fs.blocks
        if item.kind == "channel" and item.name == "response"
    )
    factor_level = centered[:, factor_block.centered_slice][
        :, int(factor_block.layout.idx_alpha)
    ]
    centered[:, channel_block.centered_slice][
        :, int(channel_block.layout.idx_alpha)
    ] = 0.5 * factor_level
    ncp = fs.from_centered(centered, params)
    eta_before = fs.eta(ncp, params=params)

    updated, outcomes, _, handled = loading_deviation_interweave_sweep(
        ncp,
        compiled,
        fs,
        params,
        {key: 0.5},
        _DeterministicRng(),
        keys=(key,),
        adapt=False,
        iteration=0,
    )
    assert handled == (key,)
    assert outcomes[key]
    assert params[key] == pytest.approx(0.5)
    np.testing.assert_allclose(
        fs.eta(updated, params=params), eta_before, atol=1e-12
    )


def test_initial_state_standard_deviations_must_be_non_negative():
    with pytest.raises(ValueError, match="initial_slope_sd"):
        bx.LocalLinearTrend(initial_slope_sd=-1.0)
    with pytest.raises(ValueError, match="initial_level_sd"):
        bx.LocalLinearTrend(initial_level_sd=-1.0)
