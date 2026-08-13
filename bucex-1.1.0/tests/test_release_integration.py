from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

import bucex as bx
from bucex.laplace import state_log_density
from bucex.numerics import gaussian_support


def test_release_version_and_bundled_uccle_integrity():
    assert bx.__version__ == "1.1.0"
    table = bx.validate_uccle_data()
    assert tuple(table.index) == bx.UCCLE_SERIES
    assert np.all(table["n"].to_numpy() == 1572)
    assert set(table["start"].dt.year) == {1892}
    assert set(table["end"].dt.year) == {2022}


def test_monthly_uccle_files_reproduce_from_daily_source():
    table = bx.validate_uccle_data("data", check_daily=True)
    assert np.all(table["daily_mismatches"].to_numpy() == 0)
    assert np.max(table["daily_max_abs_difference"].to_numpy()) < 1e-12


def test_small_stochastic_slope_is_not_treated_as_deterministic():
    model = bx.Model(
        bx.Gaussian(),
        [bx.LocalLinearTrend(), bx.DummySeasonal(12)],
    )
    compiled = bx.compile_model(model, np.zeros(24))
    params = {
        "sd.level": 2e-2,
        "sd.slope": 1e-6,
        "sd.seasonal": 1e-1,
        "sigma": 1.0,
    }
    factor = gaussian_support(compiled.transition_cov(params))
    assert factor.active_variances.size == 3
    z = np.zeros((24, compiled.noise_dim))
    z[:, compiled.noise_names.index("slope")] = 1.0
    path = compiled.from_noncentered(
        bx.NonCenteredPath(compiled.initial_mean.copy(), z), params
    )
    assert np.isfinite(state_log_density(path, compiled, params))


def test_full_length_txx_iterated_laplace_smoke():
    fit = bx.fit_uccle_series(
        "TXx",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=123),
        engine="laplace",
        asis=True,
    )
    assert fit.state_draws.shape == (1, 1, 1573, 13)
    assert np.all(np.isfinite(fit.state_draws))
    assert fit.plan.approximation is not None
    assert fit.obs_name == "gev"


def test_fit_is_self_contained_and_records_initial_values():
    fit = bx.fit(
        np.linspace(0.0, 1.0, 20),
        model=bx.Model(bx.Gaussian(), [bx.LocalLevel()]),
        init={"sd.level": 0.02, "sigma": 0.3},
        mcmc=bx.MCMC(draws=2, warmup=2, chains=2, seed=4),
    )
    assert fit.obs is fit.model.observation
    assert fit.config["mcmc"]["chains"] == 2
    assert fit.methods == {"engine": "ffbs", "parameterization": "noncentered", "asis": False}
    assert len(fit.initial_params["parameters_by_chain"]) == 2
    assert all(
        np.isclose(values["sd.level"], 0.02)
        for values in fit.initial_params["parameters_by_chain"]
    )
    assert fit.draws_states.shape == (4, 21, 1)
    assert "laplace_iterations" in fit.draws_aux


def test_safe_roundtrip_preserves_initial_values_and_rejects_tampering():
    fit = bx.fit(
        np.arange(16.0),
        model=bx.Model(bx.Gaussian(), [bx.LocalLevel()]),
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=5),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fit.bucex"
        fit.save(path)
        restored = bx.FitResult.load(path)
        assert restored.initial_values == fit.initial_values

        with zipfile.ZipFile(path, "r") as source:
            metadata = json.loads(source.read("metadata.json"))
            arrays = bytearray(source.read("arrays.npz"))
        arrays[-1] ^= 1
        broken = Path(directory) / "broken.bucex"
        with zipfile.ZipFile(broken, "w") as target:
            target.writestr("metadata.json", json.dumps(metadata))
            target.writestr("arrays.npz", arrays)
        try:
            bx.FitResult.load(broken)
        except ValueError as error:
            assert "integrity" in str(error).lower()
        else:
            raise AssertionError("A modified fit archive must fail its integrity check.")


def test_compact_gibbs_config_and_initial_value_migration():
    config = bx.GibbsConfig(n_iter=5, burn=2, seed=6)
    fit = bx.fit_bayes(
        np.linspace(1.0, 2.0, 24),
        family="gaussian",
        period=12,
        config=config,
        init_params_state={
            "alpha0": 1.0,
            "beta0": 0.0,
            "gamma0_season": np.zeros(11),
            "s_level": 0.02,
            "s_trend": 1e-4,
            "s_season": 0.03,
        },
        init_params_obs={"sigma": 0.2},
    )
    assert fit.n_draws == 3
    assert fit.model.components[0].initial_level == 1.0
    assert fit.model.components[1].initial_mean == tuple(np.zeros(11))
    first = fit.initial_values["parameters_by_chain"][0]
    assert np.isclose(first["sd.slope"], 1e-4)


def test_calendar_rate_and_labeled_risk_compatibility():
    fit = bx.fit_uccle_series(
        "TXx",
        start="2000-01-01",
        end="2002-12-01",
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=7),
    )
    rates = fit.period_rate_summary({"short": (2000, 2002)})
    assert "probability_positive" in rates
    probability, years = fit.exceedance_probability_draws(
        35.0, annual=True, return_labels=True
    )
    periods, period_years = fit.return_period_draws(
        35.0, annual=True, return_labels=True
    )
    assert probability.shape == periods.shape == (fit.n_draws, 3)
    np.testing.assert_array_equal(years, period_years)
    assert fit.event_label(35.0) == "P(TXx > 35)"


def test_custom_initial_seasonal_roundtrips_through_model_dict():
    component = bx.DummySeasonal(period=4, initial_mean=(1.0, -0.5, 0.25))
    model = bx.Model(bx.Gaussian(), [bx.LocalLevel(), component])
    restored = bx.Model.from_dict(model.to_dict())
    assert restored == model
    compiled = bx.compile_model(restored, np.arange(12.0))
    np.testing.assert_allclose(compiled.initial_mean[1:], [1.0, -0.5, 0.25])


def test_independent_fit_combination_preserves_chain_identity():
    y = np.linspace(-1.0, 1.0, 18)
    model = bx.Model(bx.Gaussian(), [bx.LocalLevel()])
    fits = [
        bx.fit(y, model=model, mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=seed))
        for seed in (10, 11)
    ]
    combined = bx.combine_fits(fits)
    assert combined.n_chains == 2
    assert combined.draws_per_chain == 2
    assert combined.sampler_diagnostics["mcmc"]["seeds"] == [10, 11]
    np.testing.assert_allclose(combined.state_draws[0], fits[0].state_draws[0])
    np.testing.assert_allclose(combined.state_draws[1], fits[1].state_draws[0])
