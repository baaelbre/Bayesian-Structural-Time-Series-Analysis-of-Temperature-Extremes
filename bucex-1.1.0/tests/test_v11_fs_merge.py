from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import numpy as np

import bucex as bx


def _seasonal_sample(family: str, seed: int = 10):
    rng = np.random.default_rng(seed)
    mean = 8.0 + np.tile(np.sin(2.0 * np.pi * np.arange(12) / 12.0), 2)
    if family == "gaussian":
        return mean + rng.normal(scale=0.25, size=24)
    return mean + rng.gumbel(scale=0.55, size=24)


def test_regularized_horseshoe_is_a_distinct_fs_profile():
    priors = bx.regularized_horseshoe_gaussian_priors()
    assert priors.horseshoe is not None
    assert priors.lasso is None
    fit = bx.fit_bayes(
        _seasonal_sample("gaussian"),
        family="gaussian",
        priors="horseshoe",
        n_iter=5,
        burn=2,
        seed=11,
        asis=True,
        progress=False,
    )
    assert fit.meta["parameterization"] == "fruehwirth_schnatter"
    assert fit.meta["regularized_horseshoe"]
    assert "horseshoe_global" in fit.draws_static


def test_pc_profile_has_declared_tail_probability_and_exact_mixture_draws():
    prior = bx.PCInnovationPrior(alpha=0.05)
    assert np.isclose(
        np.exp(-prior.standardized_rate_for("level")), 0.05
    )
    fit = bx.fit_bayes(
        _seasonal_sample("gaussian", 12),
        family="gaussian",
        priors="pc",
        n_iter=5,
        burn=2,
        seed=13,
        progress=False,
    )
    assert fit.meta["pc_innovation_prior"]
    assert "pc_tau_level" in fit.draws_static


def test_fs_pgas_retains_log_weights_and_reports_exact_target():
    fit = bx.fit_bayes(
        _seasonal_sample("gev", 14),
        family="gev",
        priors="horseshoe",
        state_method="pgas",
        state_kwargs={"particles": 20},
        n_iter=4,
        burn=2,
        seed=15,
        progress=False,
    )
    assert fit.plan.targets_exact_posterior
    diagnostics = fit.diagnostics()["engine"]
    assert np.isfinite(diagnostics["median_min_particle_ess"])
    assert fit.meta["restored_iterations"] == 0


def test_fs_multichain_safe_archive_and_forecast():
    fit = bx.fit_bayes(
        _seasonal_sample("gaussian", 16),
        family="gaussian",
        priors="regularized_lasso",
        n_iter=5,
        burn=2,
        chains=2,
        seed=17,
        progress=False,
    )
    assert fit.state_draws.shape == (2, 3, 25, 13)
    forecast = fit.forecast(2, draws=4, seed=18)
    assert forecast.observations.shape == (4, 2)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fit.bucex"
        fit.save(path)
        restored = bx.PosteriorBundle.load(path)
        np.testing.assert_allclose(restored.draws_states, fit.draws_states)
        with zipfile.ZipFile(path) as archive:
            metadata = archive.read("metadata.json")
            arrays = bytearray(archive.read("arrays.npz"))
        arrays[-1] ^= 1
        broken = Path(directory) / "broken.bucex"
        with zipfile.ZipFile(broken, "w") as archive:
            archive.writestr("metadata.json", metadata)
            archive.writestr("arrays.npz", arrays)
        try:
            bx.PosteriorBundle.load(broken)
        except ValueError as error:
            assert "integrity" in str(error).lower()
        else:
            raise AssertionError("Tampering must be detected.")

