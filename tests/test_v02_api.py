from __future__ import annotations

import numpy as np

from bucex import (
    fit_bayes,
    fit_uccle_series,
    manuscript_gaussian_priors,
    manuscript_gev_priors,
)


def test_manuscript_priors_include_lasso():
    gp = manuscript_gaussian_priors()
    ep = manuscript_gev_priors()
    assert gp.lasso is not None
    assert gp.lasso.variance_mode == "observation"
    assert ep.lasso is not None
    assert ep.lasso.variance_mode == "fixed"
    assert ep.sigma2.a == 2.0
    assert ep.xi.lower == -0.5
    assert ep.xi.upper == 0.5


def test_high_level_gaussian_fit_stores_lasso_draws():
    rng = np.random.default_rng(1)
    t = np.arange(60)
    y = 10.0 + 0.01 * t + 2.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.4, t.size)
    fit = fit_bayes(
        y,
        family="gaussian",
        period=12,
        n_iter=8,
        burn=3,
        progress=False,
        seed=4,
    )
    assert fit.draws_states.shape == (5, 61, 13)
    for key in ("tau_level", "tau_trend", "tau_season", "lambda2"):
        assert key in fit.draws_static
        assert np.all(np.isfinite(fit.draws_static[key]))
        assert np.all(fit.draws_static[key] > 0)


def test_uccle_gev_minimum_transform_and_risk():
    fit = fit_uccle_series(
        "TXn",
        data_dir="data",
        start="2000-01-01",
        end="2005-12-31",
        n_iter=7,
        burn=3,
        progress=False,
        seed=5,
    )
    assert fit.transform_sign == -1.0
    p, years = fit.exceedance_probability_draws(0.0, annual=True)
    assert p.shape[1] == len(years)
    assert np.all((0.0 <= p) & (p <= 1.0))
    assert "< 0" in fit.event_label(0.0)
    endpoint = fit.endpoint_draws()
    assert endpoint.shape == (fit.n_draws, fit.n_time)


def test_plot_api_runs(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    fit = fit_uccle_series(
        "TXx",
        data_dir="data",
        start="2000-01-01",
        end="2005-12-31",
        n_iter=7,
        burn=3,
        progress=False,
        seed=6,
    )
    for kind, kwargs in (
        ("level", {}),
        ("slope", {}),
        ("level_slope", {}),
        ("exceedance", {"threshold": 35.0, "annual": True}),
        ("return_period", {"threshold": 35.0, "annual": True}),
        ("endpoint", {}),
    ):
        fig, _ = fit.plot(type=kind, **kwargs)
        fig.savefig(tmp_path / f"{kind}.png")
