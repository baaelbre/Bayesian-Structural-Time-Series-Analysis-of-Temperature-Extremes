from __future__ import annotations

import numpy as np
import pytest

import bucex as bx
from bucex.inference.fit._progress import mcmc_progress_line


def _identified_model() -> bx.FactorModel:
    return bx.FactorModel(
        channels=(
            bx.Channel("reference", bx.Gaussian(), (bx.LocalLevel(),)),
            bx.Channel("response", bx.Gaussian(), (bx.LocalLevel(),)),
        ),
        factors=(
            bx.Factor(
                "common",
                (
                    bx.LocalLinearTrend(
                        initial_level=0.0,
                        initial_level_sd=0.0,
                        initial_slope=0.0,
                        initial_slope_sd=0.0,
                    ),
                ),
                {
                    "reference": 1.0,
                    "response": bx.Loading.estimated(0.8),
                },
            ),
        ),
    )


def test_uniform_progress_line_has_common_fields_and_current_parameters():
    line = mcmc_progress_line(
        label="univariate",
        engine="pgas",
        chain=2,
        chains=4,
        completed=10,
        total=500,
        warmup=250,
        saved=0,
        draws=250,
        elapsed=2.0,
        parameters={
            "sigma": 0.8044,
            "xi": -0.1355,
            "Q_level": 0.000603,
            "hs_global": 0.973,
        },
        metrics={
            "particle_min_ess": 66.5,
            "particle_changed_fraction": 0.99,
        },
        particles=128,
    )
    assert "chain 2/4" in line
    assert "it 10/500" in line
    assert "sigma=0.8044" in line
    assert "xi=-0.1355" in line
    assert "Q_level=0.000603" in line
    assert "particle_min_ess=66.5/128" in line
    assert "path_change=0.99" in line
    assert "ETA " in line


@pytest.mark.parametrize(
    ("alias", "profile"),
    (
        (" regularized horseshoe ", "regularized_horseshoe"),
        ("regularised-horseshoe", "regularized_horseshoe"),
        ("horseshoe", "regularized_horseshoe"),
        ("pc", "regularized"),
        ("normal", "half_normal"),
    ),
)
def test_factor_prior_profile_aliases(alias, profile):
    model = _identified_model()
    compiled = bx.compile_model(model, np.zeros((12, 2)))
    assert bx.default_factor_priors(compiled, alias).profile == profile


def test_identified_factor_prior_helper_fixes_requested_processes():
    model = _identified_model()
    compiled = bx.compile_model(model, np.zeros((12, 2)))
    priors = bx.identified_factor_priors(
        compiled,
        profile="regularized horseshoe",
        smooth_factor=True,
        reference_channel="reference",
    )
    assert isinstance(priors.process["factor.common.level"], bx.FixedSD)
    assert isinstance(priors.process["channel.reference.level"], bx.FixedSD)
    assert "factor.common.level" not in priors.horseshoe_processes
    assert "channel.reference.level" not in priors.horseshoe_processes
    assert priors.metadata["pure_reference_channel"] == "reference"

    loading_only = bx.identified_factor_priors(
        compiled,
        fixed_idiosyncratic="all",
    )
    assert loading_only.horseshoe is None
    assert loading_only.horseshoe_processes == ()
    assert all(
        isinstance(loading_only.process[f"channel.{channel}.level"], bx.FixedSD)
        for channel in model.channel_names
    )


def test_invalid_factor_prior_reports_the_received_profile():
    model = _identified_model()
    compiled = bx.compile_model(model, np.zeros((12, 2)))
    with pytest.raises(ValueError, match="'manuscript_lasso'.*Univariate-only"):
        bx.default_factor_priors(compiled, "manuscript_lasso")


def test_real_samplers_share_progress_vocabulary(capsys):
    gaussian_model = bx.Model(bx.Gaussian(), (bx.LocalLinearTrend(),))
    gaussian_truth = {
        "sd.level": 0.02,
        "sd.slope": 0.002,
        "sigma": 0.20,
    }
    gaussian_data = bx.simulate(
        gaussian_model, 12, gaussian_truth, seed=210
    ).y
    bx.fit(
        gaussian_data,
        gaussian_model,
        parameterization="centered",
        engine="ffbs",
        priors="normal",
        mcmc=bx.MCMC(
            draws=1,
            warmup=0,
            chains=1,
            seed=209,
            progress=True,
            progress_every=1,
        ),
    )
    centered_output = capsys.readouterr().out
    assert "[univariate | FFBS | chain 1/1]" in centered_output
    assert "it 1/1" in centered_output
    assert "sigma=" in centered_output
    assert "Q_level=" in centered_output
    assert "ETA " in centered_output

    bx.fit(
        gaussian_data,
        gaussian_model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",
        priors="normal",
        mcmc=bx.MCMC(
            draws=1,
            warmup=0,
            chains=1,
            seed=211,
            progress=True,
            progress_every=1,
        ),
    )
    gaussian_output = capsys.readouterr().out
    assert "[univariate | FFBS | chain 1/1]" in gaussian_output
    assert "it 1/1" in gaussian_output
    assert "sigma=" in gaussian_output
    assert "Q_level=" in gaussian_output
    assert "ETA " in gaussian_output

    gev_model = bx.Model(bx.GEV(), (bx.LocalLinearTrend(),))
    gev_truth = {
        "sd.level": 0.02,
        "sd.slope": 0.002,
        "sigma": 0.40,
        "xi": -0.08,
    }
    gev_data = bx.simulate(gev_model, 12, gev_truth, seed=212).y
    bx.fit(
        gev_data,
        gev_model,
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        priors="normal",
        mcmc=bx.MCMC(
            draws=1,
            warmup=0,
            chains=1,
            seed=213,
            progress=True,
            progress_every=1,
        ),
        particles=bx.Particles(n=12),
    )
    gev_output = capsys.readouterr().out
    assert "[univariate | PGAS | chain 1/1]" in gev_output
    assert "it 1/1" in gev_output
    assert "sigma=" in gev_output and "xi=" in gev_output
    assert "particle_min_ess=" in gev_output
    assert "path_change=" in gev_output
    assert "ETA " in gev_output

    factor_model = _identified_model()
    factor_compiled = bx.compile_model(factor_model, np.zeros((12, 2)))
    factor_priors = bx.identified_factor_priors(
        factor_compiled, reference_channel="reference"
    )
    bx.fit(
        np.zeros((12, 2)),
        factor_model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",
        priors=factor_priors,
        mcmc=bx.MCMC(
            draws=1,
            warmup=0,
            chains=1,
            seed=216,
            progress=True,
            progress_every=1,
        ),
    )
    factor_output = capsys.readouterr().out
    assert "[factor FS | FFBS | chain 1/1]" in factor_output
    assert "it 1/1" in factor_output
    assert "sigma=(" in factor_output
    assert "Q_factor=(" in factor_output
    assert "Q_idio=(" in factor_output
    assert "loading=(" in factor_output
    assert "ETA " in factor_output


def test_factor_identification_diagnostics_and_plot_api():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = _identified_model()
    truth = {
        "sd.factor.common.level": 0.0,
        "sd.factor.common.slope": 0.002,
        "sd.channel.reference.level": 0.0,
        "sd.channel.response.level": 0.01,
        "sigma.reference": 0.10,
        "sigma.response": 0.12,
        "loading.common.response": 0.8,
    }
    simulation = bx.simulate(model, 18, truth, seed=214)
    compiled = bx.compile_model(model, simulation.y)
    priors = bx.identified_factor_priors(
        compiled,
        reference_channel="reference",
    )
    fit = bx.fit(
        simulation.y,
        model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",
        priors=priors,
        asis=True,
        mcmc=bx.MCMC(draws=4, warmup=3, chains=2, seed=215),
    )

    innovations = fit.idiosyncratic_innovation_draws("response")
    assert innovations.shape == (fit.n_draws, fit.n_time - 1)
    paired = fit.loading_deviation_draws(
        "response", summary="factor_projection"
    )
    assert paired["loading"].shape == (fit.n_draws,)
    assert paired["deviation_summary"].shape == (fit.n_draws,)
    diagnostics = fit.factor_identification_diagnostics()
    assert set(diagnostics.index) == {"reference", "response"}
    assert bool(diagnostics.loc["reference", "loading_fixed"])

    figures = [
        fit.plot("factor_decomposition"),
        fit.plot(
            "parameter_density",
            parameters="loading.common.response",
            truths=truth,
        ),
        fit.plot("traces", truths=truth),
        fit.plot("loading_deviation", channel="response"),
        fit.plot("identification"),
        fit.plot("idiosyncratic_innovations", channel="response"),
    ]
    assert all(figure is not None for figure, _ in figures)
    for figure, _ in figures:
        plt.close(figure)
