from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bucex as bx


def test_v262_version_and_workflow_removal():
    assert bx.__version__ == "2.6.2"
    assert not hasattr(bx, "make_structural_scenarios")
    assert not hasattr(bx, "PresentationWorkflow")


def test_loess_smooth_is_robust_and_preserves_missing_positions():
    x = np.linspace(0.0, 10.0, 81)
    truth = 2.0 + 0.5 * x
    y = truth.copy()
    y[40] += 30.0
    y[7] = np.nan
    smooth = bx.loess_smooth(x, y, fraction=0.25, robust_iterations=2)
    assert np.isnan(smooth[7])
    assert np.nanmedian(np.abs(smooth - truth)) < 0.05
    assert abs(smooth[40] - truth[40]) < 0.5


def test_matched_gev_models_keep_the_same_latent_path():
    models = [
        bx.Model(bx.GEV(), (bx.LocalLevel(mode="dynamic"),), name=f"xi={xi}")
        for xi in (-0.3, 0.0, 0.3)
    ]
    simulations = [
        bx.simulate(
            model,
            48,
            {"sigma": 1.5, "xi": xi, "sd.level": 0.08},
            initial_state=[25.0],
            seed=2601,
        )
        for model, xi in zip(models, (-0.3, 0.0, 0.3))
    ]
    for simulation in simulations[1:]:
        np.testing.assert_allclose(simulation.eta, simulations[0].eta)


def test_phase_specific_season_plot_uses_level_plus_current_season():
    model = bx.Model(
        bx.Gaussian(),
        (
            bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            bx.DummySeasonal(period=4, mode="dynamic"),
        ),
    )
    simulation = bx.simulate(
        model,
        24,
        {"sigma": 0.25, "sd.level": 0.03, "sd.slope": 0.002, "sd.seasonal": 0.02},
        initial_state=[2.0, 0.01, -0.2, 0.0, 0.2],
        seed=262,
    )
    fit = bx.fit(
        simulation.y,
        model=model,
        priors="normal",
        engine="ffbs",
        parameterization="centered",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=263),
    )
    figure, axis = fit.plot(
        "season",
        labels=("one", "two", "three", "four"),
        show_interval=False,
    )
    assert len(axis.lines) == 4
    assert [line.get_label() for line in axis.lines] == ["one", "two", "three", "four"]
    assert all(len(line.get_xdata()) == 6 for line in axis.lines)
    plt.close(figure)


def test_laplace_fit_is_a_full_path_warm_start_for_pgas():
    model = bx.Model(
        bx.GEV(),
        (
            bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            bx.DummySeasonal(period=4, mode="dynamic"),
        ),
    )
    simulation = bx.simulate(
        model,
        24,
        {"sigma": 1.0, "xi": -0.2, "sd.level": 0.04, "sd.slope": 0.001, "sd.seasonal": 0.03},
        initial_state=[20.0, 0.0, -0.2, 0.0, 0.2],
        seed=264,
    )
    priors = bx.ssvs_gev_priors(period=4, alpha_mean=20.0)
    laplace = bx.fit(
        simulation.y,
        model=model,
        priors=priors,
        engine="laplace",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=265),
    )
    pgas = bx.fit(
        simulation.y,
        model=model,
        priors=laplace.priors,
        engine="pgas",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=266),
        particles=bx.Particles(n=16, proposal="guided"),
        init=laplace,
    )
    warm = laplace.warm_start()
    assert warm["__centered_path__"].shape == (laplace.n_time + 1, 5)
    assert warm["__warm_start__"]["source_engine"] == "laplace"
    assert pgas.meta["warm_start_source_engine"] == "laplace"
    assert not laplace.plan.targets_exact_posterior
    assert pgas.plan.targets_exact_posterior


def test_hpc_surface_matches_the_seven_examples():
    directory = Path(__file__).resolve().parents[1] / "examples" / "job_scripts"
    assert {path.name for path in directory.glob("*.pbs")} == {
        "00_uccle_record.pbs",
        "01_tail_simulations.pbs",
        "02_structural_simulations.pbs",
        "03_simulation_laplace.pbs",
        "04_simulation_pgas.pbs",
        "05_uccle_laplace.pbs",
        "06_uccle_pgas.pbs",
    }
