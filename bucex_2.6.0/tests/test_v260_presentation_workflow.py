from __future__ import annotations

import json

import numpy as np

import bucex as bx


def test_v260_scenario_catalog_separates_tail_and_structure_questions():
    assert bx.__version__ == "2.6.0"
    assert [scenario.xi for scenario in bx.TAIL_SCENARIOS] == [0.20, 0.0, -0.20]
    assert {scenario.sigma for scenario in bx.STRUCTURAL_SCENARIOS} == {1.5}
    assert {scenario.xi for scenario in bx.STRUCTURAL_SCENARIOS} == {-0.20}
    assert {scenario.name for scenario in bx.STRUCTURAL_SCENARIOS} == {
        "stationary",
        "linear_trend",
        "local_level",
        "local_level_fixed_season",
        "local_level_dynamic_season",
        "stochastic_trend_fixed_season",
        "full_dynamic",
    }


def test_componentwise_presentation_prior_has_readable_model_space():
    prior = bx.presentation_gev_prior(alpha_mean=25.0).ssvs
    assert prior.level_dynamic_probability == 0.5
    np.testing.assert_allclose(prior.trend_probabilities, (1.0 / 3.0,) * 3)
    np.testing.assert_allclose(prior.season_probabilities, (1.0 / 3.0,) * 3)
    assert prior.innovation_slab_sd == {
        "level": 0.12,
        "trend": 0.0015,
        "season": 0.10,
    }


def test_runtime_profiles_and_v260_paths_are_reproducible(tmp_path):
    config = bx.PresentationConfig.for_profile(
        "smoke", output_dir=tmp_path / "presentation", progress=False
    )
    assert config.start == "2015-01-01"
    assert config.runtime.particles == 24
    assert config.runtime.simulation_months == 48
    paths = bx.WorkflowPaths.from_config(config)
    assert paths.simulation_fit("pgas", "stationary", 2).name == "chain_02.bucex"
    assert paths.simulation_fit("pgas", "stationary").name == "combined.bucex"
    assert paths.uccle_fit("laplace", "TXx").parts[-4:] == (
        "uccle",
        "laplace",
        "TXx",
        "combined.bucex",
    )

    workflow = bx.PresentationWorkflow(config)
    local = workflow._mcmc(chain=None, seed_offset=100)
    expected = (
        [local.seed]
        if local.chains == 1
        else [
            int(stream.generate_state(1)[0])
            for stream in np.random.SeedSequence(local.seed).spawn(local.chains)
        ]
    )
    submitted = [
        workflow._mcmc(chain=index + 1, seed_offset=100).seed
        for index in range(local.chains)
    ]
    assert submitted == expected


def test_laplace_fit_is_a_full_path_warm_start_for_pgas(tmp_path):
    config = bx.PresentationConfig.for_profile(
        "smoke",
        output_dir=tmp_path / "run",
        progress=False,
        draws=1,
        warmup=1,
        particles=16,
    )
    workflow = bx.PresentationWorkflow(config)
    workflow.run_simulations(
        kind="structure", scenario="stationary", figures=False
    )
    workflow.fit_simulations(
        engine="laplace", scenario="stationary", figures=False
    )
    workflow.fit_simulations(
        engine="pgas", scenario="stationary", figures=False
    )

    laplace_path = workflow.paths.simulation_fit("laplace", "stationary")
    pgas_path = workflow.paths.simulation_fit("pgas", "stationary")
    laplace = bx.FitResult.load(laplace_path)
    pgas = bx.FitResult.load(pgas_path)
    warm = laplace.warm_start()

    assert warm["__centered_path__"].shape == (laplace.n_time + 1, 13)
    assert warm["__warm_start__"]["source_engine"] == "laplace"
    assert pgas.meta["warm_start_source_engine"] == "laplace"
    assert not laplace.plan.targets_exact_posterior
    assert pgas.plan.targets_exact_posterior
    assert "reference_ancestor_change_rate" in pgas.diagnostics()["engine"]
    assert pgas.component_probabilities().shape == (3, 3)

    manifest = json.loads(workflow.paths.manifest.read_text(encoding="utf-8"))
    required = {
        "simulations:structure:stationary",
        "simulation-fit:laplace:stationary:combined",
        "simulation-fit:pgas:stationary:combined",
    }
    assert required <= set(manifest["tasks"])
