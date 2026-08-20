from __future__ import annotations

import json

import numpy as np

import bucex as bx


def test_v260_scenario_catalog_separates_shape_scale_and_structure_questions():
    assert bx.__version__ == "2.6.1"
    assert [scenario.xi for scenario in bx.TAIL_SCENARIOS] == [-0.30, 0.0, 0.30]
    assert [scenario.sigma for scenario in bx.SCALE_SCENARIOS] == [0.75, 1.5, 3.0]
    assert {scenario.xi for scenario in bx.SCALE_SCENARIOS} == {-0.30}
    assert {scenario.sigma for scenario in bx.STRUCTURAL_SCENARIOS} == {1.5}
    assert {scenario.xi for scenario in bx.STRUCTURAL_SCENARIOS} == {-0.30}
    assert {scenario.period for scenario in bx.STRUCTURAL_SCENARIOS} == {4}
    assert {scenario.n_time for scenario in bx.STRUCTURAL_SCENARIOS} == {800}
    assert {scenario.name for scenario in bx.STRUCTURAL_SCENARIOS} == {
        "stationary",
        "linear_trend",
        "random_walk",
        "local_linear_trend",
        "stationary_dynamic_season",
        "local_linear_trend_fixed_season",
    }
    assert min(
        scenario.sd_level
        for scenario in bx.STRUCTURAL_SCENARIOS
        if scenario.level == "dynamic"
    ) >= 0.035

    tail_tables = [
        bx.simulate_scenario(scenario.resized(48))[1]
        for scenario in bx.TAIL_SCENARIOS
    ]
    scale_tables = [
        bx.simulate_scenario(scenario.resized(48))[1]
        for scenario in bx.SCALE_SCENARIOS
    ]
    for table in tail_tables[1:]:
        np.testing.assert_allclose(table["eta"], tail_tables[0]["eta"])
    for table in scale_tables[1:]:
        np.testing.assert_allclose(table["eta"], scale_tables[0]["eta"])


def test_componentwise_presentation_prior_has_readable_model_space():
    resolved = bx.presentation_gev_prior(alpha_mean=25.0, period=4)
    prior = resolved.ssvs
    assert prior.level_dynamic_probability == 0.5
    np.testing.assert_allclose(prior.trend_probabilities, (1.0 / 3.0,) * 3)
    np.testing.assert_allclose(prior.season_probabilities, (1.0 / 3.0,) * 3)
    assert prior.innovation_slab_sd == {
        "level": 0.10,
        "trend": 0.0008,
        "season": 0.07,
    }
    assert resolved.gamma0_season.mean_array().shape == (3,)


def test_scenario_factories_expose_period_and_scientific_controls():
    scenarios = bx.make_structural_scenarios(
        n_time=120,
        period=6,
        sigma=1.2,
        xi=-0.2,
        random_walk_sd=0.09,
        fixed_season_amplitude=1.1,
    )
    assert len(scenarios) == 6
    assert {item.period for item in scenarios} == {6}
    assert {item.n_time for item in scenarios} == {120}
    assert {item.sigma for item in scenarios} == {1.2}
    assert {item.xi for item in scenarios} == {-0.2}
    random_walk = next(item for item in scenarios if item.name == "random_walk")
    assert random_walk.params["sd.level"] == 0.09
    seasonal = next(
        item
        for item in scenarios
        if item.name == "local_linear_trend_fixed_season"
    )
    assert seasonal.season_amplitude == 1.1
    assert seasonal.initial_state.size == 7


def test_simulation_series_are_separate_and_decomposition_has_its_own_file(
    tmp_path,
):
    tail_tables = {}
    tail_truths = {}
    for scenario in bx.TAIL_SCENARIOS:
        _, tail_tables[scenario.name] = bx.simulate_scenario(scenario.resized(48))
        tail_truths[scenario.name] = scenario.resized(48).to_dict()
    tail_paths = bx.plot_tail_simulations(
        tail_tables,
        tail_truths,
        tmp_path / "tail",
        formats=("png",),
        dpi=72,
    )
    assert {path.name for path in tail_paths} == {
        "10_tail_01_bounded_tail.png",
        "10_tail_02_gumbel_tail.png",
        "10_tail_03_heavy_tail.png",
        "11_gev_shape_comparison.png",
    }

    scale_tables = {}
    scale_truths = {}
    for scale in bx.SCALE_SCENARIOS:
        _, scale_tables[scale.name] = bx.simulate_scenario(scale.resized(48))
        scale_truths[scale.name] = scale.resized(48).to_dict()
    scale_paths = bx.plot_scale_simulations(
        scale_tables,
        scale_truths,
        tmp_path / "scale",
        formats=("png",),
        dpi=72,
    )
    assert {path.name for path in scale_paths} == {
        "12_scale_01_low_scale.png",
        "12_scale_02_reference_scale.png",
        "12_scale_03_high_scale.png",
        "13_gev_scale_comparison.png",
    }

    scenario = bx.scenario_by_name("stationary_dynamic_season").resized(48)
    _, table = bx.simulate_scenario(scenario)
    structural_paths = bx.plot_structural_simulations(
        {scenario.name: table},
        {scenario.name: scenario.to_dict()},
        tmp_path / "structure",
        formats=("png",),
        dpi=72,
    )
    assert {path.name for path in structural_paths} == {
        "20_01_stationary_dynamic_season_series.png",
        "21_01_stationary_dynamic_season_decomposition.png",
    }


def test_runtime_profiles_and_v260_paths_are_reproducible(tmp_path):
    config = bx.PresentationConfig.for_profile(
        "smoke", output_dir=tmp_path / "presentation", progress=False
    )
    assert config.start == "2015-01-01"
    assert config.runtime.particles == 24
    assert config.runtime.simulation_months == 48
    assert config.runtime.simulation_period == 4
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

    assert warm["__centered_path__"].shape == (laplace.n_time + 1, 5)
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
