from __future__ import annotations

import json

import numpy as np

import bucex as bx


def test_v250_defaults_and_benchmark_states_are_explicit():
    hierarchy = bx.HierarchicalPrior()
    assert hierarchy.pool == "selection"
    assert hierarchy.model_space == "componentwise"
    assert hierarchy.trend_states == ("zero", "fixed", "dynamic")

    expected = {
        "stationary": (0.0, (1.0, 0.0, 0.0)),
        "linear_trend": (0.0, (0.0, 1.0, 0.0)),
        "huerta_local_level": (1.0, (1.0, 0.0, 0.0)),
        "gaetan_grigoletto_rw2": (0.0, (0.0, 0.0, 1.0)),
        "local_linear_trend": (1.0, (0.0, 0.0, 1.0)),
    }
    for name, (level, trend) in expected.items():
        prior = bx.txx_benchmark_prior(name).ssvs
        assert prior.level_dynamic_probability == level
        np.testing.assert_allclose(prior.trend_probabilities, trend)
        np.testing.assert_allclose(prior.season_probabilities, (0.0, 1.0, 0.0))


def test_runtime_profiles_and_paths_are_reproducible(tmp_path):
    config = bx.PresentationConfig.for_profile(
        "smoke", output_dir=tmp_path / "presentation", progress=False
    )
    assert config.start == "2018-01-01"
    assert config.runtime.particles == 20
    paths = bx.WorkflowPaths.from_config(config)
    assert paths.hierarchy_screen() == (
        tmp_path / "presentation/fits/05_hierarchy_screen/selection.bucex"
    )
    assert paths.txx_ssvs_fit(2).name == "chain_02.bucex"


def test_smoke_data_and_one_txx_fit_export_manifest(tmp_path):
    config = bx.PresentationConfig.for_profile(
        "smoke",
        output_dir=tmp_path / "run",
        progress=False,
        draws=1,
        warmup=1,
    )
    workflow = bx.PresentationWorkflow(config)
    assert workflow.validate_data().is_file()
    fit_path = workflow.run_txx_ssvs(engine="pgas")
    fit = bx.FitResult.load(fit_path)
    assert fit.plan.targets_exact_posterior
    assert fit.component_probabilities().shape == (3, 3)
    assert (workflow.paths.tables / "02_txx_componentwise_ssvs/parameters.csv").is_file()
    manifest = json.loads(workflow.paths.manifest.read_text(encoding="utf-8"))
    assert set(("data", "txx-ssvs")) <= set(manifest["stages"])
