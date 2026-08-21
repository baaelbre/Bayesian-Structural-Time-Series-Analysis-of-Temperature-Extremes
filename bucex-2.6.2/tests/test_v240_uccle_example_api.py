from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

import bucex as bx
from bucex.datasets.uccle import _normalize_series_names


@pytest.mark.parametrize("builder", [bx.ssvs_gaussian_priors, bx.ssvs_gev_priors])
def test_ssvs_prior_builders_accept_readable_component_probabilities(builder):
    priors = builder(
        period=4,
        level_dynamic_probability=0.6,
        trend_probabilities=(0.1, 0.4, 0.5),
        season_probabilities=(0.0, 0.25, 0.75),
    )
    assert priors.ssvs.level_dynamic_probability == pytest.approx(0.6)
    assert tuple(priors.ssvs.trend_probabilities) == (0.1, 0.4, 0.5)
    assert tuple(priors.ssvs.season_probabilities) == (0.0, 0.25, 0.75)


def test_ssvs_prior_builder_rejects_object_and_direct_settings_together():
    with pytest.raises(ValueError, match="either ssvs=SSVSPrior"):
        bx.ssvs_gaussian_priors(
            ssvs=bx.SSVSPrior(),
            season_probabilities=(0.0, 0.5, 0.5),
        )


def test_ssvs_gev_prior_builder_exposes_observation_and_initial_hyperparameters():
    priors = bx.ssvs_gev_priors(
        period=4,
        alpha_mean=25.0,
        alpha_sd=1.7,
        beta_sd=0.02,
        seasonal_initial_sd=0.8,
        sigma2_prior=bx.InverseGammaPrior(3.0, 4.0),
        xi_prior=bx.UniformPrior(-0.4, 0.2),
        xi_max_abs=0.45,
    )
    assert priors.alpha0.sd == 1.7
    assert priors.beta0.sd == 0.02
    assert priors.sigma2 == bx.InverseGammaPrior(3.0, 4.0)
    assert priors.xi == bx.UniformPrior(-0.4, 0.2)
    assert priors.xi_max_abs == 0.45
    assert priors.gamma0_season.sd_array().tolist() == [0.8, 0.8, 0.8]


def test_uccle_series_normalization_does_not_split_a_string():
    assert _normalize_series_names("TXm") == ("TXm",)
    with pytest.raises(ValueError, match="Unknown Uccle series"):
        _normalize_series_names("T")


def test_v262_example_surface_contains_only_seven_top_level_scripts():
    directory = Path(__file__).resolve().parents[1] / "examples"
    scripts = {path.name for path in directory.glob("*.py")}
    assert scripts == {
        "00_uccle_record.py",
        "01_tail_simulations.py",
        "02_structural_simulations.py",
        "03_simulation_laplace.py",
        "04_simulation_pgas.py",
        "05_uccle_laplace.py",
        "06_uccle_pgas.py",
    }
    assert not (directory / "presentation").exists()


def test_v262_examples_are_standalone_public_api_scripts():
    directory = Path(__file__).resolve().parents[1] / "examples"
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in directory.glob("*.py")
    }
    for name, source in sources.items():
        compile(source, str(directory / name), "exec")
        assert "import bucex as bx" in source
        assert "SOURCE_ROOT = Path(__file__).resolve().parents[1]" in source
        assert "sys.path.insert(0, str(SOURCE_ROOT))" in source
        assert "bucex.workflows" not in source
        assert "PresentationWorkflow" not in source
        assert "export_fit_results" not in source
        assert "plot_fit_results" not in source
        assert "SCRIPT_NAME = Path(__file__).stem" in source
        assert "RUN_SIGNATURE" in source
        assert 'OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"' in source
        assert 'run_config.json' in source
        assert "TIMESTAMP_RESULTS" not in source
        assert "BUCEX_TIMESTAMP_RESULTS" not in source

    assert "bx.loess_smooth(" in sources["00_uccle_record.py"]
    assert '"1892-01-01"' in sources["00_uccle_record.py"]
    assert "common_ylim" in sources["01_tail_simulations.py"]
    assert "axis.set_ylim(*common_ylim)" in sources["01_tail_simulations.py"]
    for name in ("01_tail_simulations.py", "02_structural_simulations.py"):
        assert "bx.Model(" in sources[name]
        assert "bx.simulate(" in sources[name]
    for name in (
        "03_simulation_laplace.py",
        "04_simulation_pgas.py",
        "05_uccle_laplace.py",
        "06_uccle_pgas.py",
    ):
        assert "bx.Model(" in sources[name]
        assert "bx.fit(" in sources[name]
        assert '.plot("season"' in sources[name]
    assert "init=laplace_fit" in sources["04_simulation_pgas.py"]
    assert "init=laplace_fit" in sources["06_uccle_pgas.py"]


def test_direct_path_example_prefers_the_adjacent_source_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "00_uccle_record.py"
    probe = (
        "import runpy\n"
        f"scope = runpy.run_path({str(script)!r}, run_name='bucex_example_probe')\n"
        f"assert scope['bx'].__file__.startswith({str(root)!r})\n"
        "assert hasattr(scope['bx'], 'loess_smooth')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
