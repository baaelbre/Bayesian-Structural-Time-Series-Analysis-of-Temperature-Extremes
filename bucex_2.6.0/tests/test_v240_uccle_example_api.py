from __future__ import annotations

from pathlib import Path

import pytest

import bucex as bx
from bucex.datasets.uccle import _normalize_series_names


@pytest.mark.parametrize(
    "builder",
    [bx.ssvs_gaussian_priors, bx.ssvs_gev_priors],
)
def test_ssvs_prior_builders_accept_readable_component_probabilities(builder):
    priors = builder(
        period=4,
        level_dynamic_probability=0.6,
        trend_probabilities=(0.1, 0.4, 0.5),
        season_probabilities=(0.0, 0.25, 0.75),
    )
    assert priors.ssvs is not None
    assert priors.ssvs.level_dynamic_probability == pytest.approx(0.6)
    assert tuple(priors.ssvs.trend_probabilities) == (0.1, 0.4, 0.5)
    assert tuple(priors.ssvs.season_probabilities) == (0.0, 0.25, 0.75)


def test_ssvs_prior_builder_rejects_object_and_direct_settings_together():
    with pytest.raises(ValueError, match="either ssvs=SSVSPrior"):
        bx.ssvs_gaussian_priors(
            ssvs=bx.SSVSPrior(),
            season_probabilities=(0.0, 0.5, 0.5),
        )


def test_uccle_series_normalization_does_not_split_a_string():
    assert _normalize_series_names("TXm") == ("TXm",)
    with pytest.raises(ValueError, match="Unknown Uccle series"):
        _normalize_series_names("T")


def test_v260_example_surface_contains_only_the_presentation_sequence():
    directory = Path(__file__).resolve().parents[1] / "examples" / "presentation"
    scripts = {path.name for path in directory.glob("*.py")}
    assert scripts == {
        "settings.py",
        "00_uccle_record.py",
        "01_tail_simulations.py",
        "02_structural_simulations.py",
        "03_simulation_laplace.py",
        "04_simulation_pgas.py",
        "05_uccle_laplace.py",
        "06_uccle_pgas.py",
        "07_build_results.py",
    }
