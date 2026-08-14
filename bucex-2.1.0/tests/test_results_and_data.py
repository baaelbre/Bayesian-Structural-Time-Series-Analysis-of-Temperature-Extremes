from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pytest

import bucex as bx
from bucex.cli import main as cli_main


def _fit(parameterization="fruehwirth_schnatter", seed=30):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(scale=0.08, size=24)) + rng.normal(scale=0.2, size=24)
    return bx.fit(
        y,
        family="gaussian",
        period=12,
        priors="normal",
        parameterization=parameterization,
        asis=True,
        mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=seed + 1),
    )


@pytest.mark.parametrize("parameterization", ["centered", "fruehwirth_schnatter"])
def test_safe_fit_archive_roundtrip_and_integrity(tmp_path, parameterization):
    fit = _fit(parameterization)
    path = tmp_path / f"{parameterization}.bucex"
    fit.save(path)
    restored = bx.FitResult.load(path)
    assert type(restored) is bx.FitResult
    assert restored.plan == fit.plan
    assert restored.model == fit.model
    assert type(restored.priors) is type(fit.priors)
    np.testing.assert_allclose(restored.state_draws, fit.state_draws)

    with zipfile.ZipFile(path) as archive:
        metadata = archive.read("metadata.json")
        arrays = bytearray(archive.read("arrays.npz"))
    arrays[-1] ^= 1
    broken = tmp_path / "broken.bucex"
    with zipfile.ZipFile(broken, "w") as archive:
        archive.writestr("metadata.json", metadata)
        archive.writestr("arrays.npz", arrays)
    with pytest.raises(ValueError, match="integrity"):
        bx.FitResult.load(broken)


def test_fs_forecast_ignores_private_vector_parameters_and_scores():
    fit = _fit()
    forecast = fit.forecast(3, draws=5, seed=32)
    assert forecast.observations.shape == (5, 3)
    assert set(forecast.parameters) == {
        "sd.level",
        "sd.slope",
        "sd.seasonal",
        "sigma",
    }
    table = forecast.score(
        np.zeros(3), thresholds=[0.2], quantiles=[0.9]
    )
    assert set(table["score"]) == {
        "crps",
        "twcrps",
        "exceedance_brier",
        "exceedance_log",
        "quantile",
    }


def test_minimum_tail_results_stay_on_original_orientation():
    values = 5.0 - np.random.default_rng(33).gumbel(size=20)
    fit = bx.fit(
        values,
        family="gev",
        trend="local_level",
        tail="min",
        engine="laplace",
        parameterization="centered",
        priors="normal",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=34),
        laplace=bx.Laplace(max_iterations=8),
    )
    np.testing.assert_allclose(fit.observed, values)
    assert fit.forecast(2, draws=2, seed=35).tail == "lower"
    assert "<" in fit.event_label(0.0)


def test_combine_fits_preserves_chain_identity():
    y = np.linspace(-1.0, 1.0, 18)
    model = bx.Model(bx.Gaussian(), [bx.LocalLevel()])
    fits = [
        bx.fit(
            y,
            model,
            parameterization="centered",
            priors="normal",
            mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=seed),
        )
        for seed in (40, 41)
    ]
    combined = bx.combine_fits(fits)
    assert combined.n_chains == 2
    np.testing.assert_allclose(combined.state_draws[0], fits[0].state_draws[0])
    np.testing.assert_allclose(combined.state_draws[1], fits[1].state_draws[0])


def test_rank_diagnostics_detect_a_shifted_chain():
    rng = np.random.default_rng(42)
    mixed = rng.normal(size=(4, 300))
    shifted = mixed.copy()
    shifted[-1] += 2.0
    assert bx.rhat(mixed) < 1.05
    assert bx.rhat(shifted) > 1.1
    assert 100.0 < bx.ess_bulk(mixed) <= mixed.size


def test_bundled_uccle_data_and_short_minimum_workflow():
    table = bx.validate_uccle_data("data", check_daily=True)
    assert tuple(table.index) == bx.UCCLE_SERIES
    assert np.all(table["daily_mismatches"].to_numpy() == 0)
    fit = bx.fit_uccle_series(
        "TXn",
        data_dir="data",
        start="2000-01-01",
        end="2001-12-01",
        priors="normal",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=43),
    )
    assert fit.series_name == "TXn"
    assert fit.transform_sign == -1.0
    probability, years = fit.exceedance_probability_draws(
        0.0, annual=True, return_labels=True
    )
    assert probability.shape == (2, len(years))
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_explicit_single_series_directory_is_authoritative():
    import pandas as pd

    with tempfile.TemporaryDirectory() as directory:
        dates = pd.date_range("2000-01-01", periods=24, freq="MS")
        pd.DataFrame({"date": dates, "TXm": np.linspace(1.0, 2.0, 24)}).to_csv(
            Path(directory) / "TXm.csv", index=False
        )
        values = bx.load_uccle_series("TXm", directory)
    assert values.size == 24
    assert values.iloc[0] == pytest.approx(1.0)
    assert values.iloc[-1] == pytest.approx(2.0)


def test_risk_methods_keep_compact_tuple_defaults():
    values = 5.0 - np.random.default_rng(44).gumbel(size=24)
    fit = bx.fit(
        values,
        family="gev",
        trend="local_level",
        tail="min",
        engine="laplace",
        parameterization="centered",
        priors="normal",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=45),
        laplace=bx.Laplace(max_iterations=8),
    )
    probability, labels = fit.exceedance_probability_draws(4.0, annual=True)
    periods, period_labels = fit.return_period_draws(4.0)
    assert probability.shape[1] == labels.size
    assert periods.shape[1] == period_labels.size
    np.testing.assert_array_equal(labels, period_labels)


def test_single_uccle_cli_fits_and_inspects(tmp_path, capsys):
    output = tmp_path / "TXm.bucex"
    assert cli_main(
        [
            "fit",
            "TXm",
            "--start",
            "2000-01-01",
            "--end",
            "2001-12-01",
            "--parameterization",
            "centered",
            "--priors",
            "normal",
            "--draws",
            "1",
            "--warmup",
            "1",
            "--chains",
            "1",
            "--no-progress",
            "--output",
            str(output),
        ]
    ) == 0
    assert output.is_file()
    first = json.loads(capsys.readouterr().out)
    assert first["plan"]["parameterization"] == "centered"
    assert cli_main(["inspect", str(output)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["series"] == "TXm"


def test_uccle_compact_mcmc_accepts_partial_overrides():
    from bucex.datasets.uccle import _compatibility_mcmc

    resolved = _compatibility_mcmc(
        None,
        n_iter=8,
        burn=None,
        thin=2,
        chains=3,
        seed=47,
        progress=None,
    )
    # A short explicit n_iter cannot inherit the production burn-in.
    assert resolved.warmup == 0
    assert resolved.draws == 4
    assert resolved.chains == 3
