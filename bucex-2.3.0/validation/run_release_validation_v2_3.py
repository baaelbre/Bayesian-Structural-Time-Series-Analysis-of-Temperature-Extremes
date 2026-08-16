#!/usr/bin/env python3
"""Fixed-seed release validation for the bucex 2.3 multiseries contract."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _panel(*, mixed: bool) -> bx.MultiSeriesModel:
    return bx.MultiSeriesModel(
        channels=(
            bx.Channel(
                "first",
                bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
            ),
            bx.Channel(
                "second",
                bx.GEV() if mixed else bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
            ),
        ),
        name="v2.3 validation panel",
    )


def run() -> dict[str, object]:
    started = time.perf_counter()
    if bx.__version__ != "2.3.0":
        raise RuntimeError(f"Expected bucex 2.3.0, found {bx.__version__}.")
    record: dict[str, object] = {
        "version": bx.__version__,
        "python": platform.python_version(),
        "single_result_type": bx.PosteriorBundle is bx.FitResult,
    }

    rng = np.random.default_rng(2300)
    gaussian_data = np.column_stack(
        (
            np.linspace(0.0, 0.3, 28) + rng.normal(0.0, 0.2, 28),
            np.linspace(0.0, 0.1, 28) + rng.normal(0.0, 0.25, 28),
        )
    )
    gaussian_model = _panel(mixed=False)
    gaussian = bx.fit(
        gaussian_data,
        gaussian_model,
        priors="hierarchical_ssvs",
        mcmc=bx.MCMC(draws=4, warmup=4, chains=2, seed=2301),
    )
    sign_error = float(
        np.nanmax(gaussian.draws_aux["sign_invariance_error"])
    )
    assert gaussian.plan.engine == "ffbs"
    assert gaussian.plan.targets_exact_posterior
    assert not gaussian.is_factor_model
    assert gaussian.component_probabilities().shape == (6, 3)
    assert gaussian.hierarchical_probabilities().shape == (8, 4)
    assert sign_error <= 1e-12
    record["gaussian_panel"] = {
        "plan": gaussian.plan.to_dict(),
        "state_shape": list(gaussian.state_draws.shape),
        "component_table_shape": list(gaussian.component_probabilities().shape),
        "population_table_shape": list(gaussian.hierarchical_probabilities().shape),
        "sign_invariance_error": sign_error,
        "finite": bool(
            np.all(np.isfinite(gaussian.state_draws))
            and np.all(np.isfinite(gaussian.log_posterior))
        ),
    }

    mixed_data = np.column_stack(
        (rng.normal(0.0, 0.2, 18), rng.gumbel(0.0, 0.3, 18))
    )
    mixed = bx.fit(
        mixed_data,
        _panel(mixed=True),
        priors="hierarchical_ssvs",
        engine="pgas",
        particles=bx.Particles(n=20),
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2302),
    )
    assert mixed.plan.targets_exact_posterior
    assert mixed.meta["model_selection_exact"]
    assert mixed.meta["pgas_exact_invariant"]
    assert "xi.second" in mixed.parameter_draws
    record["mixed_panel"] = {
        "plan": mixed.plan.to_dict(),
        "model_selection_exact": mixed.meta["model_selection_exact"],
        "pgas_exact_invariant": mixed.meta["pgas_exact_invariant"],
        "restored_iterations": mixed.meta["restored_iterations"],
        "engine_diagnostics": mixed.diagnostics()["engine"],
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "panel.bucex"
        gaussian.save(path)
        restored = bx.FitResult.load(path)
        forecast = restored.forecast(3, draws=3, seed=2303)
        assert isinstance(restored.model, bx.MultiSeriesModel)
        assert np.array_equal(restored.state_draws, gaussian.state_draws)
        assert np.all(np.isfinite(forecast.observations))
        record["archive_forecast"] = {
            "archive_bytes": path.stat().st_size,
            "roundtrip_equal": True,
            "forecast_shape": list(forecast.observations.shape),
        }

    uccle_model = bx.make_uccle_multiseries_model()
    uccle_data = bx.load_uccle_factor_data(
        start="2020-01-01", end="2021-12-01"
    )
    uccle_plan = bx.plan(uccle_model, uccle_data)
    assert uccle_model.factor_names == ()
    assert uccle_plan.engine == "pgas"
    assert tuple(uccle_model.transform_signs) == (1.0, 1.0, 1.0, -1.0, 1.0, -1.0)
    record["uccle_graph"] = {
        "channels": list(uccle_model.channel_names),
        "families": list(uccle_model.families),
        "transform_signs": uccle_model.transform_signs.tolist(),
        "plan": uccle_plan.to_dict(),
    }

    record["seconds"] = time.perf_counter() - started
    return record


def main() -> int:
    result = run()
    output = SOURCE_ROOT / "validation" / "release_validation_2.3.0.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
