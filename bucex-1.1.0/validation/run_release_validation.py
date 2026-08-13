#!/usr/bin/env python3
"""Reproduce the fixed-seed bucex 1.1 release checks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
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


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
parser.add_argument(
    "--output",
    type=Path,
    default=Path("validation/release_validation_2026-08-11.json"),
)
parser.add_argument("--particles", type=int, default=64)
args = parser.parse_args()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


started_total = time.perf_counter()
record: dict[str, object] = {
    "bucex_version": bx.__version__,
    "python": platform.python_version(),
    "platform": platform.platform(),
}

data_table = bx.validate_uccle_data(args.data_dir, check_daily=True)
record["uccle_data"] = data_table.reset_index().to_dict(orient="records")

# The broader v1 grammar, including dynamic regression, remains executable.
rng = np.random.default_rng(20)
exog = rng.normal(size=(30, 1))
y_general = 1.5 * exog[:, 0] + rng.normal(scale=0.25, size=30)
general_fit = bx.fit(
    y_general,
    model=bx.Model(
        bx.Gaussian(),
        [bx.LocalLevel(), bx.Regression(1, dynamic=True, name="x")],
    ),
    exog=exog,
    mcmc=bx.MCMC(draws=5, warmup=5, chains=2, seed=21),
)
record["general_engine"] = {
    "state_shape": list(general_fit.state_draws.shape),
    "parameterization": general_fit.plan.parameterization,
    "exact_target": general_fit.plan.targets_exact_posterior,
    "finite": bool(np.all(np.isfinite(general_fit.state_draws))),
}

# Each restored FS shrinkage profile runs through the same semantic state path.
profile_results = {}
sample = bx.load_uccle_series("TXm", args.data_dir).iloc[:48]
for index, profile in enumerate(
    ("manuscript", "regularized_lasso", "horseshoe", "pc", "normal", "ssvs")
):
    fit = bx.fit_bayes(
        sample.to_numpy(),
        family="gaussian",
        period=12,
        priors=profile,
        n_iter=6,
        burn=3,
        seed=30 + index,
        progress=False,
        asis=profile != "ssvs",
    )
    profile_results[profile] = {
        "draws": fit.n_draws,
        "finite_states": bool(np.all(np.isfinite(fit.draws_states))),
        "stored_parameters": sorted(fit.draws_static),
        "prior_profile": fit.priors.profile,
    }
record["fs_gaussian_profiles"] = profile_results

# Fast GEV path: convergence and restoration are explicit.
tail_sample = bx.load_uccle_series("TXx", args.data_dir).iloc[:60]
laplace_fit = bx.fit_bayes(
    tail_sample.to_numpy(),
    family="gev",
    period=12,
    priors="regularized_lasso",
    state_method="laplace",
    n_iter=5,
    burn=2,
    seed=39,
    progress=False,
    asis=True,
)
record["fs_laplace"] = {
    "exact_target": laplace_fit.plan.targets_exact_posterior,
    "approximation": laplace_fit.plan.approximation,
    "restored_iterations": laplace_fit.meta["restored_iterations"],
    "engine": laplace_fit.diagnostics()["engine"],
}

# Required full-length seed-40 PGAS smoke test. This exercises log-weight
# retention, ancestor sampling, exact GEV FS regression, and ASIS together.
started = time.perf_counter()
txx_pgas = bx.fit_uccle_series(
    "TXx",
    args.data_dir,
    priors="horseshoe",
    engine="pgas",
    asis=True,
    mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=40),
    particles=bx.Particles(n=args.particles, proposal="guided"),
)
record["txx_full_length_pgas_seed40"] = {
    "particles": args.particles,
    "seconds": time.perf_counter() - started,
    "exact_target": txx_pgas.plan.targets_exact_posterior,
    "restored_iterations": txx_pgas.meta["restored_iterations"],
    "finite_states": bool(np.all(np.isfinite(txx_pgas.draws_states))),
    "engine": txx_pgas.diagnostics()["engine"],
}

# The FS archive round trip must preserve arrays and model/prior types.
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "fit.bucex"
    txx_pgas.save(path)
    restored = bx.PosteriorBundle.load(path)
    record["safe_archive"] = {
        "roundtrip_equal": bool(
            np.array_equal(restored.draws_states, txx_pgas.draws_states)
        ),
        "model_type": type(restored.model).__name__,
        "prior_type": type(restored.priors).__name__,
        "bytes": path.stat().st_size,
    }

record["total_seconds"] = time.perf_counter() - started_total
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(record, indent=2, sort_keys=True, default=json_value),
    encoding="utf-8",
)
print(args.output)
