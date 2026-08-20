#!/usr/bin/env python3
"""Direct-API smoke validation for the seven bucex 2.6.2 examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


def run(work_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    scripts = sorted((SOURCE_ROOT / "examples").glob("[0-9][0-9]_*.py"))
    if len(scripts) != 7:
        raise RuntimeError(f"Expected seven examples, found {len(scripts)}.")
    for script in scripts:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")

    model = bx.Model(
        bx.GEV(),
        (
            bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            bx.DummySeasonal(period=4, mode="dynamic"),
        ),
        name="smoke model",
    )
    simulation = bx.simulate(
        model,
        24,
        {"sigma": 1.0, "xi": -0.2, "sd.level": 0.04, "sd.slope": 0.001, "sd.seasonal": 0.03},
        initial_state=[20.0, 0.0, -0.2, 0.0, 0.2],
        seed=2620,
    )
    priors = bx.ssvs_gev_priors(period=4, alpha_mean=20.0)
    laplace = bx.fit(
        simulation.y,
        model=model,
        priors=priors,
        engine="laplace",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=2621),
    )
    pgas = bx.fit(
        simulation.y,
        model=model,
        priors=laplace.priors,
        engine="pgas",
        parameterization="fruehwirth_schnatter",
        init=laplace,
        particles=bx.Particles(n=16, proposal="guided"),
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=2622),
    )
    if laplace.plan.targets_exact_posterior:
        raise RuntimeError("The Laplace fit was incorrectly marked exact.")
    if not pgas.plan.targets_exact_posterior:
        raise RuntimeError("The PGAS fit was not marked exact-invariant.")
    if pgas.meta.get("warm_start_source_engine") != "laplace":
        raise RuntimeError("PGAS did not retain Laplace warm-start provenance.")

    work_dir.mkdir(parents=True, exist_ok=True)
    laplace.save(work_dir / "laplace.bucex")
    pgas.save(work_dir / "pgas.bucex")
    figure, axis = pgas.plot("season", labels=("1", "2", "3", "4"))
    figure.savefig(work_dir / "season.png", dpi=72)
    plt.close(figure)

    x = np.arange(41, dtype=float)
    y = 2.0 + 0.2 * x
    y[20] += 10.0
    loess = bx.loess_smooth(x, y, fraction=0.3)
    if not np.all(np.isfinite(loess)):
        raise RuntimeError("LOESS returned non-finite values.")

    return {
        "bucex_version": bx.__version__,
        "work_dir": str(work_dir),
        "examples": [script.name for script in scripts],
        "warm_start_source": pgas.meta["warm_start_source_engine"],
        "pgas_engine_diagnostics": pgas.diagnostics()["engine"],
        "season_lines": len(axis.lines),
        "loess_max_error_without_outlier": float(np.max(np.abs(np.delete(loess - (2.0 + 0.2 * x), 20)))),
        "total_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("validation/smoke_artifacts_2.6.2"))
    parser.add_argument("--output", type=Path, default=Path("validation/presentation_smoke_2.6.2.json"))
    args = parser.parse_args()
    result = run(args.work_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
