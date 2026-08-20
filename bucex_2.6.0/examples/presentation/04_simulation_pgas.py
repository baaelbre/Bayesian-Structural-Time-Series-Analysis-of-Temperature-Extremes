"""Refit every scenario with PGAS, warm-started from its Laplace fit."""
from settings import DIAGNOSTICS, OVERWRITE, SCENARIO, workflow


if __name__ == "__main__":
    workflow().fit_simulations(
        engine="pgas",
        scenario=SCENARIO,
        overwrite=OVERWRITE,
        figures=True,
        diagnostics=DIAGNOSTICS,
    )

