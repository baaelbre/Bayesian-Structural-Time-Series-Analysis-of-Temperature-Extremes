"""Fit every structural scenario using the Laplace state approximation."""
from settings import DIAGNOSTICS, OVERWRITE, SCENARIO, workflow


if __name__ == "__main__":
    workflow().fit_simulations(
        engine="laplace",
        scenario=SCENARIO,
        overwrite=OVERWRITE,
        figures=True,
        diagnostics=DIAGNOSTICS,
    )

