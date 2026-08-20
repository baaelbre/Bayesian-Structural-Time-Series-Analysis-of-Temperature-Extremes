"""Fit TXx, TXn, TNx, and TNn using the Laplace state approximation."""
from settings import DIAGNOSTICS, OVERWRITE, SERIES, workflow


if __name__ == "__main__":
    workflow().fit_uccle(
        engine="laplace",
        series=SERIES,
        overwrite=OVERWRITE,
        figures=True,
        diagnostics=DIAGNOSTICS,
    )

