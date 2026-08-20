"""Refit all four Uccle extremes with Laplace-initialized PGAS."""
from settings import DIAGNOSTICS, OVERWRITE, SERIES, workflow


if __name__ == "__main__":
    workflow().fit_uccle(
        engine="pgas",
        series=SERIES,
        overwrite=OVERWRITE,
        figures=True,
        diagnostics=DIAGNOSTICS,
    )

