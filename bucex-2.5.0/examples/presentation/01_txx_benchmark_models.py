"""Stage 1: fixed structural alternatives for the univariate TXx story."""
from __future__ import annotations

from settings import workflow


def main() -> None:
    # Structural analogues: stationary, linear, Huerta-style local level,
    # Gaetan--Grigoletto-style RW2, and the full local-linear trend.
    workflow().run_txx_benchmarks(engine="pgas", figures=True)


if __name__ == "__main__":
    main()
