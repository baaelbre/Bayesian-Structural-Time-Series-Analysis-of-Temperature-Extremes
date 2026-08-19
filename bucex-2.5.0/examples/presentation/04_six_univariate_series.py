"""Stage 4: fit the six summaries independently as the no-pooling baseline."""
from __future__ import annotations

from settings import workflow


def main() -> None:
    workflow().run_six_univariate(gev_engine="pgas", figures=True)


if __name__ == "__main__":
    main()
