"""Stage 5: exploratory six-series Laplace screen."""
from __future__ import annotations

from settings import workflow


def main() -> None:
    # This approximation is for screening and initialization, not the final
    # non-Gaussian posterior.
    workflow().run_hierarchy_screen(pool="selection", figures=True)


if __name__ == "__main__":
    main()
