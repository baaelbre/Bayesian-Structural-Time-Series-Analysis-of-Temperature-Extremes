"""Stage 7: test shared slab magnitudes and combined pooling."""
from __future__ import annotations

from settings import workflow


def main() -> None:
    analysis = workflow()
    for pool in ("slab", "both"):
        # Start with Laplace. Promote a sensitivity fit to PGAS only when it
        # changes a scientific conclusion or is reported as a final result.
        analysis.run_sensitivity(pool=pool, engine="laplace", figures=True)


if __name__ == "__main__":
    main()
