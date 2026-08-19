"""Stage 6: final componentwise hierarchy with exact-invariant PGAS."""
from __future__ import annotations

from settings import workflow


def main() -> None:
    # If Stage 5 exists, it is used only as a warm start. PGAS still performs
    # full warmup and targets its declared posterior.
    workflow().run_hierarchy_pgas(
        pool="selection",
        figures=True,
        diagnostic_plots=True,
    )


if __name__ == "__main__":
    main()
