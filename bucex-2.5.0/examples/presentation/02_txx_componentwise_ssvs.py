"""Stage 2: TXx with componentwise structural selection."""
from __future__ import annotations

from settings import workflow


def main() -> None:
    workflow().run_txx_ssvs(
        engine="pgas",
        figures=True,
        diagnostic_plots=True,
    )


if __name__ == "__main__":
    main()
