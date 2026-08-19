"""Stage 3: compare TXx models on genuinely held-out observations."""
from __future__ import annotations

from settings import workflow


def main() -> None:
    # Proper scores and PIT are exported for every benchmark and for SSVS.
    workflow().run_txx_validation(engine="pgas")


if __name__ == "__main__":
    main()
