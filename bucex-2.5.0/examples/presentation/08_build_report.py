"""Stage 8: recreate all tables and figures from saved fit archives."""
from __future__ import annotations

from settings import workflow


def main() -> None:
    artifacts = workflow().report(figures=True, diagnostic_plots=False)
    print(f"Generated {len(artifacts)} report artifacts without refitting.")


if __name__ == "__main__":
    main()
