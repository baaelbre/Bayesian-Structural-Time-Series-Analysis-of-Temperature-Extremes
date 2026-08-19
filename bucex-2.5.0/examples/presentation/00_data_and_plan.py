"""Stage 0: inspect the resolved plan and validate the Uccle summaries."""
from __future__ import annotations

from settings import configuration, workflow


def main() -> None:
    config = configuration()
    print(config.to_dict())
    print("Data table:", workflow().validate_data())


if __name__ == "__main__":
    main()
