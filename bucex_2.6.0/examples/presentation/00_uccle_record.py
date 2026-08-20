"""Describe the Uccle extremes record, beginning with TXx."""
from settings import OVERWRITE, workflow


if __name__ == "__main__":
    workflow().run_data(overwrite=OVERWRITE, figures=True)

