"""Simulate fixed, absent, and dynamic UC structures at common sigma and xi."""
from settings import OVERWRITE, workflow


if __name__ == "__main__":
    workflow().run_simulations(kind="structure", overwrite=OVERWRITE, figures=True)

