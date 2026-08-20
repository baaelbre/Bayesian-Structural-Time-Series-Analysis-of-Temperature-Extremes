"""Simulate one local-level signal with heavy, Gumbel, and bounded GEV tails."""
from settings import OVERWRITE, workflow


if __name__ == "__main__":
    workflow().run_simulations(kind="tail", overwrite=OVERWRITE, figures=True)

