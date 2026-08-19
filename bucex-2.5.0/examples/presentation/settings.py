"""Edit this one file to configure all presentation examples."""
from pathlib import Path

import bucex as bx


# smoke: software check; pilot: timings/preliminary plots; publication: final.
PROFILE = "pilot"
OUTPUT_DIR = Path("results/presentation")
DATA_DIR = None
POOL = "selection"
SEED = 25_000


def configuration() -> bx.PresentationConfig:
    return bx.PresentationConfig.for_profile(
        PROFILE,
        output_dir=OUTPUT_DIR,
        data_dir=DATA_DIR,
        pool=POOL,
        seed=SEED,
        progress=True,
    )


def workflow() -> bx.PresentationWorkflow:
    return bx.PresentationWorkflow(configuration())
