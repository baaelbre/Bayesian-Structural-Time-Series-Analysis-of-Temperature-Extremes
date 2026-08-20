# bucex 2.6.2 examples

These seven standalone files reproduce the complete COMPSTAT analysis. They
use the public `bucex` API directly and are intended to be read as well as run:

1. `00_uccle_record.py` — record from 1892, TXx evolution, and robust LOESS.
2. `01_tail_simulations.py` — matched shape and scale experiments.
3. `02_structural_simulations.py` — six explicit structural models.
4. `03_simulation_laplace.py` — Laplace fits and selection recovery.
5. `04_simulation_pgas.py` — PGAS fits initialized from Laplace.
6. `05_uccle_laplace.py` — Laplace analysis of TXx, TXn, TNx, and TNn.
7. `06_uccle_pgas.py` — PGAS analysis and engine comparison.

Run them from the package root. To keep every output in one timestamped run,
set a run ID once:

```bash
export BUCEX_RUN_ID=$(date +%Y%m%d_%H%M%S)
python examples/00_uccle_record.py
python examples/01_tail_simulations.py
python examples/02_structural_simulations.py
python examples/03_simulation_laplace.py
python examples/04_simulation_pgas.py
python examples/05_uccle_laplace.py
python examples/06_uccle_pgas.py
```

The default root is `results/<BUCEX_RUN_ID>/`. Set
`BUCEX_TIMESTAMP_RESULTS=0` to write directly under `results/`, or set
`BUCEX_RESULTS_ROOT` to another location. Existing files are reused when that
is safe; set `BUCEX_OVERWRITE=1` to regenerate them.

Every scientific setting is near the top of the relevant script. Sampler
controls can also be supplied as `BUCEX_DRAWS`, `BUCEX_WARMUP`,
`BUCEX_CHAINS`, `BUCEX_PARTICLES`, and `BUCEX_SEED`. The PBS equivalents and
publication profile are documented in [`job_scripts/README.md`](job_scripts/README.md).
