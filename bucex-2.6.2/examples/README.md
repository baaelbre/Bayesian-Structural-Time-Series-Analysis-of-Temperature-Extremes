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

Run them from the package root. Each script writes to its own directory and
automatically combines a timestamp with the settings that identify that run.
Set one timestamp before a linked local or HPC run so the seven directories
share the same prefix:

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

The resulting layout is
`results/<script>/<BUCEX_RUN_ID>__<automatic-settings-signature>/`.
There is no switch for disabling either part of the run name. Set
`BUCEX_RESULTS_ROOT` to change the root. Every run also contains a complete
`run_config.json`; set `BUCEX_OVERWRITE=1` only when deliberately rerunning an
existing identifier.

Every scientific setting is near the top of the relevant script. For HPC use,
each example has a positional-argument runner in `bash_scripts/` and a matching
resource-and-logging submission file in `job_scripts/`. Their direct Bash and
`qsub -v` argument conventions are documented in
[`job_scripts/README.md`](job_scripts/README.md). A complete copy-and-paste
submission sequence, including the parallel-chain setup, is in
[`HPC_PARALLEL.md`](HPC_PARALLEL.md).
