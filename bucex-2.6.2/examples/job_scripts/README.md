# PBS runners and submission files

Every analysis now has two files, following the usual PBS pattern:

- `examples/bash_scripts/run_*.sh` is the executable runner. It accepts
  positional arguments, activates the bucex environment, exports the matching
  `BUCEX_*` variables, and calls the Python example.
- `examples/job_scripts/submit_*.pbs` contains resource requests, logging, and
  named PBS settings. It calls the corresponding runner.

For example:

```text
submit_03_simulation_laplace.pbs
    -> run_03_simulation_laplace.sh
        -> examples/03_simulation_laplace.py
```

Submit from the package root. `PBS_O_WORKDIR` will then point to the correct
directory:

```bash
qsub examples/job_scripts/submit_03_simulation_laplace.pbs
```

## Environment

The runners look for the virtual environment in
`${HOME}/venvs/bucex_env` by default. To use another environment, either edit
the `VENV_DIR` default in the runner or pass its directory when submitting:

```bash
qsub -v BUCEX_VENV_DIR=/absolute/path/to/bucex_env \
  examples/job_scripts/submit_03_simulation_laplace.pbs
```

You can instead provide an explicit interpreter:

```bash
qsub -v BUCEX_PYTHON=/absolute/path/to/bucex_env/bin/python \
  examples/job_scripts/submit_03_simulation_laplace.pbs
```

## Run a shell script directly

The runners use ordinary positional arguments, just like any Bash script. The
order is documented at the top of every file. For example, the Laplace
simulation runner expects:

```text
N_TIME PERIOD SIMULATION_SEED DRAWS WARMUP CHAINS MCMC_SEED
RESULTS_ROOT RUN_ID OVERWRITE
```

Thus a final run can be started interactively as:

```bash
bash examples/bash_scripts/run_03_simulation_laplace.sh \
  1000 4 13081997 2000 2000 4 13081997 results manual_test 0
```

Arguments may be omitted from the right. For example, this changes only
`N_TIME` and `PERIOD` and keeps all later defaults:

```bash
bash examples/bash_scripts/run_02_structural_simulations.sh 1500 4
```

Use the literal value `latest` for an open-ended Uccle end date:

```bash
bash examples/bash_scripts/run_05_uccle_laplace.sh \
  1892-01-01 latest 2000 2000 4 56000 data results uccle_final 0
```

## Pass arguments through qsub

For a queued job, pass named variables with `qsub -v`. The PBS file reads
these variables and places them in the correct positional order for the
runner:

```bash
qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,DRAWS=2000,WARMUP=2000,CHAINS=4,MCMC_SEED=13081997 \
  examples/job_scripts/submit_03_simulation_laplace.pbs
```

For PGAS, add the particle count:

```bash
qsub -v N_TIME=1000,PERIOD=4,DRAWS=2000,WARMUP=2000,CHAINS=4,PARTICLES=512 \
  examples/job_scripts/submit_04_simulation_pgas.pbs
```

For the Uccle analysis:

```bash
qsub -v START=1892-01-01,END=2023-12-31,DRAWS=2000,WARMUP=2000,CHAINS=4,DATA_DIR=data \
  examples/job_scripts/submit_05_uccle_laplace.pbs
```

There must be commas between variables and no spaces around the commas.
`qsub` variables override the defaults in the PBS file; the PBS file then
passes the resulting values to the `.sh` runner.

## Arguments by pair

| pair | runner positional arguments / PBS variable names |
|---|---|
| `00_uccle_record` | `START END DATA_DIR RESULTS_ROOT RUN_ID OVERWRITE` |
| `01_tail_simulations` | `N_TIME PERIOD RESULTS_ROOT RUN_ID OVERWRITE` |
| `02_structural_simulations` | `N_TIME PERIOD SIMULATION_SEED RESULTS_ROOT RUN_ID OVERWRITE` |
| `03_simulation_laplace` | `N_TIME PERIOD SIMULATION_SEED DRAWS WARMUP CHAINS MCMC_SEED RESULTS_ROOT RUN_ID OVERWRITE` |
| `04_simulation_pgas` | `N_TIME PERIOD SIMULATION_SEED DRAWS WARMUP CHAINS PARTICLES MCMC_SEED RESULTS_ROOT RUN_ID OVERWRITE` |
| `05_uccle_laplace` | `START END DRAWS WARMUP CHAINS MCMC_SEED DATA_DIR RESULTS_ROOT RUN_ID OVERWRITE` |
| `06_uccle_pgas` | `START END DRAWS WARMUP CHAINS PARTICLES MCMC_SEED DATA_DIR RESULTS_ROOT RUN_ID OVERWRITE` |

The main scientific settings—GEV scale and shape, process-noise truths, SSVS
probabilities, and prior scales—remain visible at the top of the Python files.
Edit those there for scientific sensitivity analyses.

## Shared run identifier and dependencies

Use one run identifier when several scripts should have the same timestamp
prefix:

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"

SIM_JOB=$(qsub -v RUN_ID="${RUN_ID}",N_TIME=1000,PERIOD=4 \
  examples/job_scripts/submit_02_structural_simulations.pbs)

LAP_JOB=$(qsub -W depend=afterok:"${SIM_JOB}" \
  -v RUN_ID="${RUN_ID}",N_TIME=1000,PERIOD=4,DRAWS=2000,WARMUP=2000,CHAINS=4 \
  examples/job_scripts/submit_03_simulation_laplace.pbs)

qsub -W depend=afterok:"${LAP_JOB}" \
  -v RUN_ID="${RUN_ID}",N_TIME=1000,PERIOD=4,DRAWS=2000,WARMUP=2000,CHAINS=4,PARTICLES=512 \
  examples/job_scripts/submit_04_simulation_pgas.pbs
```

Monitor jobs with `qstat -u "$USER"` and cancel one with `qdel JOB_ID`.
Adjust the `#PBS` walltime, memory, CPU, project, and queue directives in each
submission file to match the local cluster.
