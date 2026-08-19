# PBS/Torque job scripts for the Uccle componentwise hierarchy

The files follow a two-layer pattern:

- `submit_*.pbs` declares scheduler resources, reads environment overrides,
  creates a task-specific log, and calls a shell runner;
- `run_*.sh` activates the bucex environment and invokes the corresponding
  Python example.

The default virtual environment is `${HOME}/venvs/bucex`. Override it when
submitting if necessary:

```bash
qsub -v BUCEX_VENV=/path/to/venv \
  examples/job_scripts/submit_uccle_componentwise_laplace.pbs
```

## 1. Full-record approximate screen

```bash
SCREEN_JOB=$(qsub examples/job_scripts/submit_uccle_componentwise_laplace.pbs)
```

This writes
`results/componentwise_screen/full_record_laplace.bucex`. A screen fitted only
from 1980 cannot initialize PGAS fitted from 1892 because the stored state paths
have different lengths.

## 2. Four independent exact chains

After the screen succeeds:

```bash
PGAS_JOB=$(qsub -W depend=afterok:${SCREEN_JOB} \
  examples/job_scripts/submit_uccle_componentwise_pgas.pbs)
```

The array directive is `#PBS -t 1-4%2`: four independent chains, at most two
running concurrently. Each task saves one file under `results/hpc_chains/`.
Every Python process uses `chains=1`; the array tasks themselves are the four
independent chains.

## 3. Combine

Submit after all four chain archives exist:

```bash
qsub examples/job_scripts/submit_uccle_componentwise_combine.pbs
```

Array-dependency syntax differs among PBS/Torque installations, so manual
submission of this inexpensive final step is the most portable approach.

## Runtime overrides

Every PBS file exposes ordinary environment variables. For example:

```bash
qsub -v PARTICLES=256,DRAWS=200,WARMUP=200,WORKERS=6 \
  examples/job_scripts/submit_uccle_componentwise_pgas.pbs
```

Use that as a timing and particle-behaviour pilot before requesting the full
2,000 warm-up and 2,000 retained draws at 512 particles. The package currently
saves each chain only when it finishes; a wall-time kill loses that unfinished
chain.

For shared slab multipliers, submit both the screen and PGAS stages with
`POOL=both`. Never initialize a `POOL=selection` PGAS job from a `POOL=both`
screen or conversely.
