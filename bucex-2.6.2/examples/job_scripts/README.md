# PBS jobs

The seven PBS files call the seven numbered files in `examples/` directly.
There is no separate workflow API and no combine step. A PGAS script creates
or reuses its matching Laplace initializer itself.

Choose the Python executable and output root, then submit the dependency graph:

```bash
export BUCEX_PYTHON=/path/to/bucex_env/bin/python
export BUCEX_RESULTS_ROOT=/path/to/scratch/bucex-results
export BUCEX_PROFILE=publication
bash examples/job_scripts/submit_all.sh
```

`submit_all.sh` creates one `BUCEX_RUN_ID` and passes it to every job, giving
results such as `results/20260820_143000/fits/` and
`results/20260820_143000/figures/`. To append to an existing run, export its
ID before submission. To disable timestamp directories, set
`BUCEX_TIMESTAMP_RESULTS=0`.

Profiles provide convenient sampler defaults:

| profile | draws | warmup | chains | particles |
|---|---:|---:|---:|---:|
| `smoke` | 20 | 20 | 1 | 32 |
| `pilot` | 250 | 250 | 2 | 128 |
| `publication` | 2000 | 2000 | 4 | 512 |

Any explicit `BUCEX_DRAWS`, `BUCEX_WARMUP`, `BUCEX_CHAINS`, or
`BUCEX_PARTICLES` value overrides the profile. Other useful variables are
`BUCEX_DATA_DIR`, `BUCEX_START`, `BUCEX_END`, `BUCEX_N_TIME`,
`BUCEX_PERIOD`, `BUCEX_SEED`, `BUCEX_PROGRESS`, and `BUCEX_OVERWRITE=1`.

The resource requests are conservative templates. Adapt account, queue,
memory, CPU, and walltime directives to the local cluster policy.
