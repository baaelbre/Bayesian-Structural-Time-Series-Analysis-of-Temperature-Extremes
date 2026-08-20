# PBS workflow

These jobs reproduce exactly the numbered presentation examples, while
splitting every fitted scenario/series into independent chains. The combine
jobs join those chains before downstream work starts. In particular, every
PGAS task reads a completed combined Laplace fit and uses one of its posterior
draws as the parameter and full-state-path initializer.

Before submission, install the package in the Python environment available on
compute nodes and set `BUCEX_PYTHON` if its executable is not simply `python`.
The defaults use the `publication` profile: 2,000 retained draws, 2,000 warmup
iterations, four chains, 512 particles, and 800 simulated blocks with period 4.

```bash
export BUCEX_PYTHON=/path/to/venv/bin/python
export BUCEX_OUTPUT_DIR=/path/to/scratch/bucex-presentation
bash examples/job_scripts/submit_all.sh
```

Optional exports include `BUCEX_DATA_DIR`, `BUCEX_PROFILE`, `BUCEX_CHAINS`,
`BUCEX_DRAWS`, `BUCEX_WARMUP`, `BUCEX_PARTICLES`,
`BUCEX_SIMULATION_LENGTH`, `BUCEX_SIMULATION_PERIOD`, and
`BUCEX_OVERWRITE=1`. The older `BUCEX_SIMULATION_MONTHS` name remains an alias.
The array bounds are sized
for the four-chain publication profile; surplus tasks exit immediately for a
profile with fewer chains.

The resource requests are conservative templates, not measured guarantees.
Adapt the PBS `select`, memory, walltime, account, and queue directives to the
local cluster policy before the final run.
