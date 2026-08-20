# bucex 2.6.2

`bucex` fits Bayesian unobserved-components models to Gaussian and generalized
extreme-value observations. The package combines a declarative structural
model API, componentwise SSVS, approximate Laplace state updates, and exact-
density PGAS state updates.

Version 2.6.2 focuses the research interface around seven transparent scripts:

1. the complete Uccle record from 1892 and the evolution of TXx;
2. matched GEV shape and scale simulations;
3. six explicit structural simulations;
4. selection recovery with Laplace state updates;
5. the same recovery experiment with Laplace-initialized PGAS;
6. Laplace analysis of TXx, TXn, TNx, and TNn;
7. PGAS analysis of the same four series.

There is no presentation workflow layer. Every script constructs its models
with `bx.Model`, `bx.LocalLinearTrend`, `bx.DummySeasonal`, and `bx.GEV`, then
calls `bx.simulate`, `bx.fit`, `FitResult` summaries, and plotting methods
directly.

## Installation

```bash
python -m pip install ".[plot,test]"
python -c "import bucex; print(bucex.__version__)"
```

## Model

For a monthly block extreme,

\[
Y_t\mid\eta_t,\sigma,\xi\sim
\operatorname{GEV}(\eta_t,\sigma,\xi),
\qquad \eta_t=\mu_t+\gamma_t,
\]

subject to \(1+\xi(Y_t-\eta_t)/\sigma>0\). A local-linear component is

\[
\mu_{t+1}=\mu_t+\beta_t+s_\mu z^\mu_{t+1},\qquad
\beta_{t+1}=\beta_t+s_\beta z^\beta_{t+1},
\]

and the dummy seasonal component may have innovation coefficient \(s_\gamma\).
The Fruehwirth--Schnatter representation estimates signed coefficients and
standard-normal non-centred states, keeping the important neighbourhood near
zero accessible without an inverse-gamma process-variance prior.

Componentwise SSVS assigns:

- level: `fixed` or `dynamic`;
- slope: `zero`, `fixed`, or `dynamic`;
- seasonal cycle: `zero`, `fixed`, or `dynamic`.

`sigma` and `xi` are inferred but constant through time in this release.

## Direct API

```python
import numpy as np
import bucex as bx

y = bx.load_uccle_series("TXx", start="1892-01-01")

model = bx.Model(
    bx.GEV(xi_bounds=(-0.5, 0.5)),
    (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=12, mode="dynamic"),
    ),
)

priors = bx.ssvs_gev_priors(
    period=12,
    alpha_mean=float(np.median(y)),
    beta_mean=0.0,
    beta_sd=0.004,
    innovation_slab_sd={"level": 0.05, "trend": 0.00010, "season": 0.09},
    level_dynamic_probability=0.5,
    trend_probabilities=(1/3, 1/3, 1/3),
    season_probabilities=(1/3, 1/3, 1/3),
)

laplace = bx.fit(
    y,
    model=model,
    priors=priors,
    engine="laplace",
    parameterization="fruehwirth_schnatter",
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=26001),
)

pgas = bx.fit(
    y,
    model=model,
    priors=laplace.priors,
    engine="pgas",
    parameterization="fruehwirth_schnatter",
    init=laplace,
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=26002),
    particles=bx.Particles(n=512, proposal="guided"),
)

print(pgas.component_probabilities())
print(pgas.diagnostics()["engine"])
pgas.plot("season")
```

The new `season` plot overlays one trajectory per phase. At cycle/year \(i\)
and phase \(j\), it plots posterior summaries of \(\mu_{ij}+\gamma_{ij}\).
This exposes changes in the seasonal pattern without a rapidly oscillating
monthly line.

## Laplace and PGAS

For GEV observations, `engine="laplace"` samples states from an iterated local
pseudo-Gaussian approximation. It is useful for screening, debugging, and
initialization, but is not labelled exact posterior inference.

`engine="pgas"` uses conditional sequential Monte Carlo with ancestor
sampling and the exact GEV observation density. Fixed or excluded components
make transitions singular; bucex evaluates transition feasibility and
ancestor weights on the corresponding affine support.

`init=laplace` transfers one coherent posterior draw: static parameters,
structural indicators, signed innovation coefficients, and the complete state
path. It changes initialization, not the PGAS invariant distribution. Inspect
particle ESS, unique ancestors, path-update rates, reference-ancestor changes,
structural switching, and GEV support diagnostics before trusting a run.

## Seven standalone examples

Set one run ID to put every result below the same timestamp:

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

Outputs are written below `results/<BUCEX_RUN_ID>/`. Set
`BUCEX_TIMESTAMP_RESULTS=0` for an unindexed `results/` directory, change the
root with `BUCEX_RESULTS_ROOT`, or set `BUCEX_OVERWRITE=1` to regenerate safe
existing artifacts.

All scientific settings remain near the top of each script. MCMC controls can
also be overridden with `BUCEX_DRAWS`, `BUCEX_WARMUP`, `BUCEX_CHAINS`,
`BUCEX_PARTICLES`, and `BUCEX_SEED`.

The simulations use period 4 and 1,000 observations for structural recovery.
Every simulated time series has its own figure. Only each scenario's level,
slope, and seasonal decomposition is shown as a three-panel figure. Shape and
scale comparisons use a common vertical range within each group.

The Uccle descriptive figures begin in 1892 and use robust local-linear LOESS
rather than a rolling median. The analysis figures include posterior
trajectories, structural probabilities, prior-to-posterior process SDs, GEV
parameters, finite endpoints where applicable, and phase-specific seasonal
trajectories.

## HPC

The seven PBS files call the same seven examples. `submit_all.sh` creates one
run ID and submits the Laplace-to-PGAS dependencies:

```bash
export BUCEX_PYTHON=/path/to/bucex_env/bin/python
export BUCEX_RESULTS_ROOT=/path/to/scratch/bucex-results
export BUCEX_PROFILE=publication
bash examples/job_scripts/submit_all.sh
```

Profiles are `smoke` (20/20, one chain, 32 particles), `pilot` (250/250, two
chains, 128 particles), and `publication` (2,000/2,000, four chains, 512
particles). See `examples/job_scripts/README.md` and adapt resource directives
to the local cluster.

## Risk summaries

A fitted GEV result retains upper/lower-tail orientation and provides
time-varying endpoint, exceedance-probability, return-period, and return-level
draws:

```python
levels = pgas.return_level_draws(20)
periods, years = pgas.return_period_draws(35.0, annual=True)
endpoint = pgas.endpoint_draws(original_scale=True)
```

For monthly data, annual exceedance probabilities compose the twelve fitted
monthly probabilities. These remain model-based, time-indexed posterior risk
summaries, not stationary return levels.

## Validation

```bash
python -m pytest
python validation/run_release_validation.py
python -m build
```

See `docs/INFERENCE_MATRIX.md`, `docs/UCCLE.md`, and `docs/VALIDATION.md` for
the detailed contracts.
