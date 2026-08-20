# bucex 2.6.0

`bucex` fits Bayesian unobserved-components models to Gaussian and generalized
extreme-value observations. Version 2.6.0 provides one focused, reproducible
workflow for the COMPSTAT temperature-extremes presentation:

1. introduce the Uccle record and the evolution of TXx;
2. compare heavy, Gumbel, and bounded GEV tails under the same local level;
3. simulate absent, fixed, and dynamic trend/seasonal components;
4. recover those components with componentwise SSVS and a Laplace state update;
5. repeat the identical experiment with Laplace-initialized PGAS;
6. analyse TXx, TXn, TNx, and TNn in that order;
7. export selection probabilities, posterior trajectories,
   prior-to-posterior plots, parameter summaries, and algorithm diagnostics.

The old presentation scripts have been removed. The numbered scripts under
`examples/presentation/` and the PBS dependency graph under
`examples/job_scripts/` are the complete supported presentation surface.

## Installation

```bash
python -m pip install ".[plot,test]"
```

Verify the resolved plan without fitting anything:

```bash
bucex-presentation plan --profile smoke
```

Run the complete small software check:

```bash
bucex-presentation run all \
  --profile smoke \
  --output-dir results/smoke \
  --no-progress
```

The smoke output is not suitable for inference. Runtime profiles are explicit:

| Profile | Uccle window | Draws/warmup | Chains | Particles | Simulation months |
|---|---|---:|---:|---:|---:|
| `smoke` | 2015–2022 | 2/2 | 1 | 24 | 48 |
| `pilot` | 1980–2022 | 250/250 | 2 | 128 | 240 |
| `publication` | 1892–2022 | 2,000/2,000 | 4 | 512 | 720 |

Publication defaults are starting values. Final particle counts and chain
lengths must be justified by the exported mixing, support, ancestor, and
posterior-stability diagnostics.

## Model

For a monthly block extreme,

\[
Y_t\mid\eta_t,\sigma,\xi \sim
\operatorname{GEV}(\eta_t,\sigma,\xi),
\qquad \eta_t=\mu_t+\gamma_t,
\]

with support

\[
1+\xi(Y_t-\eta_t)/\sigma>0.
\]

The local-linear component is

\[
\begin{aligned}
\mu_{t+1} &= \mu_t+\beta_t+s_\mu z^\mu_{t+1},\\
\beta_{t+1} &= \beta_t+s_\beta z^\beta_{t+1},
\end{aligned}
\]

and the dummy seasonal component may have innovation coefficient
\(s_\gamma\). The Fruehwirth–Schnatter representation estimates signed
coefficients \(s_k\) and standard-normal non-centred states. This keeps the
scientifically important neighbourhood \(s_k\approx0\) accessible without an
inverse-gamma prior on the corresponding process variance.

Componentwise SSVS has a directly interpretable model space:

- level: `fixed` or `dynamic`;
- slope: `zero`, `fixed`, or `dynamic`;
- seasonal cycle: `zero`, `fixed`, or `dynamic`.

`zero` removes the contribution, `fixed` retains a deterministic component,
and `dynamic` activates its process innovation. The presentation prior gives
equal prior probability to the three slope and seasonal states and probability
one half to a dynamic level.

In this release, SSVS acts on the location-predictor components. `sigma` and
`xi` are inferred but constant through time within a series. Time-varying scale
or shape requires an additional linked predictor and support-aware transition;
it is not silently approximated by selecting a location component.

```python
import numpy as np
import bucex as bx

y = bx.load_uccle_series("TXx", start="1980-01-01")
prior = bx.presentation_gev_prior(alpha_mean=float(np.median(y)))

laplace = bx.fit(
    y,
    family="gev",
    period=12,
    priors=prior,
    engine="laplace",
    parameterization="fruehwirth_schnatter",
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=26001),
)

pgas = bx.fit(
    y,
    family="gev",
    period=12,
    priors=prior,
    engine="pgas",
    parameterization="fruehwirth_schnatter",
    init=laplace,
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=26002),
    particles=bx.Particles(n=512, proposal="guided"),
)

print(pgas.component_probabilities())
print(pgas.diagnostics()["engine"])
```

`init=laplace` selects one finite, high-log-posterior draw and transfers its
static parameters, signed innovation coefficients, and complete centred state
path. The target PGAS sampler maps that path back to its non-centred state.
Initialization changes burn-in, not the PGAS invariant distribution.

## Laplace and PGAS are not interchangeable claims

For GEV observations, `engine="laplace"` iterates a local pseudo-Gaussian
approximation and samples the resulting state approximation. It is useful for
screening, debugging, initialization, and an explicit approximation-versus-
PGAS comparison. It is not marked as exact posterior inference.

`engine="pgas"` uses conditional sequential Monte Carlo with ancestor
sampling and the exact GEV observation density. Its invariant kernel targets
the specified posterior. Fixed or excluded UC components make transition
covariances singular. The implementation evaluates transition feasibility and
ancestor weights on the corresponding affine support rather than pretending a
full-rank Gaussian density exists.

That support-aware calculation prevents an invalid density evaluation; it does
not make path degeneracy disappear. This release uses standard one-step
ancestor sampling, not a bridge/backward-simulation extension for highly
degenerate models. Inspect at least:

- `median_min_particle_ess`;
- `mean_unique_ancestors`;
- `path_change_rate` and `mean_path_update_fraction`;
- `reference_ancestor_change_rate`;
- structural-model switching and GEV support/restoration diagnostics.

A very small reference-ancestor change rate is a direct warning that the
conditioned lineage is sticky even when the particle ESS looks acceptable.

## Simulation design

Tail-class illustrations use one local-level specification and vary only
`xi`: `+0.20`, `0`, and `-0.20`. Structural-selection experiments instead fix
`sigma=1.5` and `xi=-0.20` and vary only the UC structure:

- stationary location;
- fixed linear trend;
- stochastic local level;
- local level plus fixed seasonality;
- local level plus dynamic seasonality;
- stochastic trend plus fixed seasonality;
- fully dynamic level, trend, and seasonality.

Every simulated CSV stores the observations and true level, slope, seasonal
contribution, and linear predictor. Its adjacent JSON file stores the model and
parameter truth used by selection-recovery and prior-to-posterior figures.

## Presentation workflow

Run the scripts in order:

```bash
python examples/presentation/00_uccle_record.py
python examples/presentation/01_tail_simulations.py
python examples/presentation/02_structural_simulations.py
python examples/presentation/03_simulation_laplace.py
python examples/presentation/04_simulation_pgas.py
python examples/presentation/05_uccle_laplace.py
python examples/presentation/06_uccle_pgas.py
python examples/presentation/07_build_results.py
```

Or use the CLI for a single task:

```bash
bucex-presentation run simulation-fit \
  --profile pilot \
  --engine laplace \
  --scenario local_level_dynamic_season \
  --output-dir results/presentation

bucex-presentation run simulation-fit \
  --profile pilot \
  --engine pgas \
  --scenario local_level_dynamic_season \
  --output-dir results/presentation
```

The PGAS command fails early with a precise path if its combined Laplace fit is
absent. Existing scientific artifacts are never overwritten unless
`--overwrite` is explicit.

The result tree is stable across local and HPC execution:

```text
results/presentation/
  config.json
  manifest.json
  simulations/{tail,structure}/
  fits/simulations/{laplace,pgas}/<scenario>/
  fits/uccle/{laplace,pgas}/<series>/
  tables/
  figures/
  logs/
```

Each combined fit exports parameter and convergence summaries, algorithm
diagnostics, the posterior trajectory, structural probabilities, model
switching, and figures for trajectories, selection, process-scale priors and
posteriors, `sigma`, `xi`, and the finite endpoint where relevant. Aggregate
figures compare selection recovery across engines and the four Uccle series.

## HPC

The PBS templates submit one independent chain per array task. Combine jobs
must succeed before PGAS jobs are released, guaranteeing that every PGAS chain
has its Laplace initializer.

```bash
export BUCEX_PYTHON=/path/to/venv/bin/python
export BUCEX_OUTPUT_DIR=/path/to/scratch/bucex-presentation
bash examples/job_scripts/submit_all.sh
```

Edit queue/account/resource directives for the local cluster. The supplied
walltimes are templates, not performance guarantees.

## Risk summaries

A fitted GEV result retains the original upper/lower-tail orientation. The
same result object provides time-varying endpoint, exceedance-probability,
return-period, and return-level draws:

```python
levels = pgas.return_level_draws(20)
periods, years = pgas.return_period_draws(35.0, annual=True)
endpoint = pgas.endpoint_draws(original_scale=True)
```

For monthly data, annual exceedance probabilities are composed from the twelve
fitted monthly probabilities. These are posterior risk summaries under the
model; they are not automatically stationary return levels and should be
reported with the time index and posterior uncertainty.

## Validation

```bash
python -m pytest
python validation/run_presentation_smoke.py
python validation/run_release_validation.py
python -m build
```

See `docs/INFERENCE_MATRIX.md`, `docs/UCCLE.md`, and `docs/VALIDATION.md` for
the detailed contracts and interpretation checks.
