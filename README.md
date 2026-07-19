# bucex v0.3.3

`bucex` provides Bayesian structural state-space models for ordinary and extreme
time series. Version 0.3.3 keeps the v0.3.2 samplers and regularisation unchanged,
adds explicit reasons whenever a DGEV iteration is restored, and provides a
short multi-chain Torque workflow for the six Uccle series.

## Prior profiles

```python
fit_bayes(..., priors="manuscript")   # original shared Bayesian lasso
fit_bayes(..., priors="normal")       # scale-aware Normal priors
fit_bayes(..., priors="regularized")  # component-wise Bayesian lasso
fit_bayes(..., priors="ssvs")         # zero/fixed/dynamic selection
```

The non-centred signed innovation scales are

```text
s_level, s_trend, s_season
```

with process variances `q_k = s_k**2`.

The `manuscript` profile remains unchanged. The new `regularized` profile uses
separate shrinkage parameters and coefficient scales for the three structural
blocks. Its monthly defaults are:

```text
level  : 0.03 °C
trend  : 0.0002 °C per month
season : 0.03 °C
```

These are starting scales for sensitivity analysis, not universal constants.

## Component-wise Bayesian lasso

```python
from bucex import ComponentwiseBayesianLassoPrior

prior = ComponentwiseBayesianLassoPrior(
    coefficient_scale={
        "level": 0.03,
        "trend": 0.0002,
        "season": 0.03,
    },
    a_lambda={"level": 2.0, "trend": 2.0, "season": 2.0},
    b_lambda={"level": 1.0, "trend": 1.0, "season": 1.0},
)
```

For component `k`, the hierarchy is

```text
s_k | tau_k ~ Normal(0, variance_scale * coefficient_scale_k² * tau_k)
tau_k | lambda2_k ~ Exponential(lambda2_k / 2)
lambda2_k ~ Gamma(a_k, b_k)
```

The saved posterior contains `lambda2_level`, `lambda2_trend`, and
`lambda2_season`.

## Structural SSVS

The level is always present and is either fixed or dynamic. Trend and
seasonality can be zero, fixed, or dynamic:

```text
level  : fixed | dynamic
trend  : zero  | fixed | dynamic
season : zero  | fixed | dynamic
```

This gives 18 candidate structures for the default model. The v0.3.2 SSVS
profile uses a monthly slope prior `beta0 ~ Normal(0, 0.005²)` and calibrated
innovation slabs. This prevents a very diffuse monthly slope slab from
artificially excluding the trend through a marginal-likelihood penalty.

```python
fit = fit_uccle_series(
    "TXm",
    data_dir="data",
    priors="ssvs",
    n_iter=20_000,
    burn=5_000,
)

print(fit.component_probabilities())
print(fit.component_transition_summary())
print(fit.structural_model_probabilities().head())
```

SSVS probabilities remain sensitive to slab scales, prior model probabilities,
and chain mixing. Multiple seeds and transition diagnostics should be used.
For DGEV fits, selection is based on Laplace pseudo-observations and is marked
as approximate in `fit.meta`.

## Finite-period warming rates

The instantaneous latent slope can remain weakly identified when both the level
and slope receive innovations. Version 0.3.2 therefore provides summaries based
directly on posterior changes in the fitted level:

```python
periods = {
    "early": (1892, 1949),
    "mid": (1950, 1979),
    "recent": (1980, 2022),
}

print(fit.period_rate_summary(periods))
print(
    fit.rate_contrast_summary(
        recent=(1980, 2022),
        reference=(1950, 1979),
    )
)
```

Rates are returned in °C per decade by default. These summaries include all
changes in the level trajectory and do not depend on whether the model assigns
them to level or slope innovations.

## Numerical covariance repair

The FFBS sampler now:

- uses the Joseph covariance update in the Kalman filter;
- attempts Cholesky sampling with increasing jitter;
- projects to the nearest numerical positive-semidefinite covariance only as a
  final fallback;
- never delegates an indefinite covariance to NumPy's warning-based fallback.

## Install

```bash
python -m pip install -e .
```

or install the wheel:

```bash
python -m pip install bucex-0.3.3-py3-none-any.whl
```

## Local test

```bash
python -m pytest -q

python examples/fit_uccle_series.py \
    --series TXm \
    --data-dir data \
    --out-dir results/local_regularized \
    --priors regularized \
    --n-iter 100 \
    --burn 50
```

## HPC

```bash
qsub jobs/fit_uccle_array.pbs
```

Choose a profile at submission time:

```bash
qsub -v BUCEX_PRIORS=regularized jobs/fit_uccle_array.pbs
qsub -v BUCEX_PRIORS=ssvs jobs/fit_uccle_array.pbs
```

Outputs are written by chain under `results/uccle_v033_<profile>/chains/`.

## Poster figures

```bash
python examples/make_uccle_poster_figures.py \
    --fit-dir results/uccle_v033_regularized/fits \
    --out-dir results/uccle_v033_regularized/poster
```

The acceleration and period-rate panels now use finite changes in the posterior
level trajectories rather than period averages of the instantaneous slope.


## Laplace restoration diagnostics

The DGEV Laplace update is unchanged. A restored progress line now identifies
why all retry attempts failed, for example:

```text
[it 500/8000] ... [restored attempts=25 reasons=(support_after_state_parameters:25)] window_restored=94/100 window_failures=(support_after_state_parameters:2350)
```

The fitted object stores:

```python
fit.meta["restored_iterations"]
fit.meta["restored_fraction"]
fit.meta["attempt_failure_counts"]
fit.meta["restore_failure_counts"]
```

Typical reason labels include `support_after_state_parameters`,
`support_after_observation_parameters`, and staged exception labels such as
`laplace_ffbs:LinAlgError`.

## Short parallel Uccle chains on Torque

The default PBS array launches three independent chains for each of six series:

```bash
mkdir -p logs
qsub jobs/fit_uccle_array.pbs
```

This creates 18 one-core tasks. Each chain uses 8,000 iterations and 1,500
burn-in by default. Override these values at submission time:

```bash
qsub -v BUCEX_N_ITER=6000,BUCEX_BURN=1000,BUCEX_PRIORS=regularized jobs/fit_uccle_array.pbs
```

For a targeted TNx diagnostic:

```bash
qsub -v BUCEX_SERIES=TNx,BUCEX_CHAIN_ID=4,BUCEX_N_ITER=1000,BUCEX_BURN=200 jobs/fit_uccle_single.pbs
```

After all three chains finish, pool them and write basic diagnostics:

```bash
python examples/pool_uccle_chains.py \
    --chains-dir results/uccle_v033_regularized/chains \
    --out-dir results/uccle_v033_regularized
```

The pooled files under `results/uccle_v033_regularized/fits` can then be passed
to `examples/make_uccle_poster_figures.py`. Pooling is a convenience step, not
a substitute for checking `diagnostics/split_rhat.csv` and
`diagnostics/restoration_summary.csv`.
