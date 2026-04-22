# bucex v0.1.0

`bucex` is a modular Bayesian structural state-space package for Gaussian and
extreme-value time series, with interchangeable inference backends and direct
support for dynamic risk estimation.

## What v0.1 includes

- structural state-space composition via components
- Gaussian and GEV observation models
- exact Gaussian Kalman / RTS / FFBS path
- Laplace and particle state backends already exposed for non-Gaussian models
- centered Gaussian and centered GEV Gibbs-family fitting
- a public `fit_bayes(...)` entry point
- compatibility import path `bucex.observation`
- a first `risk/` namespace for exceedance probabilities, return periods,
  return levels, and finite-endpoint trajectories

## Example

```python
from bucex import StructuralModel, LocalLinearTrend, DummySeasonal, GEVObs, fit_bayes
from bucex.inference.fit.priors import (
    GibbsConfig,
    CenteredGEVPriors,
    InitialStatePriors,
    InverseGammaPrior,
    NormalPrior,
)

model = StructuralModel(
    components=[
        LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        DummySeasonal(period=12, mode="dynamic"),
    ],
    obs=GEVObs(),
)

priors = CenteredGEVPriors(
    log_sigma=NormalPrior(mean=0.0, sd=1.0),
    xi=NormalPrior(mean=0.0, sd=0.2),
    q_level=InverseGammaPrior(a=2.0, b=0.1),
    q_trend=InverseGammaPrior(a=2.0, b=0.1),
    q_season=InverseGammaPrior(a=2.0, b=0.1),
    initial=InitialStatePriors(),
)

fit = fit_bayes(
    y=y,
    model=model,
    priors=priors,
    init_params_state={"q_level": 0.01, "q_trend": 0.001, "q_season": 0.01},
    init_params_obs={"sigma": 1.0, "xi": -0.1},
    config=GibbsConfig(n_iter=2000, burn=500, thin=2),
    state_method="laplace",
)
```
