# bucex architecture notes

## Positioning

`bucex` should be positioned as a **modular Bayesian state-space framework with pluggable observation models and interchangeable inference backends**, not merely as another BSTS package.

That gives you two layers:

1. a **general engine** for structural state-space models;
2. an **extremes layer** where GEV / DGEV is the flagship application.

This matters because the package becomes distinctive only if Gaussian structural time series is treated as one special case among several.

---

## What is already strong in the current package

The current code already has the right broad separation:

* `components/` for structural blocks;
* `models/` for composing components into a full model;
* `obs/` for observation equations;
* `inference/state/` for conditional state inference;
* `inference/fit/` for full Bayesian fitting;
* `simulate/` for generative simulation;
* `core/results.py` for standardized outputs.

That is the right overall direction.

The most promising design idea already present is:

* **linear-Gaussian latent state evolution** as the common substrate;
* **observation model modularity** on top of that;
* **multiple inference engines** depending on how much of the model remains tractable.

That is exactly the abstraction you should preserve.

---

## Main architectural issue in the current code

Right now the package is still halfway between:

* a research prototype for your DGEV manuscript;
* and a general reusable library.

The biggest risk is that the public API becomes organized around implementation accidents rather than around statistical concepts.

In particular, the current code still mixes:

* model specification,
* parameter naming,
* inference decisions,
* and representation choices such as centered vs noncentered.

These should be separated more sharply.

---

## Guiding design principles

### 1. Separate model specification from inference

A user should be able to define a model once and then choose among:

* exact Kalman filter / smoother / FFBS,
* Laplace approximation,
* particle filter / particle Gibbs,
* future PMMH or SMC-based methods.

That means the model object should describe only the generative structure, not how inference will be done.

### 2. Treat parameterization as an inference concern

Centered vs noncentered should usually not define separate end-user model classes.

Instead:

* the model stays the same;
* the inference backend decides how to represent latent states and process scales internally.

So `CenteredGEVGibbs` and `NonCenteredGEVGibbs` should ultimately become implementations of the same fitter family rather than two disconnected worlds.

### 3. Make components composable but typed

You want components like:

* local level,
* local linear trend,
* seasonal,
* cycle,
* regression,
* intervention,
* stochastic volatility / time-varying scale,
* maybe shape dynamics later.

Each component should expose a very clear contract:

* state contribution,
* innovation contribution,
* predictor contribution,
* prior contribution,
* optional inclusion indicator for model selection.

### 4. Support multiple linear predictors

A single scalar `eta_t` is too restrictive for the long run.

You should plan immediately for named predictors, e.g.

* `location`,
* `log_scale`,
* `shape`.

Even if only `location` is implemented first, the architecture should not assume forever that `eta_dim = 1`.

### 5. Keep the Gaussian path first-class

The Gaussian case is not only a baseline. It is your exact engine:

* exact filter,
* exact smoother,
* exact FFBS,
* exact Gibbs blocks where possible.

That gives you a gold standard for testing and for benchmarking approximate methods.

---

## Proposed package structure

A cleaner long-run structure would be:

```text
bucex/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── model_builders.py
│   ├── fit.py
│   ├── forecast.py
│   └── diagnose.py
│
├── core/
│   ├── typing.py
│   ├── time_index.py
│   ├── params.py
│   ├── results.py
│   ├── priors.py
│   └── exceptions.py
│
├── components/
│   ├── base.py
│   ├── level.py
│   ├── trend.py
│   ├── seasonal.py
│   ├── cycle.py
│   ├── regression.py
│   ├── intervention.py
│   ├── volatility.py
│   └── compose.py
│
├── observation/
│   ├── base.py
│   ├── gaussian.py
│   ├── student.py
│   ├── poisson.py
│   ├── bernoulli.py
│   ├── gev.py
│   ├── gpd.py
│   ├── links.py
│   └── predictor_map.py
│
├── models/
│   ├── base.py
│   ├── structural.py
│   └── compiled.py
│
├── inference/
│   ├── base.py
│   ├── state/
│   │   ├── kalman.py
│   │   ├── smoother.py
│   │   ├── ffbs.py
│   │   ├── laplace.py
│   │   ├── particle_filter.py
│   │   ├── particle_smoother.py
│   │   └── particle_gibbs.py
│   ├── fit/
│   │   ├── base.py
│   │   ├── gibbs.py
│   │   ├── mh.py
│   │   ├── pmmh.py
│   │   ├── variational.py
│   │   ├── gaussian_structural.py
│   │   ├── gev_structural.py
│   │   └── model_selection.py
│   └── dispatch.py
│
├── risk/
│   ├── exceedance.py
│   ├── return_levels.py
│   ├── return_periods.py
│   ├── endpoints.py
│   └── forecast_risk.py
│
├── diagnostics/
│   ├── residuals.py
│   ├── pit.py
│   ├── calibration.py
│   ├── cv.py
│   └── mcmc.py
│
├── simulate/
│   ├── statespace.py
│   └── forecast.py
│
├── plotting/
│   ├── states.py
│   ├── components.py
│   ├── forecasts.py
│   ├── risk.py
│   └── diagnostics.py
│
└── io/
    ├── save.py
    ├── load.py
    └── serialize.py
```

Notes:

* `obs/` should probably be renamed to `observation/` for clarity.
* A top-level `api/` is useful so users do not need to import from deep internal modules.
* `risk/` deserves its own namespace because that is one of the unique contributions of the package.
* `compiled.py` can be a frozen, backend-ready representation of a model if you want efficiency later.

---

## Core abstractions

## A. Structural component

A component should do one thing: define a block contribution to the latent system and to one or more predictors.

Conceptually:

```python
class Component(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def state_dim(self) -> int: ...

    @property
    def noise_dim(self) -> int: ...

    def initial_block(self, params, time_index) -> InitialBlock: ...

    def transition_block(self, t, params, time_index) -> TransitionBlock: ...

    def predictor_block(self, t, params, time_index, exog_t=None) -> PredictorBlock: ...

    def default_priors(self): ...
```

A predictor block should not assume a single scalar predictor. It should be keyed by predictor name.

For example:

```python
PredictorBlock(
    loadings={
        "location": Z_loc,
        "log_scale": Z_scale,
    },
    offsets={
        "location": d_loc,
        "log_scale": d_scale,
    },
)
```

That immediately solves the future extension to dynamic scale.

---

## B. Observation model

The observation model should be responsible for:

* declaring which predictors it needs,
* transforming predictors into natural parameters,
* evaluating log density,
* sampling,
* optionally derivatives with respect to predictors.

Something like:

```python
class ObservationModel(Protocol):
    @property
    def response_dim(self) -> int: ...

    @property
    def predictor_names(self) -> tuple[str, ...]: ...

    def natural_params(self, predictors, static_params, exog_t=None): ...

    def logpdf(self, y, predictors, static_params, exog_t=None): ...

    def sample(self, predictors, static_params, rng, exog_t=None): ...

    def grad_predictors(self, y, predictors, static_params, exog_t=None): ...

    def hess_predictors(self, y, predictors, static_params, exog_t=None): ...
```

For Gaussian:

* predictors might only contain `location`.

For GEV:

* initially `location` predictor plus static `sigma`, `xi`;
* later `location` and `log_scale` predictors.

This is cleaner than pushing everything through a generic `eta_t`.

---

## C. Structural model

The model object should mostly be a compiled composition of:

* components,
* observation model,
* predictor map,
* time index.

Something like:

```python
@dataclass
class StructuralModel:
    components: Sequence[Component]
    observation: ObservationModel
    time_index: TimeIndex
    exog_schema: ExogSchema | None = None
```

It should expose methods that return time-indexed system/design objects, but those objects should already allow multiple predictors.

---

## D. Fitter

Fitters should be organized by inference family, not only by observation family.

For example:

* `GibbsStructuralGaussian`
* `GibbsStructuralGEV`
* `PMMHStructuralGEV`
* `LaplaceMAPStructuralGEV`
* `SSVSStructuralGaussian`
* `SSVSStructuralGEV`

Internally they can still reuse shared utilities.

---

## What I would change first in the current code

## 1. Add package metadata and public entry points

Right now the zip does not show `__init__.py` files or a clear public API.

First step:

* add `bucex/__init__.py`;
* add module-level exports;
* define the small official entry points you want users to rely on.

For example:

```python
from .models.structural import StructuralModel
from .components.trend import LocalLinearTrend
from .components.seasonal import DummySeasonal
from .observation.gaussian import GaussianObs
from .observation.gev import GEVObs
from .inference.fit.gaussian_structural import fit_gaussian_structural
from .inference.fit.gev_structural import fit_gev_structural
```

---

## 2. Rename `obs` to `observation`

This is a minor but worthwhile readability improvement.

---

## 3. Generalize away from scalar `eta`

This is the single most important architectural refactor.

Your current model API hard-codes the idea that the state maps to one scalar predictor and then `obs_params()` inserts that as `mu`.

That is fine for the current manuscript, but it will become a bottleneck as soon as you want:

* dynamic scale,
* dynamic shape,
* count models with link functions,
* regression effects in different submodels.

So I would replace:

* `eta_dim`
* `LinearDesign(Z, d)`
* `obs_params(...)`

with a named-predictor representation.

---

## 4. Move centered vs noncentered out of model naming

You currently have separate fit modules for centered and noncentered models.

Instead I would think in terms of:

* `parameterization="centered" | "noncentered" | "auto"`

as a fitter option.

That will stop the codebase from duplicating model logic.

---

## 5. Separate exact state methods from approximate state methods

The current `dispatch.py` is a good start, but I would make the conceptual split sharper:

* exact linear-Gaussian methods,
* local Gaussian approximation methods,
* particle methods.

That leads naturally to a capability matrix:

| observation     | filter      | smoother    | state draw                         | full Bayes      |
| --------------- | ----------- | ----------- | ---------------------------------- | --------------- |
| Gaussian        | exact       | exact       | exact FFBS                         | Gibbs           |
| GEV + Laplace   | approximate | approximate | approximate FFBS                   | MH-within-Gibbs |
| GEV + particles | particle    | particle    | particle Gibbs / ancestor sampling | PMCMC           |

The dispatcher should know this matrix.

---

## 6. Introduce a risk layer now

Your manuscript is not just about fitting. It is about turning fitted state-space extremes into:

* exceedance probabilities,
* return periods,
* return levels,
* endpoint trajectories,
* forecasts of risk.

That should be explicit in the package.

So add functions like:

```python
def exceedance_probability(bundle, threshold, block="monthly") -> RiskTrajectory

def return_period(bundle, threshold, block="monthly") -> RiskTrajectory

def endpoint_trajectory(bundle) -> RiskTrajectory

def forecast_exceedance(bundle, threshold, horizon) -> ForecastRiskResult
```

This is one of the clearest ways to make `bucex` feel different from a generic BSTS library.

---

## Model selection / automatic structural discovery

This is worth doing, but I would make it phase 2, not phase 1.

The first version should fit a user-specified model reliably.

Then phase 2 introduces automatic selection.

## How I would formulate it

There are really two different selection problems:

### A. Component inclusion

Examples:

* include trend or not;
* include stochastic slope or keep slope fixed;
* include seasonal dynamics or keep seasonality fixed;
* include cycle or not;
* include intervention or not.

This is structural model search.

### B. Regression selection

Examples:

* include NAO;
* include ENSO;
* include circulation proxy;
* allow effect on location only or also on scale.

This is variable selection.

These should not be conflated.

## Recommended design

Create a `selection/` submodule or an `inference/fit/model_selection.py` module with:

* SSVS indicators for regression coefficients;
* SSVS indicators for component state variances or component activation;
* optional spike-and-slab on whole component blocks.

The natural user interface is something like:

```python
fit = fit_structural_bayes(
    y,
    model,
    method="gibbs",
    selection={
        "components": "ssvs",
        "regression": "ssvs",
    },
)
```

But under the hood the component-selection indicators should be explicit latent variables.

### Important modeling distinction

For a component you often want to distinguish three cases:

1. absent;
2. present but fixed;
3. present and dynamic.

That is more useful than a simple binary on/off indicator.

For example, seasonality may be:

* absent,
* fixed seasonal dummies,
* dynamic seasonal state.

That matches your manuscript very well.

---

## Recommended immediate MVP

## Version 0.1

The first public research version should do a small number of things very well:

### Gaussian structural model

* local level / trend / dummy seasonality / regression
* exact Kalman filter
* exact RTS smoother
* exact FFBS
* Gibbs fitting with shrinkage options

### GEV structural model

* location-driven DGEV
* local level / trend / dummy seasonality
* Laplace filter / smoother / state draw
* MH-within-Gibbs for static GEV parameters
* centered and noncentered parameterization options

### Common

* simulation
* forecasting
* PIT / PP diagnostics
* rolling-origin CV
* risk trajectories for GEV

That is already enough for a strong methods package accompanying the paper.

---

## Version 0.2

* dynamic scale for Gaussian and GEV
* intervention component
* regression component with exogenous inputs
* particle filter and particle smoother cleaned up
* stronger plotting API
* better serialization of fitted objects

---

## Version 0.3

* SSVS for components and regressors
* PMMH / particle Gibbs with ancestor sampling
* cycle component
* richer observation families
* online update / filtering interface

---

## Suggested user-facing API

I would aim for something like this:

```python
from bucex import (
    StructuralModel,
    LocalLinearTrend,
    DummySeasonal,
    GaussianObs,
    GEVObs,
    fit_bayes,
)

model = StructuralModel(
    components=[
        LocalLinearTrend(level=True, slope=True),
        DummySeasonal(period=12, mode="dynamic"),
    ],
    observation=GEVObs(location_link="identity"),
)

fit = fit_bayes(
    y,
    model,
    method="gibbs",
    state_method="laplace",
    parameterization="noncentered",
)

risk = fit.return_period(threshold=39.7)
forecast = fit.forecast(horizon=24)
```

This is much cleaner than exposing users directly to `CenteredGEVGibbs` unless they want low-level control.

Low-level classes can still exist.

---

## Immediate code cleanup checklist

1. Add missing `__init__.py` files.
2. Decide the official package name: `bucex` or `sts_extremes`, but not both.
3. Remove stale bytecode and `__pycache__` from the repository.
4. Introduce a small public API surface.
5. Generalize design objects to named predictors.
6. Move centered / noncentered into fitter configuration.
7. Create a `risk/` namespace.
8. Write tests first for Gaussian exact methods.
9. Use the Gaussian case as a benchmark for every later approximation.
10. Add one fully reproducible notebook per major workflow.

---

## What I think the package should claim

A good one-sentence claim would be:

**`bucex` is a modular Bayesian structural state-space package for Gaussian and extreme-value time series, with interchangeable inference backends and direct support for dynamic risk estimation.**

A stronger research-facing version:

**`bucex` unifies structural time series decomposition, non-Gaussian observation models, and Bayesian state inference in a single extensible framework, with dynamic GEV models for evolving environmental risk as a flagship application.**

---

## Recommended next step

Before writing more code, I would write three short design documents:

1. **Core abstractions**: component, observation model, structural model, fitter, result object.
2. **Capability matrix**: which observation families work with which state inference backends.
3. **Selection design**: how SSVS acts on regressors and on structural components.

Then refactor the package around those documents.

---

## My concrete recommendation

Do **not** start by adding many new models.

Instead, first lock in the architecture for:

* multiple predictors,
* backend-agnostic model objects,
* a clean high-level API,
* a separate risk layer,
* and a future model-selection mechanism.

That will save you a lot of pain later.
