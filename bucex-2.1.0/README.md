# bucex 2.1.5

`bucex` fits Bayesian structural time-series models to Gaussian bulk data and
dynamic-location GEV extremes. Version 2.1 adds the manuscript's single shared
climate-factor model while keeping the existing univariate API unchanged.

## The v2.1 one-factor model

For channel (i) and month (t),

\[
\mu_{i,t}=c_i+\lambda_i f_t+S_{i,t}+\alpha_{i,t}.
\]

- (f_t) is one shared local linear trend (level and slope).
- The factor loading for `TXm` is fixed at 1; the other loadings are estimated.
- Each (\alpha_{i,t}) is an independent local level.
- Each (S_{i,t}) is a series-specific dummy-seasonal block.
- `TXm` and `TNm` use Gaussian observations. `TXx`, `TXn`, `TNx`, and
  `TNn` use GEV observations; lower extremes are handled with `tail="lower"`.

The ready-made Uccle graph is:

```python
import bucex as bx

data = bx.load_uccle_factor_data()
model = bx.make_uccle_factor_model()
compiled = bx.compile_model(model, data)
priors = bx.identified_factor_priors(
    compiled,
    profile="regularized_triple_gamma",
    triple_gamma_options={
        "spike_shape": 0.10,
        "tail_shape": 0.10,
        "learn_global": True,
        "learn_shapes": False,
    },
    smooth_factor=True,
    reference_channel="TXm",
)

fit = bx.fit(
    data,
    model,
    parameterization="fruehwirth_schnatter",  # aliases: "fs", "ncp"
    engine="pgas",
    priors=priors,
    asis=True,
    mcmc=bx.MCMC(
        draws=2_000,
        warmup=2_000,
        chains=4,
        seed=42,
        progress=True,
    ),
    particles=bx.Particles(n=1_024, proposal="guided"),
)

fit.factor("common")
fit.factor_rate_summary("common")
fit.factor_probabilities("common")
fit.loading_probability("common", "TXx", threshold=1.0)
fit.reconstructed_state("TXx")
fit.normalized_factor(slice(0, 30 * 12), "common")
fit.channel_decomposition("TXx", "common", baseline=slice(0, 30 * 12))
```

The same graph can be built explicitly and extended with existing components:

```python
model = bx.FactorModel(
    channels=(
        bx.Channel(
            "TXm",
            bx.Gaussian(),
            components=(bx.LocalLevel(), bx.DummySeasonal(period=12)),
        ),
        bx.Channel(
            "TXx",
            bx.GEV(),
            components=(bx.LocalLevel(), bx.DummySeasonal(period=12)),
        ),
        bx.Channel(
            "TXn",
            bx.GEV(),
            components=(bx.LocalLevel(), bx.DummySeasonal(period=12)),
            tail="lower",
        ),
    ),
    factors=(
        bx.Factor(
            "common",
            components=(
                bx.LocalLinearTrend(
                    initial_level=0.0,
                    initial_slope=0.0,
                    initial_level_sd=0.0,
                    initial_slope_sd=0.0,
                ),
            ),
            loadings={
                "TXm": 1.0,
                "TXx": bx.Loading.estimated(1.0, sd=1.0),
                # Internal loading starts negative because TXn is sign-reversed.
                "TXn": bx.Loading.estimated(-1.0, sd=1.0),
            },
        ),
    ),
)
```

Plain numeric loadings are fixed. `Loading.estimated(...)` values are sampled
under their stored normal priors. A fixed non-zero loading anchors the factor's
scale and sign; fixing its initial level to zero separates the shared location
from channel intercepts. Setting `initial_slope_sd=0.0` makes the declared
initial slope a true fixed coefficient while stochastic slope innovations
still allow the common rate to evolve.

## Frühwirth--Schnatter and disturbance parameterizations

The eligible one-factor graph above has a full FS non-centred representation.
Every stochastic state block has unit innovation variance. Signed innovation
scales appear in the observation design:

\[
f_t=f_0+t\beta_{f,0}
  +q_{f,\alpha}\widetilde\alpha_{f,t}
  +q_{f,\beta}A_{f,t},
\]

with analogous (q_{\alpha,i}\widetilde\alpha_{i,t}) and
(q_{\gamma,i}\widetilde\gamma_{i,t}) terms. `signed_sd.*` stores the signed
FS coefficients; `sd.*` stores their absolute values. The NCP trajectories are
available in `fit.auxiliary_draws["fs_state"]`, while `state_draws` always
contains reconstructed semantic states.

`parameterization="disturbance"` remains available for this model and for more
general factor graphs. It keeps the centered semantic states and conditions
scale updates on standardized structural disturbances. With `asis=True`, both
parameterizations interweave with a centered scale update.

For a mixed Gaussian/GEV model, PGAS evaluates the joint conditional
likelihood (the product of all channel likelihoods) for each particle. The
default guided proposal works in unit-disturbance coordinates and retains the
exact prior/proposal correction. `engine="laplace"` is available as an
explicitly labelled approximation.

Version 2.1.5 samples the loading/deviation ridge more effectively but does not
claim that an unrestricted persistent decomposition is identified by the
likelihood. For Gaussian channels, the intercept, estimated loading,
idiosyncratic innovation SD, and complete deviation path are updated through
collapsed Kalman likelihoods plus an exact three-state FFBS draw. For GEV
channels, a loading/deviation interweaving move changes both components while
preserving the complete predictor and GEV support exactly. The resolved
kernels are recorded in `fit.sampler_diagnostics["loading_kernels"]`.

## Regularized horseshoe

The factor default is `priors="regularized_horseshoe"`. Its local, global, and
regularizing-slab hierarchy is applied only to
`channel.<series>.level` innovation scales. It therefore shrinks the six
idiosyncratic deviations toward a static tether while leaving the shared trend
and series-specific seasonal innovations under calibrated process-SD priors.

This is continuous shrinkage: a scale can become arbitrarily small but is not
an exact point mass at zero. Posterior draws include:

```text
horseshoe.global
horseshoe.slab2
horseshoe.local.channel.TXx.level
sd.channel.TXx.level
```

Use `priors="regularized"` for calibrated PC priors without the horseshoe.

## Triple gamma, regularized triple gamma, and SSVS

The triple-gamma option implements the normal--gamma--gamma representation of
Cadonna, Frühwirth-Schnatter, and Knaus (2020). For a standardized signed
innovation scale (u_j),

\[
u_j\mid r_j,d_j,\phi\sim N(0,\phi r_j/d_j),\qquad
r_j\sim\operatorname{Gamma}(a,1),\quad
d_j\sim\operatorname{Gamma}(c,1).
\]

The stored shrinkage factor
(\rho_j=1/(1+\phi r_j/d_j)) is close to one when the structural innovation is
strongly suppressed and close to zero when it is effectively unshrunk.
`a=c=0.5` is the horseshoe member; Bayesian-lasso, double-gamma,
folded/half-t, and Gaussian members arise through the shapes or limits
described in the paper. The calibrated PC prior remains a separate
exponential-on-SD construction and is retained as its own profile.

```python
# Ready-made univariate profiles.
fit = bx.fit(y, family="gaussian", priors="triple_gamma",
             parameterization="fruehwirth_schnatter")
fit_regularized = bx.fit(
    y,
    family="gaussian",
    priors="regularized_triple_gamma",
    parameterization="fruehwirth_schnatter",
)

# Direct control of a, c, global-scale learning, and the optional slab.
priors = bx.triple_gamma_gaussian_priors(
    spike_shape=0.5,
    tail_shape=0.5,
    learn_global=False,
    global_scale=1.0,
)
```

The same `triple_gamma` and `regularized_triple_gamma` profile names work in
factor models; there they target namespaced idiosyncratic local-level
innovations. `ssvs` is different: it assigns literal posterior mass to
zero/fixed/dynamic structures and is currently a univariate structural
profile. Inspect `component_probabilities()` and
`component_transition_summary()` rather than treating a constant indicator as
an ordinary continuous MCMC parameter.

## Interpreting loadings and rates

With `TXm=1`, a posterior probability
`fit.loading_probability("common", "TXx", threshold=1)` quantifies whether the
`TXx` *shared-factor contribution* responds more strongly than `TXm`.
Idiosyncratic (\alpha_{i,t}) can still change a channel's total rate, so use
`channel_rate_draws()` or `reconstructed_state()` when the scientific claim is
about the complete `TXx` trajectory. This distinction prevents a loading
contrast from being overstated as a total-trend result.

`channel_decomposition()` returns baseline, shared contribution, dynamic
deviation, seasonality, and complete predictor draws and verifies their sum
numerically.
This avoids mistaking `state("channel.<name>.level")`, which contains the
intercept plus deviation, for the deviation alone.

An anchored loading identifies factor scale and sign, but it does not on its
own distinguish `lambda[i] * f[t]` from a persistent `alpha[i,t]`. The helper
`identified_factor_priors()` makes the structural restrictions explicit:

```python
priors = bx.identified_factor_priors(
    compiled,
    smooth_factor=True,          # no direct factor-level shock
    reference_channel="TXm",    # no TXm idiosyncratic level shock
)
```

Use `fixed_idiosyncratic="all"` for a loading-only sensitivity analysis, or
remove these restrictions deliberately for a weak-identification stress test.
After fitting, inspect:

```python
fit.factor_identification_diagnostics()
fit.loading_deviation_correlation("TXx", summary="factor_projection")
fit.idiosyncratic_innovation_draws("TXx")
```

A large loading--deviation correlation or `ridge_flag=True` means the complete
predictor is more interpretable than its shared and idiosyncratic pieces.

## Simple plotting API

Every result uses `fit.plot(kind, ...)`. Univariate fits expose the complete
predictor separately from individual structural components, together with
prior/posterior and chain diagnostics:

```python
fit.plot("predictor")                      # complete eta, including seasonality
fit.plot("level_slope")                    # structural level and latent slope
fit.plot("process_sd", truths=truth)       # prior, posterior, and optional truth
fit.plot("parameter_density", parameters=["sigma", "xi"])
fit.plot("traces")                         # every innovation SD, by chain
fit.plot("acf")                            # ACF computed within each chain
fit.plot("predictor", save="figures/predictor.png")
fit.plot("process_sd", save={"path": "figures/prior.png", "dpi": 300})
```

Factor fits add the plots needed for decomposition recovery and identification
checks:

```python
fit.plot("factor_decomposition", baseline=slice(0, 30 * 12))
fit.plot("parameter_density", parameters=["loading.common.TXx"])
fit.plot("traces")                         # every innovation SD, by chain
fit.plot("loading_deviation", channel="TXx")
fit.plot("identification")
fit.plot("idiosyncratic_innovations", channel="TXx")
fit.plot("acf", save="figures/factor_acf.png")
```

`factor_decomposition` includes posterior bands for the complete predictor,
shared contribution, and idiosyncratic path. Simulation truth can be supplied
to the decomposition, density, trace, joint, and innovation plots.
Analytic prior curves are used in process-SD plots whenever a closed form is
available; genuinely integrated hierarchies are labelled as smooth Monte Carlo
curves.

## Uniform progress output

Set `MCMC(progress=True)` for the same log-friendly display in Gaussian, GEV,
and factor fits. Each line reports chain, `it`, warmup/sampling phase, saved
draws, current scientific parameters, elapsed time, and ETA. PGAS also reports
particle ESS, ancestor diversity, and path change. Use
`progress_every=<integer>` for an explicit cadence; otherwise each chain emits
about twenty updates.

Lower-tail channels are sign-reversed internally. Result methods return the
original temperature orientation by default; pass `original_scale=False` when
inspecting internal model coordinates.

## Univariate models remain available

```python
fit = bx.fit(
    y,
    family="gev",
    period=12,
    parameterization="fruehwirth_schnatter",
    engine="pgas",
    priors="regularized_horseshoe",
    asis=True,
    mcmc=bx.MCMC(
        draws=1_000,
        warmup=1_000,
        chains=4,
        seed=42,
        progress=True,
    ),
    particles=bx.Particles(n=256),
)
```

Declarative univariate `Model`, forecasting, return levels, diagnostics,
plotting, collection helpers, and safe `.bucex` archives retain their v2 API.
The small configurable experiments in [`examples/README.md`](examples/README.md)
cover Gaussian, GEV, parameterization, prior, factor, mixed, bulk/tail, and
Uccle workflows.

## Inference summary

| Model | Engine | Exactness | Parameterizations |
| --- | --- | --- | --- |
| Univariate Gaussian | `ffbs` | exact | centered, disturbance, FS |
| Univariate GEV | `pgas` | exact-invariant | centered, disturbance, FS |
| Eligible one-factor Gaussian | `ffbs` | exact | centered, disturbance, FS |
| Eligible mixed/GEV one-factor | `pgas` | exact-invariant | centered, disturbance, FS |
| Other factor graphs | `ffbs`/`pgas` | exact conditional kernel | centered, disturbance |
| GEV or mixed model | `laplace` | approximate | supported parameterizations |

`parameterization="auto"` selects FS for the v2.1 one-factor layout and
disturbance otherwise. `engine="auto"` selects FFBS for all-Gaussian models and
PGAS for mixed factor models.

## Installation and validation

```bash
python -m pip install .
python -m pytest -q
python validation/run_factor_validation_v2_1.py
python -m build
```

Required dependencies are NumPy, SciPy, and pandas. Matplotlib is optional via
`bucex[plot]`.

Further reading:

- [Dynamic factor guide](docs/DYNAMIC_FACTORS.md)
- [Inference matrix](docs/INFERENCE_MATRIX.md)
- [Uccle workflow](docs/UCCLE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Migration notes](docs/MIGRATION.md)
- [Validation](docs/VALIDATION.md)

## License

The software is MIT licensed. That license does not establish the right to
redistribute the Uccle observations; consult `data/README.md` before publishing
the data files.
