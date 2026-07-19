# bucex v0.3.1 architecture

## Structural supermodel

The non-centred state system is unchanged. Model selection acts only on the
Frühwirth-Schnatter regression coefficients:

```text
alpha0, beta0, gamma0_season, s_level, s_trend, s_season
```

The latent non-centred state dimension therefore stays fixed across all
candidate models.

## Structural states

```text
level  : fixed, dynamic
trend  : zero, fixed, dynamic
season : zero, fixed, dynamic
```

The default model space contains `2 * 3 * 3 = 18` candidates. The level is
never absent. Seasonal baseline coefficients are selected jointly.

## Gaussian update

Conditional on the non-centred path and sigma squared, each candidate is a
Gaussian regression. Active coefficients have proper Gaussian slab priors. The
coefficients are integrated out to calculate the candidate marginal density,
a model is sampled from the normalized probabilities, and its active
coefficients are then sampled from their Gaussian posterior.

## DGEV update

The candidate-model calculation uses the current Laplace pseudo-observations
and time-varying pseudo-variances. This step is approximate. The exact GEV
likelihood is retained for the sigma and xi Metropolis-Hastings updates and for
support validation.

## Main implementation files

```text
bucex/inference/fit/priors.py
bucex/inference/fit/model_space.py
bucex/inference/fit/noncentered_gaussian.py
bucex/inference/fit/noncentered_gev.py
bucex/core/results.py
bucex/plotting/core.py
```
