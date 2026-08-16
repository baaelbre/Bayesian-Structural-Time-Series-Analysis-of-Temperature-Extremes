# BUCEX 2.3.0 release notes

Version 2.3.0 adds a third first-class model design: joint hierarchical
structural SSVS for related series. It keeps the `Model` and `FactorModel`
contracts intact and uses the same compiler, fitter, result, prediction,
plotting, and persistence surfaces.

## New multiseries grammar

`MultiSeriesModel` (alias `PanelModel`) stores named `Channel` objects but no
factors. The compiler therefore produces namespaced channel states without a
shared latent block. `fit(..., priors="hierarchical_ssvs")` couples the channels
through learned model probabilities and learned dynamic-slab multipliers.

The factor and hierarchical alternatives now have deliberately different
semantics:

- `MultiSeriesModel`: shared structural prior, no shared time path;
- `FactorModel`: shared time path/loadings, optional idiosyncratic dynamics.

Inference-plan warnings make this distinction visible before fitting.

## Joint hierarchical SSVS sampler

- exact Gaussian FFBS and model enumeration for all-Gaussian panels;
- exact-invariant PGAS and exact-likelihood-corrected SSVS moves for mixed/GEV
  panels;
- conjugate population Dirichlet updates;
- exact log-scale slice updates for half-t slab multipliers;
- channel-specific Gaussian/GEV nuisance priors;
- one joint chain state and one safe archive, rather than post-hoc pooling of
  independent fits.

`HierarchicalSSVSPrior` exposes Dirichlet concentrations, component reference
scales, half-t degrees of freedom/scales, learned or fixed slab multipliers,
and transparent initial values. `HierarchicalSSVSPriors` additionally permits
channel-specific nuisance-prior overrides.

## Audited FS sign switching

Random sign-symmetry moves are now audited in univariate, hierarchical, and
eligible factor FS samplers. Every move flips the signed coefficient and its
standardized latent path together. The complete predictor/centered path is
checked before and after the move; counts and numerical invariance errors are
stored. Scientific process scales remain `abs(signed_sd)`.

## Result and plotting additions

- `is_multiseries_model` distinguishes all multichannel results;
- `is_factor_model` is true only when a latent factor exists;
- `component_probabilities(channel=...)` and switching/model summaries work
  across panel channels;
- `hierarchical_probabilities()` and `hierarchical_slab_summary()` expose the
  population posterior;
- `channel_rate_summary()` provides one common complete-predictor rate summary
  for panels and factors;
- `fit.plot("hierarchy")`, panel channel plots, process-SD prior/posterior
  plots, traces, ACFs, and direct `save=` output use the existing dispatcher;
- hierarchical process-SD plots represent exact zero mass and use an analytic
  slab density when possible, otherwise a smooth labelled Monte Carlo curve.

Forecasts, log predictive scores, CRPS/tail-weighted scores, PIT diagnostics,
leave-future-out validation, and safe `.bucex` archives extend to panel fits
through the existing APIs.

## Uccle and examples

`make_uccle_multiseries_model()` and `fit_uccle_multiseries()` provide the
six-summary hierarchy. New standalone scripts are:

- `14_hierarchical_ssvs.py`: readable Gaussian simulation and full workflow;
- `15_uccle_hierarchical_ssvs.py`: six-summary mixed PGAS hierarchy;
- `16_hierarchy_or_factor.py`: both scientific designs on the same data.

## Numerical and compatibility contract

The Joseph-form FS backward covariance introduced in 2.2 remains the only
backward-covariance route. It is tested on long, singular, nearly deterministic
series. Existing `Model`, `FactorModel`, prior, plot, prediction, and archive
calls remain source compatible. Safe archives from schemas 1.2, 2.0, and 2.1
remain readable; new panel archives use schema 2.3.

## Publication status

The release is software-complete and regression-tested. That is not equivalent
to a converged scientific analysis. Publication fits still require long
multiple chains, particle sensitivity for mixed models, allocation switching,
simulation recovery, prior sensitivity, held-out predictive scores/PITs, and
factor-identification sensitivity when decomposition claims are made.
