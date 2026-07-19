# Validation notes for bucex v0.3.2

## Automated checks

The source distribution passes 13 tests covering:

- the v0.2 public API and manuscript profile;
- Gaussian and DGEV SSVS smoke tests;
- the 18-model structural space and exact inactive coefficients;
- calibrated SSVS slope and innovation scales;
- positive-semidefinite covariance repair without runtime warnings;
- component-wise Bayesian-lasso storage and updates;
- finite-period level-rate calculations without a slope state.

## End-to-end smoke runs

The release was tested with:

- a 100-iteration TXm component-wise regularized fit;
- a 100-iteration TXm SSVS fit;
- a short TXx DGEV component-wise regularized fit;
- source and wheel installation in clean temporary environments.

## Scientific scope

The numerical implementation and API are tested, but the default shrinkage and
SSVS scales are not claimed to be universally optimal. Final scientific
analyses should compare at least the manuscript, regularized and SSVS profiles,
use multiple chains or seeds, inspect structural switching, and evaluate
predictive performance. Broad instantaneous-slope uncertainty can remain a real
identifiability feature even after regularisation.
