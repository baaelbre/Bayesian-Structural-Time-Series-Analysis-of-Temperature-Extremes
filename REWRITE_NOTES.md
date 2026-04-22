# bucex v0.1 rewrite notes

This rewrite aims to make the package coherent and usable as a research package for
structural Gaussian and GEV time-series models.

## Main changes

- unified the public observation namespace around `bucex.observation`
- kept `bucex.obs` as a compatibility alias
- added a working `RegressionComponent`
- passed `exog_t` through component composition and `StructuralSSM.design(...)`
- fixed the non-centred seasonal indexing inconsistency
- fixed the non-centred trend-off case so it no longer forces a deterministic `beta0 * t`
- added explicit validation of the v0.1 non-centred scope:
  `LocalLinearTrend` + optional dynamic `DummySeasonal`
- kept centered fitters generic within the current linear-Gaussian state-space design
- removed `__pycache__` from the packaged artifact

## Smoke tests run

- centered Gaussian fit with a static regression component and exogenous covariates
- non-centred Gaussian fit with dynamic trend + dynamic dummy seasonality
- non-centred Gaussian fit with trend turned off
- non-centred GEV fit with Laplace state updates
