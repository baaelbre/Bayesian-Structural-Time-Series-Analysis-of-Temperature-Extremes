# bucex v2.1.1 release

This release implements the manuscript's one-factor model

\[
\mu_{i,t}=c_i+\lambda_i f_t+S_{i,t}+\alpha_{i,t}
\]

for the six Uccle temperature summaries.

Highlights:

- one shared local-linear climate trend with `TXm=1` loading anchor;
- estimated loadings for the other five series;
- independent channel local levels and independent dummy seasonality;
- full Frühwirth--Schnatter unit-innovation factor parameterization;
- retained centered and standardized-disturbance parameterizations;
- joint mixed PGAS weights from two Gaussian and four GEV likelihood terms;
- regularized horseshoe on idiosyncratic local-level innovation scales;
- ASIS, factor/rate/loading result helpers, safe schema 2.1 archives;
- phase-aware MCMC progress with PGAS particle diagnostics and a complete
  truth-versus-posterior simulation sandbox;
- unchanged univariate model workflows and read compatibility with v1.2/v2.0
  archives.
- an all-Gaussian exact-FFBS channel tutorial and a numerical covariance
  tolerance fix for singular FS backward sampling.

Install the wheel with:

```bash
python -m pip install bucex-2.1.1-py3-none-any.whl
```

The factor horseshoe is continuous shrinkage, not a point-mass selector. A
loading above one describes the shared-factor contribution; use reconstructed
channel rates for claims about the complete trajectory.
