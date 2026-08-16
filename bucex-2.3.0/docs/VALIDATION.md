# Release validation

The v2.3.0 release check has four layers:

1. retained univariate compatibility and general factor source tests;
2. retained v2.2 tests for graph eligibility, FS algebra, fixed initial coefficients,
   collapsed/interwoven loading blocks, horseshoe and triple-gamma scope,
   mixed PGAS, analytic plotting, ACF/save dispatch, SSVS constant diagnostics,
   exact PGAS--SSVS, Joseph covariance handling, predictive scores/LFO, result
   helpers, and schema 2.1 archives;
3. `validation/run_release_validation_v2_3.py` for fixed-seed hierarchical,
   exactness, sign-symmetry, archive, prediction, and Uccle graph checks;
4. wheel/sdist/source-bundle builds plus an isolated installed-wheel smoke test.

Run locally with:

```bash
python -m pytest -q
python validation/run_release_validation_v2_3.py
python -m build
```

## v2.3 hierarchical validator

The validator requires:

- `MultiSeriesModel` to compile named channel blocks with no factor state;
- exact Gaussian FFBS and exact-invariant mixed PGAS plans;
- one joint hierarchy with series and population posterior summaries;
- finite states/log likelihoods and numerical-zero FS sign-invariance errors;
- exact GEV structural model selection under PGAS;
- safe schema-2.3 archive round-trip and finite multichannel forecasting;
- the six-series Uccle helper to preserve two Gaussian/four GEV families and
  the two lower-tail sign transforms;
- planning to distinguish a shared prior hierarchy from a shared latent path.

The complete 2.2 test matrix remains part of the source suite, including the
1,000-observation Joseph-form covariance stress test, exact univariate GEV
PGAS--SSVS, leave-future-out prediction, scores/PITs, triple gamma, factor
identification, and old archive schemas.

## v2.2 factor and numerical validator

The validator writes `validation/factor_validation_2.2.0.json` and requires:

- the Uccle helper to compile one factor, 74 centered states, and 14 structural
  innovations for 12-month dummy seasonality;
- automatic selection of FS plus PGAS for the mixed graph;
- the FS predictor and reconstructed centered predictor to agree below
  `1e-10`, with an FS→centered→FS path round-trip below `1e-10`;
- the unit-innovation transition rank to remain 14 when an idiosyncratic signed
  coefficient is set to `1e-14`;
- the six-channel particle log weight to equal the explicit sum of two
  Gaussian and four GEV log densities;
- finite FS/PGAS and disturbance/FFBS fits with hierarchical shrinkage
  restricted to channel local-level innovations;
- the factor initial slope to remain exactly fixed, Gaussian estimated
  loadings to route through collapsed FFBS, and GEV loadings to route through
  predictor-preserving interweaving;
- factor result helpers and schema 2.1 archive round-trips to preserve semantic
  and NCP states exactly;
- baseline normalization and a seasonal channel decomposition to reconstruct
  the complete predictor below `1e-10`;
- a retained univariate FS fit to return the same `FitResult` contract.
- the corrected univariate seasonal FS design and reconstructed predictor to
  agree below `1e-12`;
- a 1,000-observation singular/nearly deterministic Joseph-form FFBS run to
  remain finite;
- a GEV PGAS--SSVS smoke fit to report an exact model-selection contract and
  retain model-move diagnostics;
- analytic log predictive scores and held-out PIT values to remain finite and
  inside their mathematical ranges.

## Recorded fixed-seed result

The 2026-08-16 v2.3.0 source pass executed all 86 test cases successfully.
The dedicated validator completed in 0.66 seconds under Python 3.12.13. The
Gaussian hierarchy retained finite states with a maximum FS sign-invariance
error of exactly `0.0`; the mixed hierarchy reported exact-invariant PGAS and
exact structural model selection; its smoke path-change rate was `1.0` and no
iteration required restoration. The schema-2.3 archive round-trip was exact,
the multichannel forecast had shape `(3, 3, 2)`, and the Uccle graph retained
all six channels, the expected two-Gaussian/four-GEV family split, and
transform signs `(1, 1, 1, -1, 1, -1)`. The complete machine-readable record
is `validation/release_validation_2.3.0.json`.

The 2026-08-15 v2.2.0 source pass executed 80 test cases successfully. The
v2.2 validator passed every section in 2.13 seconds. Its univariate seasonal
regression/reconstruction error was `4.44e-16`; the 1,001-state Joseph-form
path was finite; the PGAS--SSVS fit reported
`exact_gev_rjmh_with_laplace_independence_proposals`; factor algebra retained
its previous `3.55e-15` predictor and `5.73e-14` path round-trip errors. The
complete record is `validation/factor_validation_2.2.0.json`.

The 2026-08-15 v2.1.5 release pass executed 74 source-test cases successfully
and passed every section of `run_factor_validation_v2_1.py`. The retained
algebraic errors remained `3.55e-15` for the FS/centered predictor and
`5.73e-14` for the centered/NCP path round-trip. All explicit mixed-channel
weights and archive/decomposition checks passed. The intentionally small
64-particle six-channel chain changed its path in two of three retained draws,
while its minimum particle ESS was still one; it is a software stress test,
not a recommended scientific configuration.

The historical 2026-08-14 v2.1.2 run passed every section and all 54 source
tests. The FS
and centered predictors agreed
to `3.55e-15`; the NCP path round-trip error was `5.73e-14`. All nine explicit
six-channel particle weights matched exactly, with the expected family count
of two Gaussian and four GEV terms. The baseline/seasonal channel decomposition
reconstructed the predictor to `3.55e-15`. Semantic and NCP archive round-trip
errors were zero.

The 24-month, 64-particle Uccle smoke chain is required to change its
conditioned path and remain finite. Its minimum ESS can still reach one. That is
useful as a release stress signal, not acceptable evidence for a scientific
fit: the full 1,572-month record needs substantially more particles, several
long chains, and inspection of ESS and ancestor diversity over time.

## Requirements for manuscript inference

The release smoke tests do not establish posterior accuracy for the paper.
Before reporting results:

- run multiple long chains from overdispersed loading, intercept, and scale
  initializations;
- increase particles until ESS, ancestor diversity, and changed-path fractions
  stabilize for the complete record;
- compare FS and disturbance parameterizations under aligned substantive
  priors;
- compare PGAS with Laplace only as a sensitivity/screening exercise and label
  Laplace approximate;
- perform simulation recovery for the factor path, all loadings, idiosyncratic
  scales, GEV shapes, reconstructed channel rates, and return levels;
- include prior/posterior overlays for every idiosyncratic innovation scale;
- run leave-future-out predictive checks and tail-weighted scores;
- distinguish the shared-factor loading contrast from the total channel-rate
  contrast.

Historical v2.0 and v1.2 validation records remain in `validation/` for
regression provenance. The v2.1 loader continues to read their safe archives.
