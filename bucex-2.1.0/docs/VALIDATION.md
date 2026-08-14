# Release validation

The v2.1 release check has four layers:

1. retained univariate compatibility and general factor source tests;
2. v2.1 tests for graph eligibility, FS algebra, FS/disturbance fits,
   horseshoe scope, mixed PGAS, result helpers, and schema 2.1 archives;
3. `validation/run_factor_validation_v2_1.py` for fixed-seed numerical checks;
4. wheel/sdist/source-bundle builds plus an isolated installed-wheel smoke test.

Run locally with:

```bash
python -m pytest -q
python validation/run_factor_validation_v2_1.py
python validation/run_release_validation.py
python -m build
```

## v2.1 factor validator

The validator writes `validation/factor_validation_2.1.1.json` and requires:

- the Uccle helper to compile one factor, 74 centered states, and 14 structural
  innovations for 12-month dummy seasonality;
- automatic selection of FS plus PGAS for the mixed graph;
- the FS predictor and reconstructed centered predictor to agree below
  `1e-10`, with an FS→centered→FS path round-trip below `1e-10`;
- the unit-innovation transition rank to remain 14 when an idiosyncratic signed
  coefficient is set to `1e-14`;
- the six-channel particle log weight to equal the explicit sum of two
  Gaussian and four GEV log densities;
- finite FS/PGAS and disturbance/FFBS fits with the regularized horseshoe
  restricted to channel local-level innovations;
- factor result helpers and schema 2.1 archive round-trips to preserve semantic
  and NCP states exactly;
- a retained univariate FS fit to return the same `FitResult` contract.

## Recorded fixed-seed result

The 2026-08-13 run passed every section. The FS and centered predictors agreed
to `3.55e-15`; the NCP path round-trip error was `5.73e-14`. All nine explicit
six-channel particle weights matched exactly, with the expected family count
of two Gaussian and four GEV terms. Semantic and NCP archive round-trip errors
were zero.

The 24-month, 24-particle Uccle smoke chain changed its conditioned path on all
three retained draws and remained finite. Its minimum ESS reached one. That is
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
