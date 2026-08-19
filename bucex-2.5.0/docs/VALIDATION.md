# Release and scientific validation

Version 2.5.0 has five layers:

1. unit/integration tests for models, priors, inference, results, and archives;
2. fixed-seed numerical regression in `validation/run_release_validation.py`;
3. end-to-end smoke workflow in `validation/run_presentation_smoke.py`;
4. source compilation plus wheel/source builds;
5. installed-wheel import and CLI smoke tests.

```bash
python -m pytest
python validation/run_release_validation.py
python validation/run_presentation_smoke.py
python -m build
```

Short release chains test software contracts only. They are never evidence for
a scientific conclusion.

## Release gates

- `HierarchicalPrior()` resolves to componentwise selection pooling;
- the optional 2.4.1 joint trend space still runs;
- Gaussian FFBS and GEV/hierarchical PGAS plans are exact(-invariant);
- Laplace plans and exported summaries remain explicitly approximate;
- Laplace-to-PGAS warm starts validate model, observations, and path length;
- long nearly deterministic Gaussian fits remain finite;
- singular-support PGAS keeps the conditioned path and finite ancestor weights;
- vectorized GEV calculations agree with scalar references;
- sign switching preserves complete predictors;
- all workflow stages use deterministic paths and refuse accidental overwrite;
- independent archives combine only when model/prior/data/plan match;
- schema-2.5.0 archives round-trip and 2.4.1 archives remain readable;
- result tables include scientific parameters, component probabilities,
  switching, rates, hierarchy summaries, and exactness metadata;
- report mode reads saved fits without running inference;
- PBS arrays map task IDs to unique model/series/chain targets.

## Manuscript gates

- four independent chains for each reported posterior;
- R-hat/ESS and allocation-switching checks;
- particle-count sensitivity for final PGAS analyses;
- zero unexplained restorations and valid GEV support;
- prior predictive checks and defensible slab calibration;
- simulation recovery of structural states and risk functionals;
- held-out LFO scores/PIT for model comparison;
- independent versus pooled results;
- Dirichlet, slab, pooling-mode, and record-start sensitivity;
- archived config, manifest, package version, data check, seeds, fits, tables,
  and figures.
