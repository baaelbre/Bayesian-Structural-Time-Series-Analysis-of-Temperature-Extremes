# bucex 2.2.0 examples

The examples form one sequential workflow. Each file is standalone: it
imports only normal Python packages and `bucex`, contains its own settings at
the top, and can be run directly from the project directory.

```bash
python examples/01_gaussian_local_trend.py
```

There is no shared configuration file and no environment-variable machinery.
To change draws, warmup, chains, particles, dates, priors, or the selected
Uccle series, edit the clearly labelled constants near the top of the script.

| Order | Script | Purpose |
|---:|---|---|
| 1 | `01_gaussian_local_trend.py` | Complete univariate Gaussian workflow with exact FFBS |
| 2 | `02_gev_local_trend.py` | Complete univariate GEV workflow with exact PGAS and support checks |
| 3 | `03_diagnose_gev_pgas.py` | Separate short-chain, low-particle, restoration, and parameterization problems |
| 4 | `04_compare_priors.py` | Compare normal, PC, regularized horseshoe, triple gamma, regularized triple gamma, and exact SSVS on the same data |
| 5 | `05_compare_parameterizations.py` | Compare centered, disturbance, and FS sampling under matched scientific priors |
| 6 | `06_factor_gaussian.py` | Shared Gaussian warming factor with triple-gamma idiosyncratic shrinkage |
| 7 | `07_factor_mixed_gaussian_gev.py` | Joint Gaussian/GEV factor model for a mean and an extreme |
| 8 | `08_bulk_tail_independent.py` | Parallel but independent Gaussian bulk and GEV tail analyses |
| 9 | `09_uccle_univariate.py` | Run six separate SSVS analyses (or choose another prior) and compare their results |
| 10 | `10_uccle_factor.py` | Fit the proposed six-summary Uccle factor model with a regularized triple gamma |
| 11 | `11_fixed_and_dynamic_components.py` | Verify exact zero/fixed/dynamic FS semantics and static coefficients |
| 12 | `12_gev_ssvs_pgas.py` | Exact GEV structural SSVS with PGAS and model-move diagnostics |
| 13 | `13_leave_future_out.py` | Expanding-window prediction, log score, CRPS/tail scores, and PIT |

## What to check after every fit

Do not interpret posterior paths or forecasts before checking:

1. Rank-normalized split R-hat, preferably below 1.01.
2. Bulk ESS for every scientific innovation SD, observation parameter, and
   estimated loading.
3. Chain-specific traces rather than only pooled posterior densities.
4. Chain-specific ACFs (`fit.plot("acf")`) for persistent parameters.
5. For PGAS, particle ESS, ancestor diversity, path-change rate, and changed
   fraction.
6. For GEV FS fits, `restored_iterations`, failure counts, and the minimum GEV
   support margin.
7. For factor fits, loading--deviation correlations and whether the complete
   predictor is more stable than its shared/idiosyncratic decomposition.
8. For PGAS--SSVS, model-MH acceptance and actual switching as well as ordinary
   state, particle, and parameter diagnostics.
9. For predictive claims, genuinely held-out log/CRPS/tail scores and PITs;
   in-sample posterior PIT values are not forecast validation.

The run lengths in these files are development-scale starting points. They
are more serious than a 100-draw smoke test, but they are not a promise of
convergence. Increase warmup and draws whenever diagnostics ask for it;
validate particle sensitivity before a final GEV analysis.

## Interpreting the prior comparison

`sd.slope` is the innovation SD of the latent slope per observation interval,
not the slope itself. The built-in monthly structural profiles use reference
scales of approximately 0.03 for level/seasonal innovations and 0.0002 for
slope innovations. Example 4 simulates a slope innovation compatible with
that calibration so it compares prior *shape* rather than silently placing the
truth far into every prior's tail.

- `normal` gives each signed FS scale a normal prior, hence a half-normal prior
  on the scientific innovation SD.
- `pc` gives the innovation SD an exponential prior calibrated through
  `P(SD > upper) = alpha`.
- `regularized_horseshoe` combines strong shrinkage near zero with a heavy tail
  and a regularizing slab. In v2.2 its coupled hierarchy is slice-updated;
  this fixes the avoidable random-walk bottleneck in v2.1.4.
- `triple_gamma` uses the normal-gamma-gamma representation. The stored
  shrinkage factor `rho` is close to one for a strongly suppressed innovation
  and close to zero for an effectively unshrunk innovation.
- `regularized_triple_gamma` adds an optional inverse-gamma slab that caps the
  local variance while preserving the triple-gamma spike/tail parameters.
- `ssvs` assigns exact posterior probabilities to zero, fixed, and dynamic
  structures. Its point mass at zero must be inspected with component
  probabilities, not only a smoothed density.

Gaussian FS SSVS uses exact conjugate model-space updates. GEV SSVS is
exact-invariant with `engine="pgas"`; its Laplace calculation is a proposal
corrected by the exact likelihood. `engine="laplace"` remains a faster,
explicitly approximate screening route. A fixed FS component keeps its static
coefficient and sets the corresponding innovation SD exactly to zero.

The process-prior plot uses analytic densities for PC, folded-normal, SSVS
slabs, ordinary process priors, and fixed-global unregularized triple gamma.
Hierarchies whose hyperparameters are integrated out are shown with a smooth
Monte Carlo KDE, explicitly labelled as such. Every plot can be saved directly:

```python
fit.plot("acf", save="figures/acf.png")
fit.plot("process_sd", save={"path": "figures/prior.png", "dpi": 300})
forecast.plot(save="figures/forecast.png")
```

For a constant SSVS allocation, R-hat and ESS are undefined and are now shown
as `NaN` with the status `constant posterior allocation`. A value of one and an
ESS equal to all draws in older output was a mechanical zero-variance result,
not proof of excellent mixing.

Prior-sensitivity conclusions are meaningful only when every compared fit has
converged. A difference between two unconverged posterior curves is a sampler
difference, not evidence of scientific prior sensitivity.
