# bucex 1.1 completion checkpoints

| Checkpoint | Release implementation | Verification |
| --- | --- | --- |
| Preserve 0.3 package surface | canonical `api/components/core/datasets/diagnostics/inference/io/models/observation/plotting/risk/simulate` layout | legacy 0.2--0.3.3 tests |
| Genuine FS NCP | augmented level/slope/integrated-slope/seasonal state with signed scales and sign switching | centred round trips, exact affine support, FS smoke tests |
| Manuscript lasso | original shared hierarchy and Gaussian sigma coupling | legacy prior and sampler tests |
| Regularized lasso | component-specific physical scales and lambda parameters | stored hierarchy tests |
| Regularized horseshoe | local/global half-Cauchy plus finite slab | v1.1 prior and sampler tests |
| PC innovation prior | declared tail-probability calibration and exact normal mixture | calibration and draw-storage tests |
| Structural SSVS | exact zero/fixed/dynamic FS model space | 18-model and exact-zero tests |
| ASIS | centred signed-scale interweaving with inverse FS transform | Gaussian, Laplace, and PGAS smoke tests |
| Exact GEV FS block | PGAS state update plus exact GEV elliptical slice for FS coefficients | exact-target metadata and finite particle diagnostics |
| PGAS underflow fix | normalized log-weight history retained for ancestor sampling in both engines | targeted PGAS and full suite |
| Iterated Laplace | damped/convergence-checked mode, FFBS approximation, transaction restoration | restoration and full TXx smoke tests |
| General v1 grammar | declarative components, compiler, regression, disturbance NCP | v1 compiler/regression suite |
| Self-contained fits | model, data, dates, priors, config, init, chain slices, centred/NCP draws | result-contract and round-trip tests |
| Safe persistence | JSON + compressed arrays + SHA-256, no pickle | round trip and tamper rejection |
| Forecasts and scores | state/parameter/process/observation uncertainty plus tail scores | forecast/score tests |
| Diagnostics | rank-normalized split R-hat/ESS plus engine metrics | shifted-chain and engine tests |
| Uccle workflow | six monthly series, daily reproduction check, old/new configuration styles | bundled-data and full-length TXx tests |
| Clean install | wheel and sdist with bundled monthly data | isolated offline install and import smoke test |

Publication of redistributed Uccle data remains gated on an authoritative
citation and redistribution statement; this is a provenance issue, not a code
test failure.
