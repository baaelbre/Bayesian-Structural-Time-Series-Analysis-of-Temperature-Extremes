# Uccle temperature-extremes analysis

## Series used in the presentation

| Name | Model orientation | Description |
|---|---|---|
| `TXx` | upper tail | monthly maximum daily maximum temperature |
| `TXn` | lower tail | monthly minimum daily maximum temperature |
| `TNx` | upper tail | monthly maximum daily minimum temperature |
| `TNn` | lower tail | monthly minimum daily minimum temperature |

Lower-tail series are sign-transformed internally, fitted as maxima, and
mapped back by all result methods. Loaders require finite, consecutive monthly
observations. The broader data API still includes the Gaussian monthly means
`TXm` and `TNm`; they are not part of this presentation workflow.

## Story and stages

The workflow is ordered to separate questions that are otherwise easy to mix:

1. The Uccle record and TXx evolution establish why stationarity is doubtful.
2. Tail simulations vary only `xi`, showing that tail class and latent
   nonstationarity are different modelling decisions.
3. Structural simulations fix `sigma=1.5` and `xi=-0.20`, then vary only the
   unobserved components.
4. Laplace fits give a fast, explicitly approximate SSVS analysis.
5. PGAS fits use the Laplace path as an initializer and the exact GEV density.
6. The same model/prior is fitted to TXx, TXn, TNx, and TNn.

```bash
bucex-presentation run data --profile pilot
bucex-presentation run tail-simulations --profile pilot
bucex-presentation run structural-simulations --profile pilot
bucex-presentation run simulation-fit --profile pilot --engine laplace
bucex-presentation run simulation-fit --profile pilot --engine pgas
bucex-presentation run uccle-fit --profile pilot --engine laplace
bucex-presentation run uccle-fit --profile pilot --engine pgas
bucex-presentation report --profile pilot --strict
```

PGAS commands require the matching combined Laplace files. A fit on one record
window cannot initialize a fit on another because observations and path length
must match exactly.

## What the structural probabilities mean

For each series, report posterior probabilities for:

- fixed versus dynamic level;
- absent, fixed, or dynamic slope;
- absent, fixed, or dynamic annual cycle.

These are posterior model probabilities under the stated SSVS prior and GEV
model. They are not frequentist tests and should not be collapsed to a single
selected model unless a decision rule is stated. Trajectory and risk summaries
remain model averaged over the sampled structures.

The v2.6 presentation model keeps `sigma` and `xi` constant in time while
inferring both. Structural selection applies to location, slope, and seasonal
location components only. A time-varying scale or shape analysis is a distinct
model extension, not another label in the current three-component SSVS table.

## What to report for each fit

- component and joint structural probabilities;
- structural-state switching by chain;
- model-averaged latent predictor and interval;
- prior and posterior process-scale distributions, including mass at zero;
- observation scale, shape, and finite endpoint when `xi<0`;
- particle ESS, ancestor diversity, path change, path-update fraction, and
  reference-ancestor change for PGAS;
- R-hat and ESS from independent combined chains;
- sensitivity to particles, slab widths, record start, and prior model odds.

TXx is presented first because its physical interpretation is immediate. The
other three series are confirmatory applications of the same inferential
grammar, not opportunities to change the model after seeing the answer.

## Risk assessment

The dynamic GEV fit is usable for posterior risk calculations because every
draw supplies a time-indexed location, scale, and shape. `FitResult` exposes
return-level, exceedance-probability, return-period, and endpoint draws on the
original upper/lower-tail orientation.

For nonstationary data, attach the time index to every reported risk. A
"20-year return level" at time `t` is conditional on the fitted parameters at
that time; it is not one timeless property of the entire record. Annual
exceedance probabilities are composed from the twelve monthly probabilities,
so the seasonal cycle is retained in the risk functional.

## Artifact layout

The workflow stores input truths, checksummed fits, tables, and figures under
one configured root. `manifest.json` is updated under a file lock, so parallel
PBS tasks cannot erase one another's records. `report` loads combined fits and
does not sample.
