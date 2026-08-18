# Inference matrix

## Engine selection

| Model | Family | Engine | State update | Contract |
|---|---|---|---|---|
| `Model` | Gaussian | `ffbs` | exact Gaussian FFBS | exact |
| `Model` | GEV | `laplace` | iterated pseudo-Gaussian FFBS | approximate |
| `Model` | GEV | `pgas` | conditional SMC with ancestor sampling | exact-invariant |
| `MultiSeriesModel` | all Gaussian | `ffbs` | channel FFBS in one hierarchical Gibbs sampler | exact |
| `MultiSeriesModel` | mixed or all GEV | `pgas` | channel PGAS in one hierarchical Gibbs sampler | exact-invariant |

`engine="auto"` chooses FFBS for Gaussian models, Laplace for a univariate GEV
model, and PGAS for a multiseries model containing any GEV channel.

Laplace is valuable for quick univariate screening. It is not available as a
substitute for mixed hierarchical inference because the hierarchy needs exact
structural allocation updates against the GEV likelihood.

## Parameterizations

| Parameterization | Univariate | Hierarchical | Description |
|---|---:|---:|---|
| `centered` | yes | no | scientific states with direct process innovations |
| `disturbance` | yes | no | standardized process disturbances |
| `fruehwirth_schnatter` / `fs` | eligible structural models | required | signed scales and unit-innovation paths |

ASIS may interweave valid univariate parameterizations. It is disabled for
hierarchical structural selection.

## Prior availability

| Public profile | Univariate FS | Hierarchical | Meaning |
|---|---:|---:|---|
| `normal` | yes | normal slab | signed Gaussian innovation coefficient |
| `pc` | yes | no | exponential process-SD penalty |
| `ssvs` | yes | pooled selection | exact structural point masses plus normal slab |
| `regularized_horseshoe` | yes | no | continuous global-local shrinkage with slab |
| `triple_gamma` | yes | no | continuous normal-gamma-gamma shrinkage |
| `regularized_triple_gamma` | yes | no | triple gamma with regularizing slab |
| `pooled_slab` | no | yes | shared half-t multiplier on normal slabs |
| `pooled_both` | no | yes | pooled selection and pooled slab magnitude |

Use `HierarchicalPrior` rather than a string when writing publication scripts.

## Structural states

The canonical indicators are:

- `0 = zero`;
- `1 = fixed`;
- `2 = dynamic`.

Level supports fixed/dynamic. Trend supports zero/fixed/dynamic. Monthly season
supports fixed/dynamic by default. Posterior SD draws are exactly zero whenever
a component is zero or fixed; a constant posterior allocation therefore has
undefined R-hat and ESS and is labelled as such.

## PGAS checks

For every retained PGAS fit inspect:

- median and low quantiles of minimum particle ESS;
- mean unique ancestors;
- path-change rate and changed fraction;
- restored iterations and failure counts;
- GEV support margins;
- stability after increasing the particle count.

High particle ESS alone does not establish MCMC convergence. Parameter traces,
R-hat, ESS, and between-chain agreement remain essential.

## Progress output

All MCMC engines report a common line format:

```text
[hierarchical | PGAS | chain 1/4] | [####----------------] 20.0% |
it 400/2000 | warmup 400/1000 | saved 0/1000 | ... | ETA 12m
```

The parameter portion is engine-specific but uses stable scientific names.
PGAS adds particle ESS, ancestor diversity, and path-change information.
