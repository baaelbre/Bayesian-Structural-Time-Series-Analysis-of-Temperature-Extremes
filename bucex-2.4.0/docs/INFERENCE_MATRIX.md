# Inference matrix

## Engine selection and posterior contract

| Model | Family | Engine | State update | Contract |
|---|---|---|---|---|
| `Model` | Gaussian | `ffbs` | exact Gaussian FFBS | exact |
| `Model` | GEV | `laplace` | iterated pseudo-Gaussian FFBS | approximate |
| `Model` | GEV | `pgas` | conditional SMC with ancestor sampling | exact-invariant |
| `MultiSeriesModel` | all Gaussian | `ffbs` | channel FFBS in one hierarchical Gibbs sampler | exact |
| `MultiSeriesModel` | mixed/all GEV | `laplace` | hierarchical pseudo-Gaussian channel updates | approximate |
| `MultiSeriesModel` | mixed/all GEV | `pgas` | channel PGAS in one hierarchical Gibbs sampler | exact-invariant |

`engine="auto"` chooses FFBS for Gaussian models, Laplace for a univariate GEV
model, and PGAS for a multiseries model containing any GEV channel. Hierarchical
Laplace must be requested explicitly so an exploratory approximation cannot be
mistaken for the default exact-invariant analysis.

## Laplace-to-PGAS workflow

Use hierarchical Laplace to screen a mixed model cheaply:

```python
screen = bx.fit(data, model, priors=prior, engine="laplace", ...)
```

The returned `InferencePlan` has `targets_exact_posterior=False`. Pass that fit
directly to PGAS:

```python
exact = bx.fit(
    data,
    model,
    priors=prior,
    engine="pgas",
    init=screen,
    particles=bx.Particles(n=512, proposal="guided"),
    ...,
)
```

`screen.warm_start()` exports one complete compatible draw: channel parameters,
latent paths, hierarchy probabilities, and slab scales. It does not import
Laplace posterior uncertainty into PGAS and it does not alter the PGAS target.
PGAS still requires adequate warmup and final diagnostics.

`HierarchicalSampler(initializer="laplace")` is a lighter alternative that
constructs one Laplace start per GEV channel when no external screen is used.

## Parameterizations

| Parameterization | Univariate | Hierarchical | Description |
|---|---:|---:|---|
| `centered` | yes | no | scientific states with direct process innovations |
| `disturbance` | yes | no | standardized process disturbances |
| `fruehwirth_schnatter` / `fs` | eligible structural models | required | signed scales and unit-innovation paths |

ASIS may interweave valid univariate parameterizations. It is disabled for
hierarchical structural selection. Random sign switches in FS are paired with
the latent path and checked for predictor invariance.

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

Use `HierarchicalPrior` rather than a string in publication scripts. The
hierarchical slab is intentionally normal; the methodological extension is
transparent partial pooling rather than another local shrinkage hierarchy.

## Structural model space

The default `model_space="joint_trend"` combines the level and slope innovation
indicators into four classes:

1. deterministic linear trend;
2. RW1 level with drift;
3. RW2 smooth changing trend;
4. full local linear trend.

The initial slope is estimated in every class. Monthly seasonality is always
present and is fixed or dynamic. Posterior process-SD draws are exactly zero
when the corresponding innovation is inactive. Constant draws correctly have
undefined R-hat and ESS and are labelled as constants.

The legacy `model_space="componentwise"` retains zero/fixed/dynamic trend
states for explicit sensitivity analyses.

## Performance controls

- GEV observation weights and complete-path likelihoods are vectorized across
  particles/time where the recursion permits.
- `HierarchicalSampler(channel_workers=n)` can update conditionally independent
  channels concurrently within a Gibbs sweep.
- `initializer="laplace"` avoids obviously poor GEV starting paths.
- Independent publication chains can be distributed across HPC jobs with
  `examples/15_hpc_independent_chain.py` and recombined with Example 16.

Increasing workers changes wall time, not the declared posterior target. It
does not remove the need for four independent chains.

## PGAS checks

For every retained PGAS fit inspect:

- median and lower quantiles of minimum particle ESS;
- mean unique ancestors;
- path-change rate and mean path-update fraction;
- restored iterations and failure counts;
- GEV support margins;
- stability after increasing particle count.

`path_update_fraction` is the fraction of time positions changed in a PGAS path
update. The old `changed_fraction` name is a deprecated compatibility alias.
High particle ESS alone does not establish MCMC convergence; parameter traces,
R-hat, ESS, allocation switching, and between-chain agreement remain essential.

## Progress output

All MCMC engines use a common line format:

```text
[hierarchical SSVS | PGAS | chain 1/4] | [####----------------] 20.0% |
it 400/2000 | warmup 400/1000 | saved 0/1000 | ... | ETA 12m
```

Mixed hierarchical progress groups GEV shape parameters as, for example,
`xi=(TXx:-0.08,TXn:-0.11,TNx:-0.04,TNn:-0.10)`. PGAS adds particle ESS,
ancestor diversity, and `path_update=`. Scientific names are stable across
engines.
