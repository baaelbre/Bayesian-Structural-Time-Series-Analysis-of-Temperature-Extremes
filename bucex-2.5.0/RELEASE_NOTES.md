# bucex 2.5.0 release notes

Version 2.5.0 turns the current Uccle analysis into one reproducible scientific
sequence while retaining the tested 2.4.1 inference kernels.

## Componentwise SSVS is the primary model

The default hierarchical state space is now the transparent componentwise
model:

- level: fixed/dynamic;
- slope: zero/fixed/dynamic;
- monthly seasonality: fixed/dynamic.

Population Dirichlet probabilities may be shared across the six series while
each series retains its own indicators and path. `pool="selection"` is the
default. Shared slab multipliers (`pool="slab"` or `"both"`) are explicit
sensitivity analyses. The four-class joint level/slope space remains available
through `model_space="joint_trend"` for backward-compatible sensitivity work.

## Executable paper/presentation sequence

The new `bucex.workflows` API and `bucex-presentation` command implement:

1. data validation;
2. TXx stationary, linear, Huerta-style local-level, Gaetan--Grigoletto-style
   RW2, and local-linear structural analogues;
3. TXx componentwise SSVS;
4. leave-future-out proper scores and PIT across those alternatives;
5. six independent componentwise fits;
6. hierarchical Laplace screening;
7. warm-started exact-invariant hierarchical PGAS;
8. slab/both pooling sensitivity;
9. report-only table and figure generation.

Three runtime profiles (`smoke`, `pilot`, and `publication`) make software
checks visibly distinct from inferential runs. Every artifact is placed under
a deterministic result tree and recorded in a concurrency-safe manifest.

## Reproducible HPC execution

PBS/Torque templates now cover the whole presentation rather than only the
final hierarchy. Benchmark models, TXx SSVS, six independent series, and final
hierarchical PGAS use one independent chain per array task. A final job combines
checksummed archives and regenerates the report. The same command-line stages
can be run under Slurm or another scheduler.

## Result contract and persistence

- Added tidy CSV/JSON export for parameters, chain diagnostics, component and
  joint structural probabilities, switching, hierarchy probabilities, slab
  multipliers, full-period rates, and TXx annual exceedance probabilities.
- Added report-only regeneration from `.bucex` files.
- Advanced the pickle-free, checksummed archive schema to 2.5.0 while retaining
  readers for 2.4.1 and earlier supported schemas.
- Preserved exactness and approximation flags in every fit and exported
  summary.

## Inference scope

The methodological combination remains FS noncentring and structural SSVS,
with Gaussian FFBS and exact-invariant PGAS for non-Gaussian observations.
Laplace is explicitly approximate. Singular/deterministic FS directions are
handled on their affine transition support. The release does not add a common
factor, shared latent trajectory, residual copula, or claim that vectorization
eliminates the sequential time recursion.

## Upgrade note

Code that relied on the 2.4.1 default joint trend space should request it
explicitly:

```python
bx.HierarchicalPrior(
    pool="selection",
    model_space="joint_trend",
    trend_states=("fixed", "dynamic"),
)
```

New code can use `bx.componentwise_hierarchical_prior()` or the now-equivalent
`bx.HierarchicalPrior()` default.
