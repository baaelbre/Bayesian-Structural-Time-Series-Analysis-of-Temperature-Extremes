# bucex 2.4.0 release notes

Version 2.4 focuses the package on a single coherent research design: Bayesian
structural time-series analysis for one outcome or for several related
outcomes whose structural behavior is partially pooled.

The recommended paper model gives every temperature summary a separate latent
level, slope, and annual cycle. A hierarchy can pool exact SSVS allocations,
normal dynamic-slab magnitudes, or both. This directly estimates whether
fixed or dynamic behavior recurs across summaries without forcing the paths to
be identical.

The default monthly hierarchy treats seasonality as present. SSVS distinguishes
a fixed annual pattern from an evolving one; it does not ask the physically
unhelpful question of whether the annual cycle exists.

Initial level and slope are now estimated posterior parameters. Chain starts
come from a seasonally adjusted regression, and the level--slope plot uses
seasonally adjusted observations for a scientifically fair visual comparison.

This release also fixes guided disturbance PGAS for singular transitions. The
reference path is projected onto affine support, disturbance recovery uses the
actual scaled transition loading, and an invalid optional ancestor update keeps
the validated conditioned predecessor. Mixed hierarchical PGAS should be rerun
with this release; older incomplete or failed runs should not be reported.

For Uccle, the advised sequence is:

1. transparent independent normal-prior fits;
2. independent SSVS fits;
3. pooled-selection hierarchy as the main joint analysis;
4. pooled slab and pooled both as sensitivity analyses;
5. start-date, hyperprior, particle-count, and predictive sensitivity checks.

Indecisive fixed/dynamic probabilities can be a valid finite-record result. If
R-hat, ESS, chain overlap, switching, ACF, and PGAS diagnostics are sound, the
uncertainty should be reported rather than forced into a hard classification.
