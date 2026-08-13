# Inference contract for bucex 1.1

This document separates the two non-centred parameterisations and the exact
and approximate engines. These distinctions are part of the public contract.

## FS augmented non-centring

`fit_fs` and `fit_bayes` retain the research parameterisation. For a local
linear trend,

\[
\begin{aligned}
\alpha_t &= \alpha_0 + t\beta_0
  + s_\alpha\widetilde\alpha_t + s_\beta A_t, \\
\beta_t &= \beta_0 + s_\beta\widetilde\beta_t, \\
\widetilde\alpha_t &= \widetilde\alpha_{t-1}+\epsilon_{\alpha,t}, \\
\widetilde\beta_t &= \widetilde\beta_{t-1}+\epsilon_{\beta,t}, \\
A_t &= A_{t-1}+\widetilde\beta_{t-1},
\end{aligned}
\]

with unit-variance stochastic disturbances. Dynamic dummy seasonality is
transformed analogously. The signed coefficients
`s_level`, `s_trend`, and `s_season` are joint regression coefficients with
the level, slope, and initial seasonal baselines; `q_k = s_k**2`.

This is the Fruehwirth--Schnatter augmented construction, not merely a
disturbance divided by a process standard deviation. Random sign switching
preserves the likelihood and prevents one arbitrary sign representation from
becoming sticky.

When `asis=True`, bucex maps the current FS path to its semantic centred path,
updates signed-scale magnitudes under the exact centred transition density,
and maps back. This is a genuine interweaving update. Exact structural SSVS is
not combined with ASIS.

## General disturbance non-centring

`fit` uses the compiler-generated representation

\[
x_t = F x_{t-1} + R\,\mathrm{diag}(s) z_t,\qquad z_t\sim N(0,I).
\]

It supports static and dynamic regression and arbitrary accepted component
combinations. It is useful and exact where claimed, but it is not labelled as
the FS augmented regression. The fit plan records `parameterization` so the two
forms cannot be confused in saved output.

## Innovation priors in the FS engine

All innovation priors act on signed FS scales.

### Manuscript Bayesian lasso

For active component `k`,

\[
s_k\mid\tau_k,\lambda^2 \sim N(0,v c_k^2\tau_k),\quad
\tau_k\mid\lambda^2\sim\mathrm{Exp}(\lambda^2/2),\quad
\lambda^2\sim\mathrm{Gamma}(a,b).
\]

The manuscript profile shares `lambda2`; the regularized-lasso profile uses
component-specific `lambda2_k` and physical scales `c_k`. In the Gaussian
manuscript profile, `v=sigma**2` and the scale prior contribution is included
in the conjugate observation-variance update. In GEV fits, `v=1`.

### Regularized horseshoe

Writing `u_k=s_k/c_k`,

\[
u_k\mid\lambda_k,\tau,c \sim
N\left(0,\tau^2\frac{c^2\lambda_k^2}
{c^2+\tau^2\lambda_k^2}\right),
\]

with local half-Cauchy `lambda_k`, global half-Cauchy `tau`, and
`c**2 ~ InvGamma(nu/2, nu*s**2/2)`. Conditional Gaussian regression updates
remain exact; the local/global/slab hierarchy uses log-scale MH updates with
the correct Jacobians.

### PC innovation prior

The PC profile declares

\[
P(|s_k|>u_k)=\alpha_k,
\]

so the signed density is Laplace with physical rate
`-log(alpha_k)/u_k`. Sampling uses its exact normal--exponential mixture. This
profile is distinct from both the random-global lasso and v1's general PC-SD
class.

Normal and structural SSVS profiles remain available. FS structural SSVS
enumerates exact zero/fixed/dynamic states; v1's general spike-and-slab is a
continuous near-zero/wide mixture and does not claim dimension changes.

## Gaussian engine

The Gaussian FS state block uses Kalman FFBS with the zero FS initial state.
Singular transition directions are retained on their affine support. No
artificial process jitter is added. Regression coefficients and conditional
shrinkage blocks use Gaussian updates. These are exact conditional kernels.

The general Gaussian engine uses its compiler and FFBS kernels with the same
affine-support convention.

## GEV PGAS engine

Both PGAS engines use conditional sequential Monte Carlo with ancestor
sampling. The default guided proposal is locally Gaussian in unit-disturbance
coordinates; bootstrap proposals are also available. Exact transition over
proposal corrections are included, so guidance affects efficiency, not the
invariant target.

Normalized log weights are retained independently from floating-point
probabilities. Ancestor weights therefore use the finite log probability even
if `exp(log_weight)` underflows to zero. If every log weight is non-finite, the
iteration fails explicitly; it never silently substitutes uniform weights.

In the FS GEV sampler, the static FS regression block is updated with
elliptical slice sampling under its conditional Gaussian prior and the exact
GEV likelihood. The sigma and xi MH steps also use the exact likelihood. With
lasso, horseshoe, PC, or normal signed-scale priors, the complete PGAS sweep is
labelled `targets_exact_posterior=True`.

Exact structural SSVS is not offered with PGAS in 1.1 because the current model
space update uses Laplace pseudo-observations.

## GEV Laplace engine

The fast engine iterates the likelihood quadratic approximation to a mode,
uses a line search that rejects objective/support deterioration, checks the
relative location change, and then performs a pseudo-Gaussian FFBS draw.
Metadata stores convergence, iterations, relative change, support rejections,
and restoration reasons.

This engine is an approximation. A converged mode does not turn the subsequent
MCMC into an exact GEV posterior sampler. Fits record
`targets_exact_posterior=False` and `approximation="iterated_laplace"`.

## Transaction and restoration rule

A GEV FS sweep is committed only after the state, regression, ASIS, shrinkage,
and observation updates all pass numerical and GEV-support checks. A failed
attempt cannot partially mutate the chain. After the configured attempts are
exhausted, the last complete state is restored and the reason is printed when
progress is enabled and saved in metadata.

## Chain and diagnostic contract

Independent chains use independent, recorded seeds. The legacy flat arrays
remain available, and `chain_slices` plus `chain_ids` preserve chain identity.
Rank-normalized split R-hat and bulk ESS are computed from the unflattened
chains. Engine diagnostics include:

- Laplace convergence rate, iterations, relative change, support rejections,
  and restored fraction;
- PGAS minimum particle ESS, unique ancestors, path-change rate, changed
  fraction, and restored fraction;
- MH/ASIS/horseshoe acceptance rates.

Short smoke chains are for software verification only; they do not satisfy the
production convergence gate.
