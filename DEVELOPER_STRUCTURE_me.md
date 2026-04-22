The core architecture are 'components', 'models' and 'obs'. 

# COMPONENTS
This containts the building blocks of the state space model: trend, seasonal, regression, etc. Each component is responsible for defining how it contributes to the system matrices (T, R, Q) and the design matrix (Z, d). The 'compose' module then stacks these contributions into global matrices for the full model. 

## base.py
Defines the abstract Component() class

# MODELS
This composes the components into a full state space model. 

# OBSERVATIONS

Define likelihood models for the different types of observations we consider (Gaussian, GEV). For this we define the logpdf, a sample method
and gradients.

# INFERENCE
This contains the inference algorithms: Kalman filter/smoother, Laplace approximation, particle filtering
## STATE
Conditional latent state inference.
### KALMAN
We have the Kalman script that implements the Kalman filter and smoother for linear Gaussian state space models. This should also contain the ffbs (forward filtering backward sampling) algorithm for sampling from the posterior of the states in linear Gaussian models.

### LAPLACE
FOr non-Gaussian models, where we can locally Gaussianize the observation equation. 

### PARTICLE
For exact non Gaussian state inference later. Should contain particle_filter, particle_smoother, particle_gibbs and pgas_state_update.

## FIT
Parameter + state inference.
### GIBBS
This should implement the full sampler. In one iteration we have a state update and a parameter update (two-block Gibbs)

## FIT

Aim for something like this
### conditional on fixed parameters
fr = filter_states(y, model, params_state, params_obs, exog=None, method="auto")
sr = smooth_states(y, model, params_state, params_obs, exog=None, method="auto")
xs = sample_states(y, model, params_state, params_obs, exog=None, method="auto", rng=rng)

### full Bayesian inference
fit = fit_posterior(
    y=y,
    model=model,
    priors=priors,
    init=init,
    exog=exog,
    sampler="gibbs",
    state_method="auto",
    n_iter=10000,
    burn=2000,
    rng=rng,
)

where fit returns a PosteriorBundle from core/results.py.
### NUMBA
Fast computation is achieved using Numba, a high performance just-in-time (JIT) compiler for Python.