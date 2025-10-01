"""
using the laplace importance distribution: i.e. we calculate the mode and hessian
of the log-likelihood and use a normal approximation to the posterior as importance
sampling distribution.
This is a bit more involved than the bootstrap filter, but should give better results.
TODO!
"""