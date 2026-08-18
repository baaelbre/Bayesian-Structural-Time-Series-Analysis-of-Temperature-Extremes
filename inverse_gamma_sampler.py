import numpy as np


def sample_innovation_variance(
    alpha,
    a0=2.0,
    b0=0.01,
    rng=None
):
    rng = np.random.default_rng(rng)

    innovations = ______________________

    T = ________________________________

    a_post = ___________________________

    b_post = ___________________________

    precision = rng.gamma(
        shape=__________________________,
        scale=__________________________
    )

    q = ________________________________

    return q