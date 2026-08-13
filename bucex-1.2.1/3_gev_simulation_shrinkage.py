import bucex as bx
import numpy as np
model_gev = bx.Model(
    bx.GEV(),
    [
        bx.LocalLinearTrend(),
        bx.DummySeasonal(period=12),
    ],
)

truth_small = {
    "sd.level": 0.08,
    "sd.slope": 0.0001,
    "sd.seasonal": 0.001,
    "sigma": 0.80,
    "xi": -0.15,
}

sim_small = bx.simulate(
    model_gev,
    n_time=360,
    params=truth_small,
    initial_state=np.r_[20.0, 0.01, np.zeros(11)],
    seed=300,
)

# now fit this under
prior_names = [
    "manuscript_lasso",
    "regularized_lasso",
    "regularized_horseshoe",
    "pc",
    "normal",
    "ssvs",
]