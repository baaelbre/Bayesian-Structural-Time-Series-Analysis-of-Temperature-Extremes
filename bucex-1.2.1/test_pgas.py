import bucex as bx
"""
This script tests the PGAS engine for fitting UCCLE series, after the Laplace engine has been tested.
"""

pgas_smoke = bx.fit_uccle_series(
    "TXx",
    engine="pgas",
    parameterization="fruehwirth_schnatter",
    priors="regularized_horseshoe",
    asis=True,
    particles=bx.Particles(
        n=128,
        proposal="guided",
    ),
    mcmc=bx.MCMC(
        draws=20,
        warmup=20,
        chains=1,
        seed=40,
        progress=True,
    ),
)


pgas_pilot = bx.fit_uccle_series(
    "TXx",
    engine="pgas",
    parameterization="fruehwirth_schnatter",
    priors="regularized_horseshoe",
    asis=True,
    particles=bx.Particles(
        n=512,
        proposal="guided",
        ess_threshold=0.5,
    ),
    mcmc=bx.MCMC(
        draws=500,
        warmup=500,
        chains=4,
        seed=40,
        progress=True,
    ),
)

diagnostics = pgas_pilot.diagnostics()

print(diagnostics["parameters"])
print(diagnostics["engine"])
print(diagnostics["warnings"])

####################
# PGAS diagnostics
####################
# The PGAS diagnostics include:
#median_min_particle_ess;
#mean_unique_ancestors;
#path_change_rate;
#mean_changed_fraction.