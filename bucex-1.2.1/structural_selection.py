import bucex as bx
"""
Ask whether state components are zero, fixed or dynamic.
"""
selected = bx.fit_uccle_series(
    "TXx",
    engine="laplace",
    parameterization="fruehwirth_schnatter",
    priors="ssvs",
    asis=True,
    mcmc=bx.MCMC(
        draws=2000,
        warmup=2000,
        chains=4,
        seed=300,
    ),
)

print(selected.component_probabilities())
print(selected.structural_model_probabilities())
print(selected.most_probable_structure())

selected.plot("component_probabilities")