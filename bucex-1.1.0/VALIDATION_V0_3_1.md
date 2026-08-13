# Validation of bucex v0.3.1

The release was checked with:

```bash
pytest -q
```

All eight included tests pass. They cover:

- preservation of the v0.2 manuscript-prior API;
- ordinary Normal innovation-prior profiles;
- enumeration of 18 default structural models;
- exclusion of a zero-level model;
- exact zero constraints for inactive SSVS coefficients;
- posterior component and joint-model summaries;
- DGEV SSVS approximation metadata;
- existing level, slope, risk, return-period and endpoint plotting.

The built wheel was installed into a clean target directory and a short Gaussian
SSVS fit was run successfully. The Torque shell worker was also run on TXm with
a short SSVS chain and produced the fitted object, posterior summaries,
component probabilities, structural-model table and figures.

## Important limitation

The Gaussian SSVS model-space update is conjugate and exact conditional on the
non-centred state path and observation variance. The DGEV structural update is
based on Laplace pseudo-observations and is therefore approximate. This is
stored in the fit metadata and documented in the README.

The default slab scales and structural prior probabilities are sensible package
defaults, not a completed sensitivity analysis for the Uccle application.
