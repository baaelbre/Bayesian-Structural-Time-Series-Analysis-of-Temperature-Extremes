src/sts_extremes/
├── core/
│   ├── typing.py          # Protocols / dataclasses for matrices, params, results
│   ├── time_index.py      # seasonal calendar, blocks, etc.
│   ├── rng.py
│   └── results.py         # PosteriorBundle, IO
│
├── components/
│   ├── base.py            # Component API: dims, system blocks, design blocks
│   ├── trend.py.
│   ├── seasonal.py
│   ├── regression.py
│   └── compose.py         # stack components into global matrices
│
├── obs/
│   ├── base.py            # ObservationModel API: loglik, sample, (grad/hess optional)
│   ├── gaussian.py
│   ├── gev.py
│   └── links.py           # constraints: identity, log, softplus; map eta -> params
│
├── io/
│   ├── base.py          # small helpers: ensure_dir, json encode, etc.
│   ├── save.py          # save SimResult (+meta) to npz/parquet/csv
│   └── load.py          # load back into SimResult

├── models/
│   ├── base.py            # StateSpaceModel API (system + design + obs)
│   └── structural.py      # StructuralSSM: components + observation mapping
├── simulate/
│   └── statespace.py      # simulate x, y from any model
│
inference/
├── base.py
├── state/
|   |-- base.py            
│   ├── kalman.py
│   ├── ffbs.py
│   ├── laplace.py
│   └── particle.py
├── fit/
│   ├── base.py
│   ├── centered_gaussian.py
│   ├── centered_gev.py
│   ├── noncentered_gaussian.py
|   ├── noncentered_gev.py
|   |-- priors.py
|   |-- utils.py
└── dispatch.py

│
├── diagnostics/
├── tasks/
├── plotting/
└── io/