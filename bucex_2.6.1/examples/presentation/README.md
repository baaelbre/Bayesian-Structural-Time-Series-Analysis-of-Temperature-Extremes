# Seven standalone presentation scripts

These examples teach the public `import bucex as bx` API directly. There is no
shared settings module, configuration object, or orchestration class. Each file
contains its own editable constants and can be run from a clean output
directory by itself.

1. `00_uccle_record.py` loads the complete 1892--present TXx, TXn, TNx, and TNn
   records.
2. `01_tail_simulations.py` simulates three GEV shapes (`xi=-0.30, 0, +0.30`)
   and three observation scales (`sigma=0.75, 1.50, 3.00`).
3. `02_structural_simulations.py` simulates six transparent UC/SSVS designs:
   stationary, linear trend, random walk, local linear trend, stationary level
   with changing seasonality, and local linear trend with fixed seasonality.
4. `03_simulation_laplace.py` simulates as needed, calls `bx.fit` with Laplace,
   and exports all scenario results.
5. `04_simulation_pgas.py` simulates as needed, creates or loads its Laplace
   fit, and passes that `FitResult` to PGAS through `init=`.
6. `05_uccle_laplace.py` loads and fits all four observed extremes with
   Laplace.
7. `06_uccle_pgas.py` loads the observations, creates or loads each Laplace
initializer, fits PGAS, and builds the final Laplace/PGAS comparison.

Every simulated time series is written to a separate figure. A structural
scenario also gets one three-panel truth decomposition (level, slope, and
seasonality), which makes fixed and stochastic components directly comparable.

The simulations default to period 4. The structural series contain 800 blocks:
the longer record makes component evidence clearer while retaining modest
process noises and seasonal amplitudes. These are examples rather than hidden
settings. At the top of scripts 01--04 you can directly edit `N_TIME`, `PERIOD`,
`SIGMA`, `XI`, process standard deviations, slopes, seasonal amplitudes, and
seeds. The scripts construct their designs with the public
`bx.make_*_scenarios()` and `bx.simulate_scenario()` API.

Scripts 03--06 also expose the observation, initial-state, and SSVS
hyperparameters: the inverse-gamma prior on `sigma^2`, the uniform bounds for
`xi`, initial level/slope/seasonal scales, innovation slab scales, and all
component model probabilities. If an existing simulation or fit does not match
the edited controls, the script stops instead of silently reusing it; set
`OVERWRITE=True` when you intend to regenerate that artifact.

Each script automatically prefers the adjacent `bucex/` source checkout. This
matters when a script is executed by path on an HPC system: without the small
bootstrap at the top of every file, Python can select a different, previously
installed package that happens to have the same version number. For development,
an editable installation also keeps arbitrary working directories synchronized:

```bash
python -m pip install --no-deps -e .
```

To inspect an installed copy independently of the checkout, change to a neutral
directory and print both its path and the required plotting symbol:

```bash
(cd /tmp && python -c \
  "import bucex; print(bucex.__file__); print(hasattr(bucex, 'plot_uccle_record_figures'))")
```

Run one file, or run the sequence:

```bash
python examples/presentation/00_uccle_record.py
python examples/presentation/01_tail_simulations.py
python examples/presentation/02_structural_simulations.py
python examples/presentation/03_simulation_laplace.py
python examples/presentation/04_simulation_pgas.py
python examples/presentation/05_uccle_laplace.py
python examples/presentation/06_uccle_pgas.py
```

The fitting examples default to readable pilot settings (`250` retained draws,
`250` warmup iterations, and `2` chains). The publication settings used in the
final analysis are `2_000` draws, `2_000` warmup iterations, `4` chains, and
`512` guided particles for PGAS. Edit the constants at the top of the relevant
file. Set `OVERWRITE=True` only when you intend to regenerate existing primary
simulation or fit artifacts.
