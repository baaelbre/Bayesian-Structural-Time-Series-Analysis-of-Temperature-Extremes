# Seven standalone presentation scripts

These examples teach the public `import bucex as bx` API directly. There is no
shared settings module, configuration object, or orchestration class. Each file
contains its own editable constants and can be run from a clean output
directory by itself.

1. `00_uccle_record.py` loads the complete 1892--present TXx, TXn, TNx, and TNn
   records.
2. `01_tail_simulations.py` simulates three GEV shapes (`xi=-0.30, 0, +0.30`)
   and three observation scales (`sigma=0.75, 1.50, 3.00`).
3. `02_structural_simulations.py` simulates the seven UC/SSVS designs.
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
