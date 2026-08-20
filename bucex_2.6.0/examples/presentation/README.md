# Presentation scripts

Run the files in numerical order. They produce the complete COMPSTAT result
tree under `results/presentation/`:

1. the four-series Uccle record and the TXx opening figures;
2. local-level GEV tail-class simulations;
3. structural simulations with common `sigma=1.5` and `xi=-0.20`;
4. componentwise SSVS fits using Laplace;
5. the same fits using Laplace-initialized PGAS;
6. Laplace fits of TXx, TXn, TNx, and TNn;
7. corresponding PGAS fits;
8. aggregate selection, trajectory, prior-to-posterior, and diagnostics output.

For a quick check, set `BUCEX_PROFILE=smoke`. The default is `pilot`; final
results should use `BUCEX_PROFILE=publication`. Useful optional environment
variables include `BUCEX_OUTPUT_DIR`, `BUCEX_DATA_DIR`, `BUCEX_OVERWRITE=1`,
`BUCEX_SCENARIO`, `BUCEX_SERIES`, and the numeric runtime overrides documented
in `settings.py`.

Example:

```bash
export BUCEX_PROFILE=smoke
python examples/presentation/00_uccle_record.py
python examples/presentation/01_tail_simulations.py
python examples/presentation/02_structural_simulations.py
python examples/presentation/03_simulation_laplace.py
python examples/presentation/04_simulation_pgas.py
python examples/presentation/05_uccle_laplace.py
python examples/presentation/06_uccle_pgas.py
python examples/presentation/07_build_results.py
```
