# DLM Plotter

This plotter reads a posterior bundle (`posterior.npz`) produced by the Gaussian DLM sampler and writes diagnostic figures:

- `overview.png`
- `trace_hist_acf_*.png`
- `state_level/slope/seasonality.png`
- `quick_report.png`

It supports kwarg overrides from the CLI via:

- `--overview-kw K=V`
- `--traceacf-kw K=V`
- `--states-kw K=V`
- `--quick-kw K=V`

Values are parsed with `ast.literal_eval`, so you can pass tuples/lists/dicts, e.g. `trace_ylim=(-2,2)`.


## CLI usage

### Basic
```bash
python simulator/dlm_plotter.py --target path/to/run_or_posterior.npz
```

### Save figures elsewhere
```bash
python simulator/dlm_plotter.py --target ... --out results/my_figs
```

### Only trace plots
```bash
python simulator/dlm_plotter.py --target ... --skip-overview --skip-states --skip-quick
```


## Kwarg override system

You can override any kwargs of the corresponding plotting method:

- `--overview-kw` → `plotter.figure_overview(..., **kwargs)`
- `--traceacf-kw` → `plotter.figure_trace_acf_core(..., **kwargs)`
- `--states-kw` → `plotter.figure_states_separate(..., **kwargs)`
- `--quick-kw` → `plotter.quick_report(..., **kwargs)`

### Examples

Zoom trace + histogram (without changing data):
```bash
python simulator/dlm_plotter.py --target ... \
  --traceacf-kw trace_ylim=(-2,2) \
  --traceacf-kw hist_xlim=(-2,2)
```

Clip outliers for plotting and compute ACF/ESS on the “clean” (outlier-removed) chain:
```bash
python simulator/dlm_plotter.py --target ... \
  --traceacf-kw plot_policy="clip" \
  --traceacf-kw diag_policy="clean" \
  --traceacf-kw clip_q=(0.001,0.999)
```

Robust MAD-based bounds:
```bash
python simulator/dlm_plotter.py --target ... \
  --traceacf-kw plot_policy="drop" \
  --traceacf-kw clip_nmad=12.0
```

Hard absolute cap:
```bash
python simulator/dlm_plotter.py --target ... \
  --traceacf-kw plot_policy="clip" \
  --traceacf-kw max_abs=1000
```


## Method kwargs (complete list)


## 1) `DLMPlotter(draws, meta, level=0.90)`

Constructor arguments:

- `draws: Dict[str, np.ndarray]` (must contain `mu` with shape `(S,T)`)
- `meta: Dict[str, Any]`
- `level: float = 0.90`  
  Credible band level used for ribbons (`0 < level < 1`).


## 2) `figure_overview(...)`

Signature + defaults:

- `save_dir: Optional[str] = None`  
- `fname: str = "overview.png"`
- `show: bool = True`
- `color: str = "C0"`
- `band_alpha: float = 0.25`
- `band_label: Optional[str] = None` (defaults to `"90% band"` etc.)
- `title_mu: str = r"Posterior $\mu_t$"`
- `title_sigma_trace: str = r"trace: $\sigma$"`
- `title_sigma_hist: Optional[str] = None`
- `title_Q: str = r"Process variances (log$_{10}$ scale)"`
- `title_baselines: str = r"Baselines"`
- `title_rmse: str = r"running RMSE($\mu$) vs truth"`
- `xlabel_time: str = r"$t$"`
- `ylabel_mu: str = r"$\mu_t$"`
- `ylims_mu: Optional[Tuple[float,float]] = None`
- `yscale_mu: Optional[str] = None` (e.g. `"log"`)

Example:
```bash
python simulator/dlm_plotter.py --target ... \
  --overview-kw ylims_mu=(10,30) \
  --overview-kw band_alpha=0.15
```


## 3) `figure_trace_acf_core(...)`

This produces the `trace + histogram + ACF` panel for:

- `sigma` (if present)
- `s_alpha`, `s_beta`, `s_gamma` (if present)
- remaining `Q` columns (as `log10(Q)`), if present
- optional other scalar parameters (excluding `lambda2` and `tau_*`)

Signature + defaults:

- `save_dir: Optional[str] = None`
- `show: bool = True`
- `max_lag: int = 200`
- `name_sigma: str = r"$\sigma$"`
- `name_s_alpha: str = r"$s_\alpha$"`
- `name_s_beta: str = r"$s_\beta$"`
- `name_s_gamma: str = r"$s_\gamma$"`
- `plot_other_scalars: bool = True`

### Global trace/hist controls (apply to ALL panels)

- `trace_ylim: Optional[Tuple[float,float]] = None`  
  Zoom only; does not change samples.
- `hist_xlim: Optional[Tuple[float,float]] = None`  
  Zoom only; does not change samples.

### Outlier handling (plot + diagnostics)

- `plot_policy: str = "none"`  
  Options:
  - `"none"`: plot raw series (except non-finites handled separately)
  - `"clip"`: clip series to bounds for plotting/hist
  - `"drop"`: set outliers to `nan` in trace, remove from hist
- `diag_policy: str = "clean"`  
  What ACF/ESS/Geweke use:
  - `"raw"`: finite samples only
  - `"clipped"`: finite samples clipped to bounds
  - `"clean"`: finite samples with outliers removed
- `drop_nonfinite: bool = True`  
  If `True`, removes inf/nan from hist/diagnostics; trace shows them as gaps.
- `clip_q: Optional[Tuple[float,float]] = None`  
  Quantile bounds, e.g. `(0.001, 0.999)`
- `clip_nmad: Optional[float] = None`  
  Median ± `k * (MAD/0.6745)` bounds, e.g. `12.0`
- `max_abs: Optional[float] = None`  
  Absolute hard bound, e.g. `1e3` means clamp to `[-1000,+1000]`
- `hist_bins: int = 40`
- `auto_zoom_if_clipped: bool = True`  
  If clipping/dropping and bounds are finite, automatically zooms trace ylim and hist range.

Notes on bounds:

- Bounds can be combined; internally it intersects them (tightest wins).  
  Example: `clip_q` + `max_abs` means “within both”.

Examples:
```bash
# Simple zoom (no sample modification)
python simulator/dlm_plotter.py --target ... \
  --traceacf-kw trace_ylim=(-5,5) \
  --traceacf-kw hist_xlim=(-5,5)

# Quantile clipping for plotting + clean diagnostics
python simulator/dlm_plotter.py --target ... \
  --traceacf-kw plot_policy="clip" \
  --traceacf-kw diag_policy="clean" \
  --traceacf-kw clip_q=(0.001,0.999)
```


## 4) `figure_states_separate(...)`

Writes up to three files:

- `<fname_prefix>_level.png`
- `<fname_prefix>_slope.png`
- `<fname_prefix>_seasonality.png`

Signature + defaults:

- `save_dir: Optional[str] = None`
- `fname_prefix: str = "state"`
- `show: bool = True`
- `color: str = "C0"`
- `band_alpha: float = 0.25`
- `band_label: Optional[str] = None`
- `xlabel_time: str = r"$t$"`
- `title_level: str = r"Level $\alpha_t$"`
- `ylabel_level: str = r"$\alpha_t$"`
- `title_slope: str = r"Slope $\beta_t$"`
- `ylabel_slope: str = r"$\beta_t$"`
- `title_seasonality: str = r"Seasonality $\gamma_t$ (contribution)"`
- `ylabel_seasonality: str = r"$\gamma_t$"`
- `slope_scale: float = 1.0`  
  Only rescales slope values plotted (no label/title change by design).
- `ylims: Optional[Dict[str,Tuple[float,float]]] = None`  
  Dict keys: `"level"`, `"slope"`, `"seasonality"`.
- `yscales: Optional[Dict[str,str]] = None`  
  Dict keys: `"level"`, `"slope"`, `"seasonality"`.
- `zero_line_slope: bool = True`
- `zero_line_seasonality: bool = True`

Example:
```bash
python simulator/dlm_plotter.py --target ... \
  --states-kw slope_scale=100 \
  --states-kw ylims={"slope":(-2,2)}
```


## 5) `quick_report(...)`

Signature + defaults:

- `save_dir: Optional[str] = None`
- `fname: str = "quick_report.png"`
- `show: bool = True`
- `color: str = "C0"`
- `band_alpha: float = 0.25`
- `band_label: Optional[str] = None`
- `title_mu: str = r"$\mu_t$"`
- `title_sigma: str = r"$\sigma \mid y$"`
- `title_scale: str = r"process scale"`
- `xlabel_time: str = r"$t$"`
- `ylabel_mu: str = r"$\mu_t$"`
- `ylabel_scale: Optional[str] = None`

Example:
```bash
python simulator/dlm_plotter.py --target ... \
  --quick-kw band_alpha=0.15 \
  --quick-kw title_scale="s_alpha posterior"
```


## Practical recommendations for “weird outliers”

If you suspect numerical glitches (huge spikes), don’t delete draws from disk first—start by clipping or dropping just for plotting, and compute diagnostics on the cleaned series.

Recommended default:
```bash
python simulator/dlm_plotter.py --target ... \
  --traceacf-kw plot_policy="drop" \
  --traceacf-kw diag_policy="clean" \
  --traceacf-kw clip_q=(0.001,0.999)
```

If tails are legitimately heavy but occasional blow-ups happen:
```bash
python simulator/dlm_plotter.py --target ... \
  --traceacf-kw plot_policy="clip" \
  --traceacf-kw diag_policy="clipped" \
  --traceacf-kw clip_nmad=12.0
```
