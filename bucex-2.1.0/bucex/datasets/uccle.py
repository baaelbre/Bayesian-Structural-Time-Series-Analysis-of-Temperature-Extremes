"""Reproducible workflows for the six Uccle temperature series."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ..api.fit import fit
from ..core.fit import FitResult
from ..inference.config import Laplace, MCMC, Particles
from ..components import DummySeasonal, LocalLevel, LocalLinearTrend
from ..models.factor import Channel, Factor, FactorModel, Loading
from ..observation import GEV, Gaussian


UCCLE_SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
UCCLE_INFO = {
    "TXm": {"family": "gaussian", "tail": "max", "description": "monthly mean daily maximum temperature"},
    "TNm": {"family": "gaussian", "tail": "max", "description": "monthly mean daily minimum temperature"},
    "TXx": {"family": "gev", "tail": "max", "description": "monthly maximum daily maximum temperature"},
    "TXn": {"family": "gev", "tail": "min", "description": "monthly minimum daily maximum temperature"},
    "TNx": {"family": "gev", "tail": "max", "description": "monthly maximum daily minimum temperature"},
    "TNn": {"family": "gev", "tail": "min", "description": "monthly minimum daily minimum temperature"},
}


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    candidates.extend(
        [
            Path.cwd() / "data",
            Path(__file__).resolve().parent / "data",
            Path(__file__).resolve().parents[1] / "data",
        ]
    )
    for candidate in candidates:
        if all((candidate / f"{name}.csv").is_file() for name in UCCLE_SERIES):
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate all six Uccle monthly files. Tried: {tried}")


def load_uccle_series(
    series: str,
    data_dir: str | Path | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Load and validate one monthly Uccle series.

    The six derived monthly files are bundled with the installed package, so
    ``data_dir`` is optional.  When supplied, it is checked first; this also
    lets a source workflow pass the directory containing the distinct daily
    source file without duplicating the monthly files there.
    """

    if series not in UCCLE_INFO:
        raise ValueError(f"Unknown Uccle series '{series}'. Choose from {UCCLE_SERIES}.")
    explicit = None if data_dir is None else Path(data_dir) / f"{series}.csv"
    path = (
        explicit
        if explicit is not None and explicit.is_file()
        else _resolve_data_dir(data_dir) / f"{series}.csv"
    )
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError(f"{path.name} must contain a date column.")
    value_column = series if series in frame else next((c for c in frame if c != "date"), None)
    if value_column is None:
        raise ValueError(f"{path.name} has no value column.")
    dates = pd.to_datetime(frame["date"], errors="raise")
    values = pd.to_numeric(frame[value_column], errors="raise")
    if dates.duplicated().any():
        raise ValueError(f"{path.name} contains duplicate dates.")
    result = pd.Series(values.to_numpy(dtype=float), index=dates, name=series).sort_index()
    if not np.all(np.isfinite(result.to_numpy())):
        raise ValueError(f"{path.name} contains missing or non-finite values.")
    expected = pd.date_range(result.index[0], result.index[-1], freq="MS")
    if not result.index.equals(expected):
        raise ValueError(f"{path.name} is not a complete consecutive monthly series.")
    if start is not None:
        result = result.loc[pd.Timestamp(start) :]
    if end is not None:
        result = result.loc[: pd.Timestamp(end)]
    if result.size < 24:
        raise ValueError("At least 24 monthly observations are required.")
    return result


def load_uccle_daily(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the supplied daily Uccle source file when using the source tree."""

    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    candidates.extend([Path.cwd() / "data", Path(__file__).resolve().parents[1] / "data"])
    path = next(
        (candidate / "Uccle_24_10_23.csv" for candidate in candidates if (candidate / "Uccle_24_10_23.csv").is_file()),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            "Uccle_24_10_23.csv is distributed with the source repository, not the wheel; "
            "pass data_dir explicitly when needed."
        )
    frame = pd.read_csv(path)
    required = {"DAY", "TX", "TN", "RR"}
    if set(frame) != required:
        raise ValueError(f"Daily file must contain exactly {sorted(required)}.")
    frame["DAY"] = pd.to_datetime(frame["DAY"], errors="raise")
    return frame.sort_values("DAY").reset_index(drop=True)


def derive_uccle_monthly(
    data_dir: str | Path | None = None,
    *,
    end: str = "2022-12-31",
) -> pd.DataFrame:
    """Recreate the six monthly series from the supplied daily source."""

    daily = load_uccle_daily(data_dir).set_index("DAY").loc[: pd.Timestamp(end)]
    monthly = pd.DataFrame(
        {
            "TXm": daily["TX"].resample("MS").mean(),
            "TNm": daily["TN"].resample("MS").mean(),
            "TXx": daily["TX"].resample("MS").max(),
            "TXn": daily["TX"].resample("MS").min(),
            "TNx": daily["TN"].resample("MS").max(),
            "TNn": daily["TN"].resample("MS").min(),
        }
    )
    monthly.index.name = "date"
    return monthly


def validate_uccle_data(
    data_dir: str | Path | None = None,
    *,
    check_daily: bool = False,
) -> pd.DataFrame:
    """Return a compact integrity table for the six monthly series."""

    rows = []
    for name in UCCLE_SERIES:
        values = load_uccle_series(name, data_dir)
        rows.append(
            {
                "series": name,
                "n": int(values.size),
                "start": values.index[0],
                "end": values.index[-1],
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    table = pd.DataFrame(rows).set_index("series")
    if check_daily:
        rebuilt = derive_uccle_monthly(data_dir)
        for name in UCCLE_SERIES:
            supplied = load_uccle_series(name, data_dir)
            difference = np.abs(
                rebuilt[name].reindex(supplied.index).to_numpy()
                - supplied.to_numpy()
            )
            table.loc[name, "daily_max_abs_difference"] = float(np.nanmax(difference))
            table.loc[name, "daily_mismatches"] = int(np.sum(difference > 1e-10))
    return table


def load_uccle_factor_data(
    data_dir: str | Path | None = None,
    *,
    series: Iterable[str] = UCCLE_SERIES,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load aligned Uccle channels in factor-model column order."""

    selected = tuple(series)
    unknown = sorted(set(selected) - set(UCCLE_SERIES))
    if unknown:
        raise ValueError(f"Unknown Uccle series: {unknown}.")
    columns = [
        load_uccle_series(name, data_dir, start=start, end=end)
        for name in selected
    ]
    frame = pd.concat(columns, axis=1, join="inner")
    if frame.shape[1] != len(selected) or frame.isna().any().any():
        raise ValueError("Uccle factor channels are not completely aligned.")
    return frame.loc[:, list(selected)]


def make_uccle_factor_model(
    *,
    structure: str = "estimated",
    individual: str = "local_level",
    seasonal: str | None = "series_specific",
    seasonal_mode: str = "dynamic",
    period: int = 12,
) -> FactorModel:
    """Construct a documented six-channel Uccle factor graph.

    ``structure='estimated'`` creates the v2.1 manuscript model: one anchored
    common local-linear trend, independent channel local levels and independent
    dummy seasonal blocks.  The
    ``'contrasts'`` alternative creates fixed common, day-night,
    extremes-versus-mean and upper-versus-lower factors.  Contrast loadings are
    converted to each lower-tail channel's internal orientation.
    """

    individual_key = str(individual).lower().replace("-", "_")
    if individual_key not in {"none", "static", "local_level"}:
        raise ValueError("individual must be none, static, or local_level.")
    seasonal_key = (
        "none"
        if seasonal is None
        else str(seasonal).lower().replace("-", "_")
    )
    aliases = {"off": "none", "dummy": "series_specific"}
    seasonal_key = aliases.get(seasonal_key, seasonal_key)
    if seasonal_key not in {"none", "series_specific"}:
        raise ValueError("seasonal must be series_specific/dummy or none/off.")
    if seasonal_mode not in {"dynamic", "static", "off"}:
        raise ValueError("seasonal_mode must be dynamic, static, or off.")
    if int(period) < 2:
        raise ValueError("period must be at least 2.")
    channels = []
    for name in UCCLE_SERIES:
        info = UCCLE_INFO[name]
        observation = Gaussian() if info["family"] == "gaussian" else GEV()
        components = []
        if individual_key != "none":
            components.append(
                LocalLevel(
                    mode="static" if individual_key == "static" else "dynamic"
                )
            )
        if seasonal_key == "series_specific" and seasonal_mode != "off":
            components.append(
                DummySeasonal(period=int(period), mode=seasonal_mode)
            )
        channels.append(
            Channel(
                name,
                observation,
                components=tuple(components),
                tail="lower" if info["tail"] == "min" else None,
                description=info["description"],
            )
        )

    structure_key = str(structure).lower().replace("-", "_")
    if structure_key == "estimated":
        loadings = {
            "TXm": 1.0,
            **{
                name: Loading.estimated(
                    -1.0 if UCCLE_INFO[name]["tail"] == "min" else 1.0,
                    sd=1.0,
                )
                for name in UCCLE_SERIES
                if name != "TXm"
            },
        }
        factors = (
            Factor(
                "common",
                (
                    LocalLinearTrend(
                        initial_level=0.0,
                        initial_slope=0.0,
                        initial_level_sd=0.0,
                        initial_slope_sd=0.0,
                    ),
                ),
                loadings,
                description="estimated shared climate trend",
            ),
        )
    elif structure_key == "contrasts":
        contrast = {
            "common": {name: 1.0 for name in UCCLE_SERIES},
            "day_night": {
                "TXm": 1.0,
                "TNm": -1.0,
                "TXx": 1.0,
                "TXn": 1.0,
                "TNx": -1.0,
                "TNn": -1.0,
            },
            "extremes_mean": {
                "TXm": -2.0,
                "TNm": -2.0,
                "TXx": 1.0,
                "TXn": 1.0,
                "TNx": 1.0,
                "TNn": 1.0,
            },
            "upper_lower": {
                "TXm": 0.0,
                "TNm": 0.0,
                "TXx": 1.0,
                "TXn": -1.0,
                "TNx": 1.0,
                "TNn": -1.0,
            },
        }
        factors = tuple(
            Factor(
                factor_name,
                (
                    LocalLinearTrend(
                        initial_level=0.0,
                        initial_slope=0.0,
                        initial_level_sd=0.0,
                        initial_slope_sd=0.0,
                    ),
                ),
                {
                    name: value
                    * (-1.0 if UCCLE_INFO[name]["tail"] == "min" else 1.0)
                    for name, value in values.items()
                },
                description=f"fixed {factor_name.replace('_', ' ')} contrast",
            )
            for factor_name, values in contrast.items()
        )
    else:
        raise ValueError("structure must be estimated or contrasts.")
    return FactorModel(
        channels=tuple(channels),
        factors=factors,
        name=f"Uccle {structure_key} dynamic factor model",
    )


def fit_uccle_factor(
    data_dir: str | Path | None = None,
    *,
    model: FactorModel | None = None,
    structure: str = "estimated",
    individual: str = "local_level",
    seasonal: str | None = "series_specific",
    seasonal_mode: str = "dynamic",
    period: int = 12,
    start: str | None = None,
    end: str | None = None,
    **kwargs,
) -> FitResult:
    """Load all six aligned series and fit one shared-factor model."""

    resolved_model = (
        make_uccle_factor_model(
            structure=structure,
            individual=individual,
            seasonal=seasonal,
            seasonal_mode=seasonal_mode,
            period=period,
        )
        if model is None
        else model
    )
    values = load_uccle_factor_data(
        data_dir,
        series=resolved_model.channel_names,
        start=start,
        end=end,
    )
    return fit(values, resolved_model, **kwargs)


def _compatibility_mcmc(
    mcmc: MCMC | None,
    *,
    n_iter: int | None,
    burn: int | None,
    thin: int | None,
    chains: int | None,
    seed: int | None,
    progress: bool | None,
) -> MCMC:
    supplied = any(value is not None for value in (n_iter, burn, thin, chains, progress))
    if mcmc is not None and supplied:
        raise ValueError("Give mcmc or the compatibility iteration arguments, not both.")
    if mcmc is not None:
        if seed is not None and seed != mcmc.seed:
            raise ValueError("seed conflicts with mcmc.seed.")
        return mcmc
    if not supplied:
        return MCMC(seed=seed, progress=False if progress is None else progress)
    resolved_n_iter = 2_000 if n_iter is None else int(n_iter)
    resolved_burn = (
        (1_000 if n_iter is None else 0)
        if burn is None
        else int(burn)
    )
    resolved_thin = 1 if thin is None else int(thin)
    if resolved_n_iter <= resolved_burn:
        raise ValueError("Require n_iter > burn.")
    draws = len(range(resolved_burn, resolved_n_iter, resolved_thin))
    return MCMC(
        draws=draws,
        warmup=resolved_burn,
        thin=resolved_thin,
        chains=1 if chains is None else int(chains),
        seed=seed,
        progress=False if progress is None else bool(progress),
    )


def fit_uccle_series(
    series: str,
    data_dir: str | Path | None = None,
    *,
    mcmc: MCMC | None = None,
    priors="manuscript_lasso",
    engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = True,
    start: str | None = None,
    end: str | None = None,
    particles: Particles | None = None,
    laplace: Laplace | None = None,
    n_iter: int | None = None,
    burn: int | None = None,
    thin: int | None = None,
    chains: int | None = None,
    seed: int | None = None,
    progress: bool | None = None,
    state_method: str | None = None,
    state_kwargs: dict | None = None,
    **kwargs,
) -> FitResult:
    """Fit one Uccle series with the canonical or compact compatibility API."""

    values = load_uccle_series(series, data_dir, start=start, end=end)
    info = UCCLE_INFO[series]
    if state_method is not None:
        if engine != "auto":
            raise ValueError("Give engine or state_method, not both.")
        engine = state_method
    options = {} if state_kwargs is None else dict(state_kwargs)
    if options:
        n_particles = int(options.pop("n_particles", options.pop("particle_n_particles", 256)))
        if particles is None and ("n_particles" in state_kwargs or "particle_n_particles" in state_kwargs):
            particles = Particles(n=n_particles)
        max_iterations = int(options.pop("max_iter", 30))
        tolerance = float(options.pop("tol", 1e-5))
        draw_attempts = int(options.pop("draw_attempts", options.pop("max_state_tries", 30)))
        if laplace is None:
            laplace = Laplace(
                max_iterations=max_iterations,
                tolerance=tolerance,
                draw_attempts=draw_attempts,
            )
        if options:
            raise ValueError(f"Unsupported state_kwargs: {sorted(options)}")
    resolved_mcmc = _compatibility_mcmc(
        mcmc,
        n_iter=n_iter,
        burn=burn,
        thin=thin,
        chains=chains,
        seed=seed,
        progress=progress,
    )
    return fit(
        values.to_numpy(),
        family=info["family"],
        period=12,
        priors=priors,
        engine=engine,
        parameterization=parameterization,
        asis=asis,
        mcmc=resolved_mcmc,
        particles=particles,
        laplace=laplace,
        dates=values.index.to_numpy(),
        name=series,
        tail=info["tail"],
        **kwargs,
    )


@dataclass
class UccleFitCollection:
    fits: dict[str, FitResult]
    data_dir: str | None = None

    def __getitem__(self, name: str) -> FitResult:
        return self.fits[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fits)

    def __len__(self) -> int:
        return len(self.fits)

    def keys(self):
        return self.fits.keys()

    def values(self):
        return self.fits.values()

    def items(self):
        return self.fits.items()

    def summary(self):
        rows = []
        for name, result in self.fits.items():
            for parameter, summary in result.static_summary().items():
                rows.append({"series": name, "family": result.family, "parameter": parameter, **summary})
        return pd.DataFrame(rows)

    def plot(self, kind: str = "level", **kwargs):
        from ..plotting import plot_collection

        return plot_collection(self, kind=kind, **kwargs)

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for name, result in self.fits.items():
            result.save(target / f"{name}.bucex")


def fit_uccle_all(
    data_dir: str | Path | None = None,
    *,
    series: Iterable[str] = UCCLE_SERIES,
    mcmc: MCMC | None = None,
    seed: int | None = None,
    **kwargs,
) -> UccleFitCollection:
    selected = list(series)
    unknown = set(selected) - set(UCCLE_SERIES)
    if unknown:
        raise ValueError(f"Unknown Uccle series: {sorted(unknown)}")
    compatibility_keys = {"n_iter", "burn", "thin", "chains", "progress"}
    compact = mcmc is None and any(key in kwargs for key in compatibility_keys)
    base = MCMC() if mcmc is None else mcmc
    resolved_seed = base.seed if seed is None else seed
    seeds = np.random.SeedSequence(resolved_seed).spawn(len(selected))
    fits = {}
    resolved_dir = _resolve_data_dir(data_dir)
    for name, sequence in zip(selected, seeds):
        series_seed = int(sequence.generate_state(1)[0])
        if compact:
            fits[name] = fit_uccle_series(
                name,
                resolved_dir,
                seed=series_seed,
                **kwargs,
            )
        else:
            chain_mcmc = MCMC(
                draws=base.draws,
                warmup=base.warmup,
                thin=base.thin,
                chains=base.chains,
                seed=series_seed,
                progress=base.progress,
                adapt=base.adapt,
            )
            fits[name] = fit_uccle_series(
                name,
                resolved_dir,
                mcmc=chain_mcmc,
                **kwargs,
            )
    return UccleFitCollection(fits=fits, data_dir=str(resolved_dir))
