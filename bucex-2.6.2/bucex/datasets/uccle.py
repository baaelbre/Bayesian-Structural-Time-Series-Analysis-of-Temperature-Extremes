"""Reproducible univariate and hierarchical Uccle data helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ..api.fit import fit
from ..components import DummySeasonal, LocalLinearTrend
from ..core.fit import FitResult
from ..inference.config import Laplace, MCMC, Particles
from ..models.multiseries import Channel, MultiSeriesModel
from ..observation import GEV, Gaussian
from ..priors.hierarchical import HierarchicalPrior, HierarchicalPriors


UCCLE_SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
UCCLE_INFO = {
    "TXm": {"family": "gaussian", "tail": "max", "description": "monthly mean daily maximum temperature"},
    "TNm": {"family": "gaussian", "tail": "max", "description": "monthly mean daily minimum temperature"},
    "TXx": {"family": "gev", "tail": "max", "description": "monthly maximum daily maximum temperature"},
    "TXn": {"family": "gev", "tail": "min", "description": "monthly minimum daily maximum temperature"},
    "TNx": {"family": "gev", "tail": "max", "description": "monthly maximum daily minimum temperature"},
    "TNn": {"family": "gev", "tail": "min", "description": "monthly minimum daily minimum temperature"},
}


def _normalize_series_names(series: Iterable[str] | str) -> tuple[str, ...]:
    """Treat a single series name as one name rather than an iterable of letters."""

    selected = (series,) if isinstance(series, str) else tuple(series)
    unknown = sorted(set(selected) - set(UCCLE_SERIES))
    if unknown:
        raise ValueError(f"Unknown Uccle series: {unknown}; choose from {UCCLE_SERIES}.")
    if len(set(selected)) != len(selected):
        raise ValueError("Uccle series names must be unique.")
    return selected


def _resolve_data_dir(
    data_dir: str | Path | None = None,
    *,
    required_series: Iterable[str] | str = UCCLE_SERIES,
) -> Path:
    required = _normalize_series_names(required_series)
    if data_dir is not None:
        explicit = Path(data_dir)
        missing = [name for name in required if not (explicit / f"{name}.csv").is_file()]
        if not missing:
            return explicit
        has_any_summary = any(
            (explicit / f"{name}.csv").is_file() for name in UCCLE_SERIES
        )
        is_daily_source_directory = (
            explicit.is_dir()
            and (explicit / "Uccle_24_10_23.csv").is_file()
            and not has_any_summary
        )
        if not is_daily_source_directory:
            raise FileNotFoundError(
                f"Explicit Uccle data directory {explicit} is missing: "
                + ", ".join(f"{name}.csv" for name in missing)
            )

    candidates = [Path.cwd() / "data", Path(__file__).resolve().parents[1] / "data"]
    for candidate in candidates:
        if all((candidate / f"{name}.csv").is_file() for name in required):
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    requested = ", ".join(f"{name}.csv" for name in required)
    raise FileNotFoundError(
        f"Could not locate requested Uccle files ({requested}). Tried: {tried}"
    )


def load_uccle_series(
    series: str,
    data_dir: str | Path | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Load one complete monthly Uccle series."""

    if series not in UCCLE_INFO:
        raise ValueError(f"Unknown Uccle series {series!r}; choose from {UCCLE_SERIES}.")
    path = _resolve_data_dir(data_dir, required_series=series) / f"{series}.csv"
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError(f"{path.name} must contain a date column.")
    value_column = series if series in frame else next((c for c in frame if c != "date"), None)
    if value_column is None:
        raise ValueError(f"{path.name} has no value column.")
    dates = pd.to_datetime(frame["date"], errors="raise")
    values = pd.to_numeric(frame[value_column], errors="raise")
    result = pd.Series(values.to_numpy(float), index=dates, name=series).sort_index()
    if result.index.duplicated().any() or not np.all(np.isfinite(result.to_numpy())):
        raise ValueError(f"{path.name} contains duplicate dates or non-finite values.")
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


def load_uccle_multiseries(
    data_dir: str | Path | None = None,
    *,
    series: Iterable[str] | str = UCCLE_SERIES,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load aligned summaries in the requested channel order."""

    selected = _normalize_series_names(series)
    if len(selected) < 2:
        raise ValueError("A hierarchical analysis requires at least two series.")
    frame = pd.concat(
        [load_uccle_series(name, data_dir, start=start, end=end) for name in selected],
        axis=1,
        join="inner",
    )
    if frame.isna().any().any():
        raise ValueError("Uccle channels are not completely aligned.")
    return frame.loc[:, list(selected)]


def load_uccle_daily(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the optional daily source file from a source checkout."""

    candidates = [] if data_dir is None else [Path(data_dir)]
    candidates.extend([Path.cwd() / "data", Path(__file__).resolve().parents[1] / "data"])
    path = next(
        (candidate / "Uccle_24_10_23.csv" for candidate in candidates if (candidate / "Uccle_24_10_23.csv").is_file()),
        None,
    )
    if path is None:
        raise FileNotFoundError("Pass data_dir containing Uccle_24_10_23.csv.")
    frame = pd.read_csv(path)
    required = {"DAY", "TX", "TN", "RR"}
    if set(frame) != required:
        raise ValueError(f"Daily file must contain exactly {sorted(required)}.")
    frame["DAY"] = pd.to_datetime(frame["DAY"], errors="raise")
    return frame.sort_values("DAY").reset_index(drop=True)


def derive_uccle_monthly(data_dir: str | Path | None = None, *, end: str = "2022-12-31") -> pd.DataFrame:
    """Recreate the six summaries from the optional daily source."""

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


def validate_uccle_data(data_dir: str | Path | None = None, *, check_daily: bool = False) -> pd.DataFrame:
    """Return an integrity table for the bundled summaries."""

    # A source checkout keeps the optional daily file in ``data/`` and the six
    # packaged summaries in ``bucex/data/``. If the explicit directory has no
    # summary files at all, use it only for the requested daily cross-check.
    summary_data_dir = data_dir
    if data_dir is not None:
        candidate = Path(data_dir)
        has_any_summary = any(
            (candidate / f"{name}.csv").is_file() for name in UCCLE_SERIES
        )
        if not has_any_summary:
            summary_data_dir = None

    rows = []
    for name in UCCLE_SERIES:
        values = load_uccle_series(name, summary_data_dir)
        rows.append(
            {"series": name, "n": values.size, "start": values.index[0], "end": values.index[-1], "minimum": values.min(), "maximum": values.max()}
        )
    table = pd.DataFrame(rows).set_index("series")
    if check_daily:
        rebuilt = derive_uccle_monthly(data_dir)
        for name in UCCLE_SERIES:
            supplied = load_uccle_series(name, summary_data_dir)
            delta = rebuilt[name].reindex(supplied.index).to_numpy() - supplied.to_numpy()
            table.loc[name, "daily_max_abs_difference"] = np.nanmax(np.abs(delta))
    return table


def make_uccle_hierarchical_model(
    *,
    series: Iterable[str] | str = UCCLE_SERIES,
    seasonal: bool = True,
    period: int = 12,
) -> MultiSeriesModel:
    """Create separate structural paths coupled through a prior hierarchy.

    Seasonality is included in every channel by default. Under hierarchical
    selection it may be fixed or dynamic, but it is not allowed to disappear.
    """

    selected = _normalize_series_names(series)
    if len(selected) < 2:
        raise ValueError("A hierarchical model requires at least two series.")
    if int(period) < 2:
        raise ValueError("period must be at least 2.")
    channels = []
    for name in selected:
        info = UCCLE_INFO[name]
        components = [LocalLinearTrend()]
        if seasonal:
            components.append(DummySeasonal(period=int(period)))
        channels.append(
            Channel(
                name=name,
                observation=Gaussian() if info["family"] == "gaussian" else GEV(),
                components=tuple(components),
                tail="lower" if info["tail"] == "min" else None,
                description=info["description"],
            )
        )
    return MultiSeriesModel(
        channels=tuple(channels),
        name="Uccle hierarchical structural model",
        description="Separate paths with pooled structural probabilities and/or normal-slab scales.",
    )


def fit_uccle_hierarchical(
    data_dir: str | Path | None = None,
    *,
    model: MultiSeriesModel | None = None,
    series: Iterable[str] | str = UCCLE_SERIES,
    pooling: str = "selection",
    priors: HierarchicalPrior | HierarchicalPriors | str | None = None,
    seasonal: bool = True,
    period: int = 12,
    start: str | None = "1980-01-01",
    end: str | None = None,
    **kwargs,
) -> FitResult:
    """Fit aligned Uccle series with pooled SSVS, slabs, or both."""

    resolved_model = model or make_uccle_hierarchical_model(series=series, seasonal=seasonal, period=period)
    values = load_uccle_multiseries(data_dir, series=resolved_model.channel_names, start=start, end=end)
    resolved_priors = HierarchicalPrior(pool=pooling) if priors is None else priors
    options = dict(kwargs)
    options.setdefault("parameterization", "fruehwirth_schnatter")
    options.setdefault("engine", "auto")
    options.setdefault("asis", False)
    return fit(values, resolved_model, priors=resolved_priors, **options)


def fit_uccle_series(
    series: str,
    data_dir: str | Path | None = None,
    *,
    priors: object = "normal",
    mcmc: MCMC | None = None,
    engine: str = "auto",
    parameterization: str = "fruehwirth_schnatter",
    asis: bool = True,
    start: str | None = "1980-01-01",
    end: str | None = None,
    particles: Particles | None = None,
    laplace: Laplace | None = None,
    **kwargs,
) -> FitResult:
    """Fit one summary with the same structural grammar used by the hierarchy."""

    values = load_uccle_series(series, data_dir, start=start, end=end)
    info = UCCLE_INFO[series]
    return fit(
        values.to_numpy(), family=info["family"], period=12, priors=priors,
        engine=engine, parameterization=parameterization, asis=asis,
        mcmc=MCMC() if mcmc is None else mcmc, particles=particles, laplace=laplace,
        dates=values.index.to_numpy(), name=series, tail=info["tail"], **kwargs,
    )


@dataclass
class UccleFitCollection:
    """Small mapping returned by :func:`fit_uccle_all`."""

    fits: dict[str, FitResult]

    def __getitem__(self, name: str) -> FitResult:
        return self.fits[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fits)

    def __len__(self) -> int:
        return len(self.fits)

    def items(self):
        return self.fits.items()

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, result in self.fits.items():
            for parameter, values in result.static_summary().items():
                rows.append({"series": name, "parameter": parameter, **values})
        return pd.DataFrame(rows)


def fit_uccle_all(
    data_dir: str | Path | None = None,
    *,
    series: Iterable[str] | str = UCCLE_SERIES,
    **kwargs,
) -> UccleFitCollection:
    """Fit the requested summaries independently with a common user API."""

    selected = _normalize_series_names(series)
    if not selected:
        raise ValueError("At least one Uccle series is required.")
    return UccleFitCollection(
        {name: fit_uccle_series(name, data_dir, **kwargs) for name in selected}
    )


__all__ = [
    "UCCLE_INFO", "UCCLE_SERIES", "UccleFitCollection", "derive_uccle_monthly",
    "fit_uccle_all", "fit_uccle_hierarchical", "fit_uccle_series",
    "load_uccle_daily", "load_uccle_multiseries", "load_uccle_series",
    "make_uccle_hierarchical_model", "validate_uccle_data",
]
