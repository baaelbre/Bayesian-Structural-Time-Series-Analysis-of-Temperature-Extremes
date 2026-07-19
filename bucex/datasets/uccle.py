from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import numpy as np
import pandas as pd

from ..__about__ import __version__
from ..api.fit import fit_bayes
from ..core.results import PosteriorBundle
from ..inference.fit.base import GibbsConfig

UCCLE_SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
UCCLE_INFO: Mapping[str, dict[str, Any]] = {
    "TXm": {"family": "gaussian", "tail": "max", "description": "monthly mean daily maximum temperature"},
    "TNm": {"family": "gaussian", "tail": "max", "description": "monthly mean daily minimum temperature"},
    "TXx": {"family": "gev", "tail": "max", "description": "monthly maximum of daily maximum temperature"},
    "TXn": {"family": "gev", "tail": "min", "description": "monthly minimum of daily maximum temperature"},
    "TNx": {"family": "gev", "tail": "max", "description": "monthly maximum of daily minimum temperature"},
    "TNn": {"family": "gev", "tail": "min", "description": "monthly minimum of daily minimum temperature"},
}


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    candidates.extend(
        [
            Path.cwd() / "data",
            Path(__file__).resolve().parents[2] / "data",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and all((candidate / f"{name}.csv").exists() for name in UCCLE_SERIES):
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate all six Uccle CSV files. Tried: {tried}")


def load_uccle_series(
    series: str,
    data_dir: str | Path | None = None,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """Load one manuscript Uccle monthly series."""
    if series not in UCCLE_INFO:
        raise ValueError(f"Unknown Uccle series '{series}'. Choose from {UCCLE_SERIES}.")
    root = _resolve_data_dir(data_dir)
    frame = pd.read_csv(root / f"{series}.csv")
    if "date" not in frame.columns:
        raise ValueError(f"{series}.csv must contain a 'date' column.")
    values = pd.to_numeric(frame[series] if series in frame.columns else frame.iloc[:, 1], errors="coerce")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    out = pd.Series(values.to_numpy(dtype=float), index=dates, name=series).dropna().sort_index()
    if start is not None:
        out = out.loc[pd.Timestamp(start):]
    if end is not None:
        out = out.loc[:pd.Timestamp(end)]
    if len(out) < 24:
        raise ValueError(f"{series} contains too few observations after filtering.")
    if int(out.index[0].month) != 1:
        january = np.flatnonzero(out.index.month == 1)
        if january.size:
            out = out.iloc[int(january[0]):]
    n = len(out) - len(out) % 12
    return out.iloc[:n]


@dataclass
class UccleFitCollection:
    """Dictionary-like container returned by :func:`fit_uccle_all`."""

    fits: dict[str, PosteriorBundle]
    data_dir: Optional[str] = None

    def __getitem__(self, key: str) -> PosteriorBundle:
        return self.fits[key]

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

    def plot(self, type: str = "level_slope", **kwargs):
        from ..plotting import plot

        return plot(self, type=type, **kwargs)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, fit in self.fits.items():
            fit.save(directory / f"{name}_bucex_v{__version__}.pkl")

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, fit in self.fits.items():
            row: dict[str, Any] = {"series": name, "family": fit.obs_name, **fit.acceptance}
            for key, stats in fit.static_summary().items():
                row[f"{key}_median"] = stats["median"]
                row[f"{key}_lower"] = stats["lower"]
                row[f"{key}_upper"] = stats["upper"]
            rows.append(row)
        return pd.DataFrame(rows).set_index("series")


def fit_uccle_series(
    series: str,
    data_dir: str | Path | None = None,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    priors: Any = "manuscript",
    config: Optional[GibbsConfig] = None,
    n_iter: Optional[int] = None,
    burn: Optional[int] = None,
    thin: Optional[int] = None,
    seed: Optional[int] = 40,
    progress: Optional[bool] = None,
    state_method: str = "auto",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    """Fit one of TXm, TNm, TXx, TXn, TNx or TNn with manuscript defaults."""
    values = load_uccle_series(series, data_dir, start=start, end=end)
    info = UCCLE_INFO[series]
    return fit_bayes(
        values.to_numpy(),
        family=info["family"],
        priors=priors,
        period=12,
        dates=values.index.to_numpy(),
        name=series,
        tail=info["tail"],
        config=config,
        n_iter=n_iter,
        burn=burn,
        thin=thin,
        seed=seed,
        progress=progress,
        state_method=state_method,
        state_kwargs=state_kwargs,
    )


def fit_uccle_all(
    data_dir: str | Path | None = None,
    *,
    series: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    priors: Any = "manuscript",
    config: Optional[GibbsConfig] = None,
    n_iter: Optional[int] = None,
    burn: Optional[int] = None,
    thin: Optional[int] = None,
    seed: Optional[int] = 40,
    progress: Optional[bool] = None,
    state_method: str = "auto",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> UccleFitCollection:
    """Fit all six manuscript series with one call.

    Separate deterministic seeds are used for each series so that a failed or
    re-run fit does not change the random stream of the others.
    """
    selected = list(UCCLE_SERIES if series is None else series)
    unknown = set(selected) - set(UCCLE_SERIES)
    if unknown:
        raise ValueError(f"Unknown Uccle series: {sorted(unknown)}")
    resolved = _resolve_data_dir(data_dir)
    fits: dict[str, PosteriorBundle] = {}
    for i, name in enumerate(selected):
        fits[name] = fit_uccle_series(
            name,
            resolved,
            start=start,
            end=end,
            priors=priors,
            config=config,
            n_iter=n_iter,
            burn=burn,
            thin=thin,
            seed=None if seed is None else int(seed) + i,
            progress=progress,
            state_method=state_method,
            state_kwargs=state_kwargs,
        )
    return UccleFitCollection(fits=fits, data_dir=str(resolved))
