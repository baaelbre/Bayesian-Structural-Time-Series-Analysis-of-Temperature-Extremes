"""Rolling-origin leave-future-out evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Tuple

import numpy as np
import pandas as pd

from .calibration import pit_diagnostics


def rolling_origin_splits(
    n_time: int,
    initial: int,
    horizon: int = 1,
    step: int = 1,
) -> Iterator[Tuple[range, range]]:
    """Yield expanding-window train/test ranges for time-series validation."""

    n_time = int(n_time)
    initial = int(initial)
    horizon = int(horizon)
    step = int(step)
    if n_time < 2:
        raise ValueError("n_time must be at least 2.")
    if initial <= 0 or initial >= n_time:
        raise ValueError("initial must lie between 1 and n_time - 1.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")

    train_end = initial
    while train_end + horizon <= n_time:
        yield range(0, train_end), range(train_end, train_end + horizon)
        train_end += step


def _slice_rows(value: Any, rows: range):
    if value is None:
        return None
    if hasattr(value, "iloc"):
        return value.iloc[list(rows)]
    return np.asarray(value)[list(rows)]


def _observed_array(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return value.to_numpy(dtype=float)
    return np.asarray(value, dtype=float)


@dataclass
class LFOResult:
    """Results from expanding-window leave-future-out evaluation."""

    predictions: pd.DataFrame
    scores: pd.DataFrame
    pits: pd.DataFrame
    origins: tuple[int, ...]
    forecasts: tuple[Any, ...]
    fits: tuple[Any, ...] | None
    settings: dict[str, Any]

    @property
    def n_origins(self) -> int:
        return len(self.origins)

    def score_summary(self) -> pd.DataFrame:
        """Average and sum each proper score over all held-out forecasts."""

        grouping = ["score", "setting"]
        if "channel" in self.scores.columns:
            grouping.append("channel")
        table = self.scores.copy()
        table["setting"] = table["setting"].astype(object).where(
            table["setting"].notna(), ""
        )
        return (
            table.groupby(grouping, dropna=False)["value"]
            .agg(mean="mean", sum="sum", n="size")
            .reset_index()
        )

    def pit_diagnostics(self, *, channel: str | None = None, bins: int = 10):
        """Summarize the held-out PIT sequence, optionally for one channel."""

        table = self.pits
        if "channel" in table.columns:
            if channel is None:
                raise ValueError(
                    f"Choose channel from {tuple(table['channel'].drop_duplicates())}."
                )
            table = table.loc[table["channel"] == channel]
        elif channel is not None:
            raise ValueError("channel= is only valid for factor-model results.")
        return pit_diagnostics(table["pit"].to_numpy(), bins=bins)

    def plot_pit(
        self,
        *,
        channel: str | None = None,
        bins: int = 10,
        ax=None,
        save=None,
    ):
        return self.pit_diagnostics(channel=channel, bins=bins).plot(
            ax=ax, save=save
        )


def leave_future_out(
    y: Any,
    model: Any,
    *,
    initial: int,
    horizon: int = 1,
    step: int = 1,
    exog: Any = None,
    fit_options: dict[str, Any] | None = None,
    forecast_options: dict[str, Any] | None = None,
    thresholds: tuple[float, ...] | list[float] = (),
    quantiles: tuple[float, ...] | list[float] = (0.9, 0.95, 0.99),
    keep_fits: bool = False,
    progress: bool = True,
) -> LFOResult:
    """Refit at successive origins and score genuinely future observations.

    This is expanding-window leave-future-out prediction, not an in-sample
    posterior-predictive check. Every fold calls the ordinary :func:`bucex.fit`
    and :meth:`FitResult.forecast` APIs, so Gaussian, GEV, and factor models use
    the same interface. Overlapping horizons are retained explicitly through
    ``origin`` and ``horizon`` columns rather than silently deduplicated.
    """

    from ..api.fit import fit

    n_time = len(y)
    splits = tuple(rolling_origin_splits(n_time, initial, horizon, step))
    if not splits:
        raise ValueError("No LFO folds fit inside the supplied series.")
    base_fit_options = {} if fit_options is None else dict(fit_options)
    base_forecast_options = (
        {} if forecast_options is None else dict(forecast_options)
    )
    if "exog" in base_fit_options:
        if exog is not None:
            raise ValueError("Give exog= or fit_options['exog'], not both.")
        exog = base_fit_options.pop("exog")
    supplied_dates = base_fit_options.pop("dates", None)
    if supplied_dates is not None and len(supplied_dates) != n_time:
        raise ValueError("fit_options['dates'] must have the same length as y.")

    prediction_tables: list[pd.DataFrame] = []
    score_tables: list[pd.DataFrame] = []
    pit_rows: list[dict[str, Any]] = []
    forecasts: list[Any] = []
    fits: list[Any] = []
    origins: list[int] = []

    for fold, (train_rows, test_rows) in enumerate(splits, start=1):
        train = _slice_rows(y, train_rows)
        test = _slice_rows(y, test_rows)
        train_exog = _slice_rows(exog, train_rows)
        future_exog = _slice_rows(exog, test_rows)
        train_options = dict(base_fit_options)
        if supplied_dates is not None:
            train_options["dates"] = _slice_rows(supplied_dates, train_rows)
        if progress:
            print(
                f"[LFO] fold {fold}/{len(splits)} | train 1:{train_rows.stop} "
                f"| forecast {test_rows.start + 1}:{test_rows.stop}",
                flush=True,
            )
        fitted = fit(train, model, exog=train_exog, **train_options)

        forecast_kwargs = dict(base_forecast_options)
        if future_exog is not None:
            forecast_kwargs["exog_future"] = future_exog
        if "dates" not in forecast_kwargs:
            if hasattr(y, "index"):
                forecast_kwargs["dates"] = np.asarray(y.index[list(test_rows)])
            elif supplied_dates is not None:
                forecast_kwargs["dates"] = _slice_rows(
                    supplied_dates, test_rows
                )
        if "seed" in forecast_kwargs and forecast_kwargs["seed"] is not None:
            forecast_kwargs["seed"] = int(forecast_kwargs["seed"]) + fold - 1
        forecast = fitted.forecast(horizon, **forecast_kwargs)
        observed = _observed_array(test)
        origin = int(train_rows.stop)
        origins.append(origin)
        forecasts.append(forecast)
        if keep_fits:
            fits.append(fitted)

        prediction = forecast.summary()
        if not isinstance(prediction, pd.DataFrame):
            prediction = pd.DataFrame(prediction)
        prediction.insert(0, "origin", origin)
        if forecast.is_factor_forecast:
            channel_indices = {
                name: index for index, name in enumerate(forecast.channel_names)
            }
            within_channel = prediction.groupby("channel").cumcount()
            prediction["observed"] = [
                float(observed[h, channel_indices[channel]])
                for h, channel in zip(within_channel, prediction["channel"])
            ]
            score = forecast.score(
                observed,
                thresholds=thresholds,
                quantiles=quantiles,
                aggregate=False,
            )
            for channel_index, channel in enumerate(forecast.channel_names):
                pit_values = forecast.pit(
                    observed[:, channel_index], channel=channel
                )
                for h, value in enumerate(pit_values):
                    pit_rows.append(
                        {
                            "origin": origin,
                            "horizon": h + 1,
                            "time": forecast.dates[h],
                            "channel": channel,
                            "pit": float(value),
                        }
                    )
        else:
            observed = observed.reshape(-1)
            prediction["observed"] = observed
            score = forecast.score(
                observed,
                thresholds=thresholds,
                quantiles=quantiles,
                aggregate=False,
            )
            for h, value in enumerate(forecast.pit(observed)):
                pit_rows.append(
                    {
                        "origin": origin,
                        "horizon": h + 1,
                        "time": forecast.dates[h],
                        "pit": float(value),
                    }
                )
        if "channel" in prediction.columns:
            prediction["horizon"] = prediction.groupby("channel").cumcount() + 1
        else:
            prediction["horizon"] = np.arange(1, len(prediction) + 1)
        prediction_tables.append(prediction)

        if not isinstance(score, pd.DataFrame):
            score = pd.DataFrame(score)
        score.insert(0, "origin", origin)
        score["horizon"] = score["time_index"].astype(int) + 1
        score["time"] = [forecast.dates[index] for index in score["time_index"]]
        score_tables.append(score)

    return LFOResult(
        predictions=pd.concat(prediction_tables, ignore_index=True),
        scores=pd.concat(score_tables, ignore_index=True),
        pits=pd.DataFrame(pit_rows),
        origins=tuple(origins),
        forecasts=tuple(forecasts),
        fits=tuple(fits) if keep_fits else None,
        settings={
            "initial": int(initial),
            "horizon": int(horizon),
            "step": int(step),
            "thresholds": tuple(map(float, thresholds)),
            "quantiles": tuple(map(float, quantiles)),
        },
    )
