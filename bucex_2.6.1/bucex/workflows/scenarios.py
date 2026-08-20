"""Deterministic simulation catalogue for the COMPSTAT presentation.

The catalogue separates two pedagogical jobs:

* ``TAIL_SCENARIOS`` hold the latent local-level path fixed and change only
  the GEV shape, so the Weibull/Gumbel/Fréchet distinction is visible;
* ``SCALE_SCENARIOS`` hold the latent path and shape fixed and change only
  ``sigma``, separating observation scale from structural evolution;
* ``STRUCTURAL_SCENARIOS`` hold ``sigma`` and ``xi`` fixed and change only the
  unobserved-component structure, so SSVS recovery has an unambiguous truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from ..components import DummySeasonal, LocalLinearTrend
from ..models import Model
from ..models.compiler import compile_model
from ..observation import GEV
from ..simulate import Simulation, simulate


COMPONENT_CODES = {"zero": 0, "fixed": 1, "dynamic": 2}
DEFAULT_SIMULATION_PERIOD = 4
STRUCTURAL_SIGMA = 1.5
STRUCTURAL_XI = -0.30


@dataclass(frozen=True)
class GEVScenario:
    """One reproducible GEV structural-data experiment."""

    name: str
    title: str
    group: str
    description: str
    n_time: int
    sigma: float
    xi: float
    level: str = "dynamic"
    trend: str = "zero"
    season: str = "zero"
    sd_level: float = 0.0
    sd_slope: float = 0.0
    sd_seasonal: float = 0.0
    initial_level: float = 25.0
    initial_slope: float = 0.0
    season_amplitude: float = 0.0
    seed: int = 1
    period: int = DEFAULT_SIMULATION_PERIOD

    def __post_init__(self) -> None:
        if self.group not in {"tail", "scale", "structure"}:
            raise ValueError("group must be 'tail', 'scale', or 'structure'.")
        if self.level not in {"fixed", "dynamic"}:
            raise ValueError("level must be fixed or dynamic.")
        if self.trend not in COMPONENT_CODES or self.season not in COMPONENT_CODES:
            raise ValueError("trend and season must be zero, fixed, or dynamic.")
        if int(self.n_time) != self.n_time or int(self.n_time) < 24:
            raise ValueError("A presentation scenario requires at least 24 blocks.")
        if int(self.period) != self.period or int(self.period) < 2:
            raise ValueError("period must be an integer of at least two.")
        if not np.isfinite(self.sigma) or float(self.sigma) <= 0.0:
            raise ValueError("sigma must be finite and positive.")
        if not np.isfinite(self.xi):
            raise ValueError("xi must be finite.")
        process_sd = np.asarray(
            (self.sd_level, self.sd_slope, self.sd_seasonal), dtype=float
        )
        if np.any(~np.isfinite(process_sd)) or np.any(process_sd < 0.0):
            raise ValueError("Process standard deviations must be finite and non-negative.")
        state_values = np.asarray(
            (self.initial_level, self.initial_slope, self.season_amplitude),
            dtype=float,
        )
        if np.any(~np.isfinite(state_values)):
            raise ValueError("Initial states and seasonal amplitude must be finite.")

    def resized(self, n_time: int) -> "GEVScenario":
        return replace(self, n_time=int(n_time))

    @property
    def model(self) -> Model:
        trend = LocalLinearTrend(
            level_mode="dynamic" if self.level == "dynamic" else "static",
            trend_mode={"zero": "off", "fixed": "static", "dynamic": "dynamic"}[
                self.trend
            ],
            initial_level=self.initial_level,
            initial_slope=self.initial_slope,
            initial_level_sd=0.0,
            initial_slope_sd=0.0,
        )
        components: list[Any] = [trend]
        if self.season != "zero":
            components.append(
                DummySeasonal(
                    self.period,
                    mode="static" if self.season == "fixed" else "dynamic",
                    initial_mean=tuple(self._season_initial()),
                    initial_sd=0.0,
                )
            )
        return Model(GEV(), tuple(components), name=self.title)

    def _season_cycle(self) -> np.ndarray:
        # The effects complete one cycle and sum to zero for any period.
        phase = np.arange(self.period, dtype=float)
        cycle = -float(self.season_amplitude) * np.cos(
            2.0 * np.pi * phase / self.period
        )
        return cycle - float(np.mean(cycle))

    def _season_initial(self) -> np.ndarray:
        # The dummy-season transition produces phase 1 from minus the sum of
        # its previous period-1 states, hence the one-position rotation.
        return self._season_cycle()[1:]

    @property
    def initial_state(self) -> np.ndarray:
        values = [float(self.initial_level)]
        if self.trend != "zero":
            values.append(float(self.initial_slope))
        if self.season != "zero":
            values.extend(self._season_initial().tolist())
        return np.asarray(values, dtype=float)

    @property
    def params(self) -> dict[str, float]:
        values = {"sigma": float(self.sigma), "xi": float(self.xi)}
        if self.level == "dynamic":
            values["sd.level"] = float(self.sd_level)
        if self.trend == "dynamic":
            values["sd.slope"] = float(self.sd_slope)
        if self.season == "dynamic":
            values["sd.seasonal"] = float(self.sd_seasonal)
        return values

    @property
    def structural_truth(self) -> dict[str, int]:
        return {
            "level": COMPONENT_CODES[self.level],
            "slope": COMPONENT_CODES[self.trend],
            "seasonal": COMPONENT_CODES[self.season],
        }

    @property
    def parameter_truth(self) -> dict[str, float]:
        return {
            "sigma": float(self.sigma),
            "xi": float(self.xi),
            "sd.level": float(self.sd_level if self.level == "dynamic" else 0.0),
            "sd.slope": float(self.sd_slope if self.trend == "dynamic" else 0.0),
            "sd.seasonal": float(
                self.sd_seasonal if self.season == "dynamic" else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "structural_truth": self.structural_truth,
            "parameter_truth": self.parameter_truth,
        }


def make_tail_scenarios(
    *,
    n_time: int = 600,
    period: int = DEFAULT_SIMULATION_PERIOD,
    sigma: float = STRUCTURAL_SIGMA,
    xi_values: tuple[float, float, float] = (-0.30, 0.0, 0.30),
    initial_level: float = 25.0,
    level_sd: float = 0.08,
    seed: int = 2_601,
) -> tuple[GEVScenario, ...]:
    """Build the matched bounded, Gumbel, and heavy-tail illustrations."""

    xi_values = tuple(float(value) for value in xi_values)
    if len(xi_values) != 3 or not (
        xi_values[0] < 0.0
        and np.isclose(xi_values[1], 0.0)
        and xi_values[2] > 0.0
    ):
        raise ValueError("xi_values must contain (negative, zero, positive).")
    definitions = (
        ("bounded_tail", "Local level, bounded tail", "Weibull-type GEV with a finite endpoint."),
        ("gumbel_tail", "Local level, exponential tail", "Gumbel limit with xi = 0."),
        ("heavy_tail", "Local level, heavy tail", "Fréchet-type GEV with xi > 0."),
    )
    return tuple(
        GEVScenario(
            name=name,
            title=title,
            group="tail",
            description=description,
            n_time=n_time,
            sigma=sigma,
            xi=xi,
            period=period,
            level="dynamic",
            sd_level=level_sd,
            initial_level=initial_level,
            seed=seed,
        )
        for (name, title, description), xi in zip(definitions, xi_values)
    )


def make_scale_scenarios(
    *,
    n_time: int = 600,
    period: int = DEFAULT_SIMULATION_PERIOD,
    sigma_values: tuple[float, float, float] = (0.75, 1.50, 3.00),
    xi: float = STRUCTURAL_XI,
    initial_level: float = 25.0,
    level_sd: float = 0.08,
    seed: int = 2_602,
) -> tuple[GEVScenario, ...]:
    """Build three scale experiments sharing one latent local-level path."""

    sigma_values = tuple(float(value) for value in sigma_values)
    if len(sigma_values) != 3 or any(value <= 0.0 for value in sigma_values):
        raise ValueError("sigma_values must contain three positive scales.")
    definitions = (
        ("low_scale", "Low observation scale"),
        ("reference_scale", "Reference observation scale"),
        ("high_scale", "High observation scale"),
    )
    return tuple(
        GEVScenario(
            name=name,
            title=title,
            group="scale",
            description=f"Same latent process with sigma={sigma:.2f}.",
            n_time=n_time,
            sigma=sigma,
            xi=xi,
            period=period,
            level="dynamic",
            sd_level=level_sd,
            initial_level=initial_level,
            seed=seed,
        )
        for (name, title), sigma in zip(definitions, sigma_values)
    )


def make_structural_scenarios(
    *,
    n_time: int = 800,
    period: int = DEFAULT_SIMULATION_PERIOD,
    sigma: float = STRUCTURAL_SIGMA,
    xi: float = STRUCTURAL_XI,
    initial_level: float = 25.0,
    linear_slope: float = 0.006,
    random_walk_sd: float = 0.06,
    local_level_sd: float = 0.035,
    local_slope_sd: float = 0.00050,
    local_initial_slope: float = 0.003,
    dynamic_season_amplitude: float = 1.25,
    fixed_season_amplitude: float = 1.75,
    seasonal_sd: float = 0.04,
    seed: int = 2_610,
) -> tuple[GEVScenario, ...]:
    """Build the six structural truths used for SSVS recovery.

    Together they exercise fixed/dynamic level, absent/fixed/dynamic slope,
    and absent/fixed/dynamic seasonality without adding a redundant seventh
    all-dynamic design.
    """

    common = {
        "group": "structure",
        "n_time": n_time,
        "sigma": sigma,
        "xi": xi,
        "period": period,
        "initial_level": initial_level,
    }
    return (
        GEVScenario(
            name="stationary",
            title="Stationary",
            description="Fixed level; trend and seasonality absent.",
            level="fixed",
            seed=seed,
            **common,
        ),
        GEVScenario(
            name="linear_trend",
            title="Linear trend",
            description="Deterministic non-zero slope; seasonality absent.",
            level="fixed",
            trend="fixed",
            initial_slope=linear_slope,
            seed=seed + 1,
            **common,
        ),
        GEVScenario(
            name="random_walk",
            title="Random walk",
            description="Stochastic level; slope and seasonality absent.",
            level="dynamic",
            sd_level=random_walk_sd,
            seed=seed + 2,
            **common,
        ),
        GEVScenario(
            name="local_linear_trend",
            title="Local linear trend",
            description="Stochastic level and stochastic slope; seasonality absent.",
            level="dynamic",
            trend="dynamic",
            sd_level=local_level_sd,
            sd_slope=local_slope_sd,
            initial_slope=local_initial_slope,
            seed=seed + 3,
            **common,
        ),
        GEVScenario(
            name="stationary_dynamic_season",
            title="Stationary level + changing seasonality",
            description="Fixed level and evolving seasonal cycle; slope absent.",
            level="fixed",
            season="dynamic",
            sd_seasonal=seasonal_sd,
            season_amplitude=dynamic_season_amplitude,
            seed=seed + 4,
            **common,
        ),
        GEVScenario(
            name="local_linear_trend_fixed_season",
            title="Local linear trend + fixed seasonality",
            description="Stochastic level and slope with a fixed seasonal cycle.",
            level="dynamic",
            trend="dynamic",
            season="fixed",
            sd_level=local_level_sd,
            sd_slope=local_slope_sd,
            initial_slope=local_initial_slope,
            season_amplitude=fixed_season_amplitude,
            # Match the no-season local-linear-trend path; only the fixed
            # seasonal contribution changes between the two designs.
            seed=seed + 3,
            **common,
        ),
    )


TAIL_SCENARIOS = make_tail_scenarios()
SCALE_SCENARIOS = make_scale_scenarios()
STRUCTURAL_SCENARIOS = make_structural_scenarios()


ALL_SCENARIOS = (*TAIL_SCENARIOS, *SCALE_SCENARIOS, *STRUCTURAL_SCENARIOS)


def scenario_by_name(name: str, *, group: str | None = None) -> GEVScenario:
    key = str(name).lower().replace("-", "_")
    matches = [
        scenario
        for scenario in ALL_SCENARIOS
        if scenario.name == key and (group is None or scenario.group == group)
    ]
    if not matches:
        available = [
            scenario.name
            for scenario in ALL_SCENARIOS
            if group is None or scenario.group == group
        ]
        raise ValueError(f"Unknown scenario {name!r}; choose from {available}.")
    return matches[0]


def simulate_scenario(scenario: GEVScenario) -> tuple[Simulation, pd.DataFrame]:
    """Simulate a scenario and return a tidy truth table."""

    result = simulate(
        scenario.model,
        scenario.n_time,
        scenario.params,
        initial_state=scenario.initial_state,
        seed=scenario.seed,
    )
    compiled = compile_model(scenario.model, np.zeros(scenario.n_time))
    trend_slice = compiled.component_slices["trend"]
    level = result.states[1:, trend_slice.start]
    slope = (
        result.states[1:, trend_slice.start + 1]
        if trend_slice.stop - trend_slice.start > 1
        else np.zeros(scenario.n_time)
    )
    if "seasonal" in compiled.component_slices:
        seasonal_slice = compiled.component_slices["seasonal"]
        seasonal = result.states[1:, seasonal_slice.start]
    else:
        seasonal = np.zeros(scenario.n_time)
    frequency = "QS" if scenario.period == 4 else "MS"
    dates = pd.date_range("1901-01-01", periods=scenario.n_time, freq=frequency)
    table = pd.DataFrame(
        {
            "date": dates,
            "time": np.arange(1, scenario.n_time + 1),
            "phase": np.arange(scenario.n_time) % scenario.period + 1,
            "y": result.y,
            "eta": result.eta,
            "level": level,
            "slope": slope,
            "seasonal": seasonal,
        }
    )
    return result, table


def scenario_catalog() -> pd.DataFrame:
    """Return the complete simulation design as a presentation-ready table."""

    rows = []
    for scenario in ALL_SCENARIOS:
        row = asdict(scenario)
        row.update(
            truth_level=scenario.structural_truth["level"],
            truth_slope=scenario.structural_truth["slope"],
            truth_seasonal=scenario.structural_truth["seasonal"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "ALL_SCENARIOS",
    "COMPONENT_CODES",
    "DEFAULT_SIMULATION_PERIOD",
    "GEVScenario",
    "SCALE_SCENARIOS",
    "STRUCTURAL_SCENARIOS",
    "STRUCTURAL_SIGMA",
    "STRUCTURAL_XI",
    "TAIL_SCENARIOS",
    "make_scale_scenarios",
    "make_structural_scenarios",
    "make_tail_scenarios",
    "scenario_by_name",
    "scenario_catalog",
    "simulate_scenario",
]
