"""Deterministic simulation catalogue for the COMPSTAT presentation.

The catalogue separates two pedagogical jobs:

* ``TAIL_SCENARIOS`` hold the latent local-level path fixed and change only
  the GEV shape, so the Fréchet/Gumbel/Weibull distinction is visible;
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
STRUCTURAL_SIGMA = 1.5
STRUCTURAL_XI = -0.20


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

    def __post_init__(self) -> None:
        if self.group not in {"tail", "structure"}:
            raise ValueError("group must be 'tail' or 'structure'.")
        if self.level not in {"fixed", "dynamic"}:
            raise ValueError("level must be fixed or dynamic.")
        if self.trend not in COMPONENT_CODES or self.season not in COMPONENT_CODES:
            raise ValueError("trend and season must be zero, fixed, or dynamic.")
        if int(self.n_time) < 24:
            raise ValueError("A presentation scenario requires at least 24 months.")
        if not np.isfinite(self.sigma) or float(self.sigma) <= 0.0:
            raise ValueError("sigma must be finite and positive.")
        if min(self.sd_level, self.sd_slope, self.sd_seasonal) < 0.0:
            raise ValueError("Process standard deviations cannot be negative.")

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
                    12,
                    mode="static" if self.season == "fixed" else "dynamic",
                    initial_mean=tuple(self._season_initial()),
                    initial_sd=0.0,
                )
            )
        return Model(GEV(), tuple(components), name=self.title)

    def _season_cycle(self) -> np.ndarray:
        # January is cool and July is warm; the twelve effects sum to zero.
        month = np.arange(12, dtype=float)
        cycle = -float(self.season_amplitude) * np.cos(2.0 * np.pi * month / 12.0)
        return cycle - float(np.mean(cycle))

    def _season_initial(self) -> np.ndarray:
        # The dummy-season transition produces month 1 from minus the sum of
        # its previous eleven states, hence the one-position rotation.
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


TAIL_SCENARIOS: tuple[GEVScenario, ...] = (
    GEVScenario(
        name="heavy_tail",
        title="Local level, heavy tail",
        group="tail",
        description="Fréchet-type GEV with xi > 0.",
        n_time=240,
        sigma=STRUCTURAL_SIGMA,
        xi=0.20,
        level="dynamic",
        sd_level=0.10,
        seed=2_601,
    ),
    GEVScenario(
        name="gumbel_tail",
        title="Local level, exponential tail",
        group="tail",
        description="Gumbel limit with xi = 0.",
        n_time=240,
        sigma=STRUCTURAL_SIGMA,
        xi=0.0,
        level="dynamic",
        sd_level=0.10,
        seed=2_601,
    ),
    GEVScenario(
        name="bounded_tail",
        title="Local level, bounded tail",
        group="tail",
        description="Weibull-type GEV with xi < 0 and a finite endpoint.",
        n_time=240,
        sigma=STRUCTURAL_SIGMA,
        xi=STRUCTURAL_XI,
        level="dynamic",
        sd_level=0.10,
        seed=2_601,
    ),
)


STRUCTURAL_SCENARIOS: tuple[GEVScenario, ...] = (
    GEVScenario(
        "stationary",
        "Stationary location",
        "structure",
        "Fixed level; trend and season absent.",
        720,
        STRUCTURAL_SIGMA,
        STRUCTURAL_XI,
        level="fixed",
        seed=2_610,
    ),
    GEVScenario(
        "linear_trend",
        "Fixed linear trend",
        "structure",
        "Fixed level evolution through a non-zero constant slope.",
        720,
        STRUCTURAL_SIGMA,
        STRUCTURAL_XI,
        level="fixed",
        trend="fixed",
        initial_slope=0.012,
        seed=2_611,
    ),
    GEVScenario(
        "local_level",
        "Stochastic local level",
        "structure",
        "Dynamic level; trend and season absent.",
        720,
        STRUCTURAL_SIGMA,
        STRUCTURAL_XI,
        level="dynamic",
        sd_level=0.08,
        seed=2_612,
    ),
    GEVScenario(
        "local_level_fixed_season",
        "Local level + fixed season",
        "structure",
        "Dynamic level with a fixed annual cycle.",
        720,
        STRUCTURAL_SIGMA,
        STRUCTURAL_XI,
        level="dynamic",
        season="fixed",
        sd_level=0.06,
        season_amplitude=7.0,
        seed=2_613,
    ),
    GEVScenario(
        "local_level_dynamic_season",
        "Local level + dynamic season",
        "structure",
        "Dynamic level and slowly evolving annual cycle.",
        720,
        STRUCTURAL_SIGMA,
        STRUCTURAL_XI,
        level="dynamic",
        season="dynamic",
        sd_level=0.06,
        sd_seasonal=0.035,
        season_amplitude=7.0,
        seed=2_614,
    ),
    GEVScenario(
        "stochastic_trend_fixed_season",
        "Stochastic trend + fixed season",
        "structure",
        "Dynamic slope with no separate level shocks and a fixed annual cycle.",
        720,
        STRUCTURAL_SIGMA,
        STRUCTURAL_XI,
        level="fixed",
        trend="dynamic",
        season="fixed",
        sd_slope=0.00045,
        initial_slope=0.004,
        season_amplitude=7.0,
        seed=2_615,
    ),
    GEVScenario(
        "full_dynamic",
        "Level + trend + dynamic season",
        "structure",
        "All three structural innovation channels are active.",
        720,
        STRUCTURAL_SIGMA,
        STRUCTURAL_XI,
        level="dynamic",
        trend="dynamic",
        season="dynamic",
        sd_level=0.05,
        sd_slope=0.00035,
        sd_seasonal=0.030,
        initial_slope=0.004,
        season_amplitude=7.0,
        seed=2_616,
    ),
)


ALL_SCENARIOS = (*TAIL_SCENARIOS, *STRUCTURAL_SCENARIOS)


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
    dates = pd.date_range("1961-01-01", periods=scenario.n_time, freq="MS")
    table = pd.DataFrame(
        {
            "date": dates,
            "time": np.arange(1, scenario.n_time + 1),
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
    "GEVScenario",
    "STRUCTURAL_SCENARIOS",
    "STRUCTURAL_SIGMA",
    "STRUCTURAL_XI",
    "TAIL_SCENARIOS",
    "scenario_by_name",
    "scenario_catalog",
    "simulate_scenario",
]
