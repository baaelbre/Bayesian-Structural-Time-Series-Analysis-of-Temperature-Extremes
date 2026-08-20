"""Configuration and deterministic artifact paths for the 2.6 presentation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeSettings:
    """MCMC, particle, and simulation settings for one named profile."""

    draws: int
    warmup: int
    chains: int
    particles: int
    simulation_months: int

    def __post_init__(self) -> None:
        if min(self.draws, self.chains, self.particles, self.simulation_months) < 1:
            raise ValueError(
                "draws, chains, particles, and simulation_months must be positive."
            )
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative.")
        if self.particles < 2:
            raise ValueError("particles must be at least two.")
        if self.simulation_months < 24:
            raise ValueError("simulation_months must be at least 24.")


PROFILE_SETTINGS: dict[str, RuntimeSettings] = {
    "smoke": RuntimeSettings(2, 2, 1, 24, 48),
    "pilot": RuntimeSettings(250, 250, 2, 128, 240),
    "publication": RuntimeSettings(2_000, 2_000, 4, 512, 720),
}

PROFILE_WINDOWS: dict[str, tuple[str, str | None]] = {
    "smoke": ("2015-01-01", "2022-12-01"),
    "pilot": ("1980-01-01", None),
    "publication": ("1892-01-01", None),
}


@dataclass(frozen=True)
class PresentationConfig:
    """One serializable configuration used by local scripts and PBS jobs."""

    profile: str = "pilot"
    start: str | None = "1980-01-01"
    end: str | None = None
    seed: int = 26_000
    output_dir: Path = Path("results/presentation")
    data_dir: Path | None = None
    runtime: RuntimeSettings = PROFILE_SETTINGS["pilot"]
    progress: bool = True
    figure_formats: tuple[str, ...] = ("pdf", "png")
    figure_dpi: int = 180

    def __post_init__(self) -> None:
        profile = str(self.profile).lower()
        if profile not in PROFILE_SETTINGS:
            raise ValueError(f"profile must be one of {tuple(PROFILE_SETTINGS)}.")
        formats = tuple(str(value).lower().lstrip(".") for value in self.figure_formats)
        if not formats or any(value not in {"pdf", "png", "svg"} for value in formats):
            raise ValueError("figure_formats must contain pdf, png, and/or svg.")
        if int(self.figure_dpi) < 72:
            raise ValueError("figure_dpi must be at least 72.")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "figure_formats", formats)
        object.__setattr__(self, "figure_dpi", int(self.figure_dpi))
        if self.data_dir is not None:
            object.__setattr__(self, "data_dir", Path(self.data_dir))

    @classmethod
    def for_profile(
        cls,
        profile: str = "pilot",
        *,
        start: str | None = None,
        end: str | None = None,
        seed: int = 26_000,
        output_dir: str | Path = "results/presentation",
        data_dir: str | Path | None = None,
        progress: bool = True,
        draws: int | None = None,
        warmup: int | None = None,
        chains: int | None = None,
        particles: int | None = None,
        simulation_months: int | None = None,
        figure_formats: tuple[str, ...] = ("pdf", "png"),
        figure_dpi: int = 180,
    ) -> "PresentationConfig":
        key = str(profile).lower()
        if key not in PROFILE_SETTINGS:
            raise ValueError(f"profile must be one of {tuple(PROFILE_SETTINGS)}.")
        default_start, default_end = PROFILE_WINDOWS[key]
        runtime = PROFILE_SETTINGS[key]
        overrides = {
            "draws": draws,
            "warmup": warmup,
            "chains": chains,
            "particles": particles,
            "simulation_months": simulation_months,
        }
        runtime = replace(
            runtime,
            **{
                name: int(value)
                for name, value in overrides.items()
                if value is not None
            },
        )
        return cls(
            profile=key,
            start=default_start if start is None else start,
            end=default_end if end is None else end,
            seed=int(seed),
            output_dir=Path(output_dir),
            data_dir=None if data_dir is None else Path(data_dir),
            runtime=runtime,
            progress=bool(progress),
            figure_formats=figure_formats,
            figure_dpi=int(figure_dpi),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["data_dir"] = None if self.data_dir is None else str(self.data_dir)
        payload["figure_formats"] = list(self.figure_formats)
        return payload


@dataclass(frozen=True)
class WorkflowPaths:
    """Stable local/HPC artifact layout."""

    root: Path

    @classmethod
    def from_config(cls, config: PresentationConfig) -> "WorkflowPaths":
        return cls(Path(config.output_dir))

    @property
    def simulations(self) -> Path:
        return self.root / "simulations"

    @property
    def fits(self) -> Path:
        return self.root / "fits"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def simulation_data(self, group: str, scenario: str) -> Path:
        return self.simulations / group / f"{scenario}.csv"

    def simulation_truth(self, group: str, scenario: str) -> Path:
        return self.simulations / group / f"{scenario}.json"

    def simulation_fit(
        self, engine: str, scenario: str, chain: int | None = None
    ) -> Path:
        base = self.fits / "simulations" / engine / scenario
        return base / ("combined.bucex" if chain is None else f"chain_{chain:02d}.bucex")

    def uccle_fit(
        self, engine: str, series: str, chain: int | None = None
    ) -> Path:
        base = self.fits / "uccle" / engine / series
        return base / ("combined.bucex" if chain is None else f"chain_{chain:02d}.bucex")

    def fit_table_dir(self, study: str, engine: str, name: str) -> Path:
        return self.tables / study / engine / name

    def fit_figure_dir(self, study: str, engine: str, name: str) -> Path:
        return self.figures / study / engine / name

    def create(self) -> None:
        for directory in (
            self.root,
            self.simulations,
            self.fits,
            self.tables,
            self.figures,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROFILE_SETTINGS",
    "PROFILE_WINDOWS",
    "PresentationConfig",
    "RuntimeSettings",
    "WorkflowPaths",
]
