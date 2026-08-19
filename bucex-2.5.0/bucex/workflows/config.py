"""Configuration and deterministic paths for the Uccle presentation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeSettings:
    """MCMC and execution settings for one named runtime profile."""

    draws: int
    warmup: int
    chains: int
    particles: int
    channel_workers: int

    def __post_init__(self) -> None:
        if min(self.draws, self.chains, self.particles, self.channel_workers) < 1:
            raise ValueError("draws, chains, particles, and workers must be positive.")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative.")
        if self.particles < 2:
            raise ValueError("particles must be at least two.")


PROFILE_SETTINGS: dict[str, RuntimeSettings] = {
    # A short end-to-end software check, not an inferential result.
    "smoke": RuntimeSettings(
        draws=2, warmup=2, chains=1, particles=20, channel_workers=1
    ),
    # Useful for timings, plots, and catching model/data problems.
    "pilot": RuntimeSettings(
        draws=200, warmup=200, chains=2, particles=128, channel_workers=2
    ),
    # Starting point for final inference; still inspect particle sensitivity.
    "publication": RuntimeSettings(
        draws=2_000, warmup=2_000, chains=4, particles=512, channel_workers=6
    ),
}

PROFILE_WINDOWS: dict[str, tuple[str, str | None]] = {
    "smoke": ("2018-01-01", "2022-12-01"),
    "pilot": ("1980-01-01", None),
    "publication": ("1892-01-01", None),
}


@dataclass(frozen=True)
class PresentationConfig:
    """One serializable configuration shared by every presentation stage."""

    profile: str = "pilot"
    start: str | None = "1980-01-01"
    end: str | None = None
    pool: str = "selection"
    seed: int = 25_000
    output_dir: Path = Path("results/presentation")
    data_dir: Path | None = None
    runtime: RuntimeSettings = PROFILE_SETTINGS["pilot"]
    progress: bool = True

    def __post_init__(self) -> None:
        profile = str(self.profile).lower()
        if profile not in PROFILE_SETTINGS:
            raise ValueError(f"profile must be one of {tuple(PROFILE_SETTINGS)}.")
        pool = str(self.pool).lower().replace("-", "_")
        if pool not in {"selection", "slab", "both"}:
            raise ValueError("pool must be selection, slab, or both.")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "pool", pool)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.data_dir is not None:
            object.__setattr__(self, "data_dir", Path(self.data_dir))

    @classmethod
    def for_profile(
        cls,
        profile: str = "pilot",
        *,
        start: str | None = None,
        end: str | None = None,
        pool: str = "selection",
        seed: int = 25_000,
        output_dir: str | Path = "results/presentation",
        data_dir: str | Path | None = None,
        progress: bool = True,
        draws: int | None = None,
        warmup: int | None = None,
        chains: int | None = None,
        particles: int | None = None,
        channel_workers: int | None = None,
    ) -> "PresentationConfig":
        """Construct a profile, with explicit runtime overrides when needed."""

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
            "channel_workers": channel_workers,
        }
        runtime = replace(
            runtime,
            **{name: int(value) for name, value in overrides.items() if value is not None},
        )
        return cls(
            profile=key,
            start=default_start if start is None else start,
            end=default_end if end is None else end,
            pool=pool,
            seed=int(seed),
            output_dir=Path(output_dir),
            data_dir=None if data_dir is None else Path(data_dir),
            runtime=runtime,
            progress=bool(progress),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["data_dir"] = None if self.data_dir is None else str(self.data_dir)
        return payload


@dataclass(frozen=True)
class WorkflowPaths:
    """Stable artifact layout used locally and by scheduler jobs."""

    root: Path

    @classmethod
    def from_config(cls, config: PresentationConfig) -> "WorkflowPaths":
        return cls(Path(config.output_dir))

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

    def benchmark_fit(self, benchmark: str, chain: int | None = None) -> Path:
        base = self.fits / "01_txx_benchmarks" / benchmark
        return base / ("combined.bucex" if chain is None else f"chain_{chain:02d}.bucex")

    def txx_ssvs_fit(self, chain: int | None = None) -> Path:
        base = self.fits / "02_txx_componentwise_ssvs"
        return base / ("combined.bucex" if chain is None else f"chain_{chain:02d}.bucex")

    def univariate_fit(self, series: str, chain: int | None = None) -> Path:
        base = self.fits / "04_six_univariate" / series
        return base / ("combined.bucex" if chain is None else f"chain_{chain:02d}.bucex")

    def hierarchy_screen(self, pool: str = "selection") -> Path:
        return self.fits / "05_hierarchy_screen" / f"{pool}.bucex"

    def hierarchy_fit(self, pool: str = "selection", chain: int | None = None) -> Path:
        base = self.fits / "06_hierarchy_pgas" / pool
        return base / ("combined.bucex" if chain is None else f"chain_{chain:02d}.bucex")

    def sensitivity_fit(
        self, pool: str, engine: str, chain: int | None = None
    ) -> Path:
        base = self.fits / "07_pooling_sensitivity" / f"{pool}_{engine}"
        return base / ("combined.bucex" if chain is None else f"chain_{chain:02d}.bucex")

    def create(self) -> None:
        for directory in (self.root, self.fits, self.tables, self.figures, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
