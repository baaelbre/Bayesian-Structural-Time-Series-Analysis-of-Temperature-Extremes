"""The staged analysis behind the Uccle paper and presentation.

The literature-labelled TXx models below are structural analogues fitted in a
common likelihood and inference framework.  They are useful for isolating the
effect of a local level, a changing slope, and structural uncertainty; they
are not claimed to reproduce every modelling or computational detail of the
original publications.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..core import FitResult, combine_fits
from ..datasets import (
    UCCLE_INFO,
    UCCLE_SERIES,
    fit_uccle_hierarchical,
    fit_uccle_series,
    load_uccle_series,
    validate_uccle_data,
)
from ..components import DummySeasonal, LocalLinearTrend
from ..diagnostics import leave_future_out
from ..inference import HierarchicalSampler, MCMC, Particles
from ..models import Model
from ..observation import GEV
from ..priors import (
    HierarchicalPrior,
    ssvs_gaussian_priors,
    ssvs_gev_priors,
)
from .config import PresentationConfig, WorkflowPaths
from .results import export_fit_results, plot_fit_results


BENCHMARK_MODELS = (
    "stationary",
    "linear_trend",
    "huerta_local_level",
    "gaetan_grigoletto_rw2",
    "local_linear_trend",
)

VALIDATION_MODELS = (*BENCHMARK_MODELS, "componentwise_ssvs")

PRESENTATION_STAGES = (
    "data",
    "txx-benchmarks",
    "txx-ssvs",
    "txx-validation",
    "six-univariate",
    "hierarchy-screen",
    "hierarchy-pgas",
    "sensitivity",
    "report",
)

_BENCHMARK_STATES = {
    # level: fixed/dynamic; slope and season: zero/fixed/dynamic.
    "stationary": (0.0, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "linear_trend": (0.0, (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    "huerta_local_level": (1.0, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "gaetan_grigoletto_rw2": (0.0, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "local_linear_trend": (1.0, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}


def componentwise_univariate_prior(series: str):
    """SSVS prior used in the univariate and no-pooling analyses."""

    if series not in UCCLE_INFO:
        raise ValueError(f"Unknown Uccle series {series!r}.")
    builder = (
        ssvs_gaussian_priors
        if UCCLE_INFO[series]["family"] == "gaussian"
        else ssvs_gev_priors
    )
    return builder(
        period=12,
        innovation_slab_sd={"level": 0.03, "trend": 0.0002, "season": 0.03},
        level_dynamic_probability=0.5,
        trend_probabilities=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        # Monthly temperature seasonality is present a priori.
        season_probabilities=(0.0, 0.5, 0.5),
    )


def txx_benchmark_prior(name: str):
    """Forced structural state for one TXx benchmark model."""

    key = str(name).lower().replace("-", "_")
    if key not in _BENCHMARK_STATES:
        raise ValueError(f"benchmark must be one of {BENCHMARK_MODELS}.")
    level, trend, season = _BENCHMARK_STATES[key]
    return ssvs_gev_priors(
        period=12,
        innovation_slab_sd={"level": 0.03, "trend": 0.0002, "season": 0.03},
        level_dynamic_probability=level,
        trend_probabilities=trend,
        season_probabilities=season,
    )


def componentwise_hierarchical_prior(pool: str = "selection") -> HierarchicalPrior:
    """Primary six-series hierarchy: shared probabilities, separate paths."""

    return HierarchicalPrior(
        pool=pool,
        model_space="componentwise",
        level_states=("fixed", "dynamic"),
        trend_states=("zero", "fixed", "dynamic"),
        season_states=("fixed", "dynamic"),
        level_concentration=(1.0, 1.0),
        trend_concentration=(1.0, 1.0, 1.0),
        season_concentration=(1.0, 1.0),
        coefficient_scale={"level": 0.03, "trend": 0.0002, "season": 0.03},
        slab_df=4.0,
        slab_prior_scale={"level": 1.0, "trend": 1.0, "season": 1.0},
    )


@dataclass
class PresentationWorkflow:
    """Run, combine, and report the analyses in presentation order."""

    config: PresentationConfig

    def __post_init__(self) -> None:
        self.paths = WorkflowPaths.from_config(self.config)
        self.paths.create()
        self._write_config()

    def _write_config(self) -> None:
        temporary = self.paths.config.with_name(f".config.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.paths.config)

    def _manifest(self) -> dict[str, Any]:
        if self.paths.manifest.is_file():
            return json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        return {"schema": "bucex-presentation-1", "stages": {}}

    def _record(
        self,
        stage: str,
        *,
        status: str,
        artifacts: Iterable[str | Path] = (),
        details: dict[str, Any] | None = None,
    ) -> None:
        # PBS array tasks may finish at the same instant.  Serialize the short
        # read/modify/write section so one task cannot erase another record.
        lock_path = self.paths.root / ".manifest.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            try:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows local use
                fcntl = None
            try:
                manifest = self._manifest()
                manifest["configuration"] = str(self.paths.config)
                manifest["stages"][stage] = {
                    "status": status,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "artifacts": [str(Path(path)) for path in artifacts],
                    "details": {} if details is None else details,
                }
                temporary = self.paths.manifest.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
                )
                temporary.replace(self.paths.manifest)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _require_chain(chain: int | None) -> int | None:
        if chain is not None and int(chain) < 1:
            raise ValueError("chain is one-based and must be positive.")
        return None if chain is None else int(chain)

    @staticmethod
    def _prepare_target(path: Path, *, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {path}. Pass overwrite=True explicitly."
            )
        path.parent.mkdir(parents=True, exist_ok=True)

    def _mcmc(self, *, chain: int | None, seed_offset: int) -> MCMC:
        runtime = self.config.runtime
        return MCMC(
            draws=runtime.draws,
            warmup=runtime.warmup,
            chains=runtime.chains if chain is None else 1,
            seed=self.config.seed + seed_offset + (0 if chain is None else chain),
            progress=self.config.progress,
        )

    def _particles(self) -> Particles:
        return Particles(n=self.config.runtime.particles, proposal="guided")

    def _export(
        self,
        fit: FitResult,
        *,
        table_name: str,
        source: Path,
        figures: bool,
        diagnostic_plots: bool = False,
    ) -> list[Path]:
        artifacts = export_fit_results(
            fit, self.paths.tables / table_name, source=source
        )
        if figures:
            artifacts.extend(
                plot_fit_results(
                    fit,
                    self.paths.figures / table_name,
                    diagnostics=diagnostic_plots,
                )
            )
        return artifacts

    def validate_data(self) -> Path:
        """Stage 0: validate the six aligned monthly summaries."""

        table = validate_uccle_data(self.config.data_dir)
        target = self.paths.tables / "00_data" / "uccle_integrity.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target)
        self._record("data", status="complete", artifacts=(target,))
        return target

    def run_txx_benchmarks(
        self,
        *,
        benchmark: str | None = None,
        chain: int | None = None,
        engine: str = "pgas",
        overwrite: bool = False,
        figures: bool = False,
    ) -> list[Path]:
        """Stage 1: fit fixed structural alternatives to TXx."""

        chain = self._require_chain(chain)
        names = BENCHMARK_MODELS if benchmark is None else (benchmark,)
        artifacts: list[Path] = []
        for index, name in enumerate(names):
            key = str(name).lower().replace("-", "_")
            target = self.paths.benchmark_fit(key, chain)
            self._prepare_target(target, overwrite=overwrite)
            fit = fit_uccle_series(
                "TXx",
                self.config.data_dir,
                start=self.config.start,
                end=self.config.end,
                priors=txx_benchmark_prior(key),
                engine=engine,
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=self._mcmc(
                    chain=chain,
                    seed_offset=100 + 20 * BENCHMARK_MODELS.index(key),
                ),
                particles=self._particles(),
            )
            fit.metadata["presentation_model"] = key
            fit.metadata["literature_comparison"] = "structural_analogue"
            fit.save(target)
            artifacts.append(target)
            if chain is None:
                artifacts.extend(
                    self._export(
                        fit,
                        table_name=f"01_txx_benchmarks/{key}",
                        source=target,
                        figures=figures,
                    )
                )
        self._record(
            (
                "txx-benchmarks"
                if benchmark is None and chain is None
                else f"txx-benchmarks-{benchmark or 'all'}-chain-{chain or 'combined'}"
            ),
            status="complete",
            artifacts=artifacts,
            details={"engine": engine, "chain": chain},
        )
        return artifacts

    def run_txx_ssvs(
        self,
        *,
        chain: int | None = None,
        engine: str = "pgas",
        overwrite: bool = False,
        figures: bool = False,
        diagnostic_plots: bool = False,
    ) -> Path:
        """Stage 2: let componentwise SSVS select the TXx structure."""

        chain = self._require_chain(chain)
        target = self.paths.txx_ssvs_fit(chain)
        self._prepare_target(target, overwrite=overwrite)
        fit = fit_uccle_series(
            "TXx",
            self.config.data_dir,
            start=self.config.start,
            end=self.config.end,
            priors=componentwise_univariate_prior("TXx"),
            engine=engine,
            parameterization="fruehwirth_schnatter",
            asis=False,
            mcmc=self._mcmc(chain=chain, seed_offset=300),
            particles=self._particles(),
        )
        fit.metadata["presentation_model"] = "txx_componentwise_ssvs"
        fit.save(target)
        artifacts: list[Path] = [target]
        if chain is None:
            artifacts.extend(
                self._export(
                    fit,
                    table_name="02_txx_componentwise_ssvs",
                    source=target,
                    figures=figures,
                    diagnostic_plots=diagnostic_plots,
                )
            )
        self._record(
            "txx-ssvs",
            status="complete",
            artifacts=artifacts,
            details={"engine": engine, "chain": chain},
        )
        return target

    def run_txx_validation(
        self,
        *,
        model_name: str | None = None,
        engine: str = "pgas",
        initial: int | None = None,
        horizon: int = 12,
        step: int | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        """Stage 3: expanding-window forecast validation for TXx models.

        This is the comparison stage.  It evaluates genuinely held-out months
        with proper scores and PIT diagnostics; an attractive reconstructed
        latent path is not treated as evidence of predictive adequacy.
        """

        names = VALIDATION_MODELS if model_name is None else (model_name,)
        unknown = sorted(set(names) - set(VALIDATION_MODELS))
        if unknown:
            raise ValueError(f"Unknown validation model(s): {unknown}.")
        values = load_uccle_series(
            "TXx",
            self.config.data_dir,
            start=self.config.start,
            end=self.config.end,
        )
        n_time = len(values)
        default_holdout = {
            "smoke": 24,
            "pilot": 120,
            "publication": 240,
        }[self.config.profile]
        resolved_initial = (
            max(24, n_time - default_holdout) if initial is None else int(initial)
        )
        resolved_initial = min(resolved_initial, n_time - int(horizon))
        resolved_step = (
            {"smoke": 12, "pilot": 24, "publication": 24}[self.config.profile]
            if step is None
            else int(step)
        )
        model = Model(
            GEV(),
            (LocalLinearTrend(), DummySeasonal(12)),
            name="TXx forecast validation",
        )
        artifacts: list[Path] = []
        for name in names:
            target = self.paths.tables / "03_txx_validation" / name
            marker = target / "settings.json"
            if marker.exists() and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite {target}. Pass overwrite=True explicitly."
                )
            target.mkdir(parents=True, exist_ok=True)
            prior = (
                componentwise_univariate_prior("TXx")
                if name == "componentwise_ssvs"
                else txx_benchmark_prior(name)
            )
            result = leave_future_out(
                values,
                model,
                initial=resolved_initial,
                horizon=int(horizon),
                step=resolved_step,
                fit_options={
                    "priors": prior,
                    "parameterization": "fruehwirth_schnatter",
                    "engine": engine,
                    "asis": False,
                    "mcmc": self._mcmc(
                        chain=None,
                        seed_offset=400 + 20 * VALIDATION_MODELS.index(name),
                    ),
                    "particles": self._particles(),
                },
                forecast_options={
                    "draws": max(100, min(1_000, self.config.runtime.draws)),
                    "seed": self.config.seed + 450,
                },
                thresholds=(35.0, 40.0),
                quantiles=(0.90, 0.95, 0.99),
                progress=self.config.progress,
            )
            scores = target / "scores.csv"
            predictions = target / "predictions.csv"
            pits = target / "pits.csv"
            score_summary = target / "score_summary.csv"
            result.scores.to_csv(scores, index=False)
            result.predictions.to_csv(predictions, index=False)
            result.pits.to_csv(pits, index=False)
            result.score_summary().to_csv(score_summary, index=False)
            marker.write_text(
                json.dumps(
                    {
                        "model": name,
                        "engine": engine,
                        "targets_exact_posterior": engine == "pgas",
                        "initial": resolved_initial,
                        "horizon": int(horizon),
                        "step": resolved_step,
                        "n_origins": result.n_origins,
                        "pit": result.pit_diagnostics().summary,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            artifacts.extend((scores, predictions, pits, score_summary, marker))
        if model_name is None:
            artifacts.extend(self._aggregate_txx_validation())
        self._record(
            "txx-validation" if model_name is None else f"txx-validation-{model_name}",
            status="complete",
            artifacts=artifacts,
            details={
                "engine": engine,
                "initial": resolved_initial,
                "horizon": int(horizon),
                "step": resolved_step,
            },
        )
        return artifacts

    def _aggregate_txx_validation(self) -> list[Path]:
        """Collect independently scheduled validation jobs into two tables."""

        root = self.paths.tables / "03_txx_validation"
        score_frames: list[pd.DataFrame] = []
        pit_rows: list[dict[str, Any]] = []
        for name in VALIDATION_MODELS:
            score_path = root / name / "score_summary.csv"
            settings_path = root / name / "settings.json"
            if score_path.is_file():
                frame = pd.read_csv(score_path)
                frame.insert(0, "model", name)
                score_frames.append(frame)
            if settings_path.is_file():
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                pit_rows.append({"model": name, **settings.get("pit", {})})
        artifacts: list[Path] = []
        if score_frames:
            target = root / "model_score_comparison.csv"
            pd.concat(score_frames, ignore_index=True).to_csv(target, index=False)
            artifacts.append(target)
        if pit_rows:
            target = root / "model_pit_comparison.csv"
            pd.DataFrame(pit_rows).to_csv(target, index=False)
            artifacts.append(target)
        return artifacts

    def run_six_univariate(
        self,
        *,
        series: str | None = None,
        chain: int | None = None,
        gev_engine: str = "pgas",
        overwrite: bool = False,
        figures: bool = False,
    ) -> list[Path]:
        """Stage 4: six independent componentwise-SSVS fits (no pooling)."""

        chain = self._require_chain(chain)
        names = UCCLE_SERIES if series is None else (series,)
        artifacts: list[Path] = []
        for index, name in enumerate(names):
            if name not in UCCLE_INFO:
                raise ValueError(f"Unknown Uccle series {name!r}.")
            target = self.paths.univariate_fit(name, chain)
            self._prepare_target(target, overwrite=overwrite)
            engine = "ffbs" if UCCLE_INFO[name]["family"] == "gaussian" else gev_engine
            fit = fit_uccle_series(
                name,
                self.config.data_dir,
                start=self.config.start,
                end=self.config.end,
                priors=componentwise_univariate_prior(name),
                engine=engine,
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=self._mcmc(
                    chain=chain,
                    seed_offset=500 + 20 * UCCLE_SERIES.index(name),
                ),
                particles=self._particles(),
            )
            fit.metadata["presentation_model"] = "independent_componentwise_ssvs"
            fit.save(target)
            artifacts.append(target)
            if chain is None:
                artifacts.extend(
                    self._export(
                        fit,
                        table_name=f"04_six_univariate/{name}",
                        source=target,
                        figures=figures,
                    )
                )
        self._record(
            (
                "six-univariate"
                if series is None and chain is None
                else f"six-univariate-{series or 'all'}-chain-{chain or 'combined'}"
            ),
            status="complete",
            artifacts=artifacts,
            details={"gev_engine": gev_engine, "chain": chain},
        )
        return artifacts

    def run_hierarchy_screen(
        self,
        *,
        pool: str | None = None,
        overwrite: bool = False,
        figures: bool = False,
    ) -> Path:
        """Stage 5: exploratory Laplace screen for the six-series hierarchy."""

        mode = self.config.pool if pool is None else pool
        target = self.paths.hierarchy_screen(mode)
        self._prepare_target(target, overwrite=overwrite)
        fit = fit_uccle_hierarchical(
            self.config.data_dir,
            start=self.config.start,
            end=self.config.end,
            priors=componentwise_hierarchical_prior(mode),
            engine="laplace",
            parameterization="fruehwirth_schnatter",
            asis=False,
            mcmc=self._mcmc(chain=None, seed_offset=700),
            particles=self._particles(),
            hierarchical_sampler=HierarchicalSampler(
                initializer="laplace",
                channel_workers=self.config.runtime.channel_workers,
            ),
        )
        fit.save(target)
        artifacts = [target, *self._export(
            fit,
            table_name=f"05_hierarchy_screen/{mode}",
            source=target,
            figures=figures,
        )]
        self._record(
            "hierarchy-screen",
            status="complete",
            artifacts=artifacts,
            details={"pool": mode, "approximate": True},
        )
        return target

    def run_hierarchy_pgas(
        self,
        *,
        pool: str | None = None,
        chain: int | None = None,
        screen: str | Path | None = None,
        overwrite: bool = False,
        figures: bool = False,
        diagnostic_plots: bool = False,
    ) -> Path:
        """Stage 6: exact-invariant PGAS, optionally initialized by the screen."""

        mode = self.config.pool if pool is None else pool
        chain = self._require_chain(chain)
        target = self.paths.hierarchy_fit(mode, chain)
        self._prepare_target(target, overwrite=overwrite)
        screen_path = self.paths.hierarchy_screen(mode) if screen is None else Path(screen)
        initial = None
        if screen_path.is_file():
            screen_fit = FitResult.load(screen_path)
            if chain is None:
                initial = screen_fit
            else:
                initial = screen_fit.warm_start(
                    chain=(chain - 1) % screen_fit.n_chains,
                    draw=screen_fit.draws_per_chain - 1,
                )
        fit = fit_uccle_hierarchical(
            self.config.data_dir,
            start=self.config.start,
            end=self.config.end,
            priors=componentwise_hierarchical_prior(mode),
            engine="pgas",
            parameterization="fruehwirth_schnatter",
            asis=False,
            init=initial,
            mcmc=self._mcmc(chain=chain, seed_offset=900),
            particles=self._particles(),
            hierarchical_sampler=HierarchicalSampler(
                initializer="laplace",
                channel_workers=self.config.runtime.channel_workers,
            ),
        )
        fit.save(target)
        artifacts: list[Path] = [target]
        if chain is None:
            artifacts.extend(
                self._export(
                    fit,
                    table_name=f"06_hierarchy_pgas/{mode}",
                    source=target,
                    figures=figures,
                    diagnostic_plots=diagnostic_plots,
                )
            )
        self._record(
            "hierarchy-pgas" if chain is None else f"hierarchy-pgas-chain-{chain}",
            status="complete",
            artifacts=artifacts,
            details={
                "pool": mode,
                "chain": chain,
                "screen": str(screen_path) if screen_path.is_file() else None,
                "exact_invariant": True,
            },
        )
        return target

    def run_sensitivity(
        self,
        *,
        pool: str,
        engine: str = "laplace",
        chain: int | None = None,
        overwrite: bool = False,
        figures: bool = False,
    ) -> Path:
        """Stage 7: repeat the hierarchy under slab or combined pooling."""

        if pool not in {"selection", "slab", "both"}:
            raise ValueError("pool must be selection, slab, or both.")
        if engine not in {"laplace", "pgas"}:
            raise ValueError("engine must be laplace or pgas.")
        chain = self._require_chain(chain)
        if engine == "laplace" and chain is not None:
            raise ValueError("Laplace sensitivity runs use the configured chains together.")
        target = self.paths.sensitivity_fit(pool, engine, chain)
        self._prepare_target(target, overwrite=overwrite)
        initial = None
        screen_path = self.paths.sensitivity_fit(pool, "laplace", None)
        if engine == "pgas" and screen_path.is_file():
            screen_fit = FitResult.load(screen_path)
            initial = (
                screen_fit
                if chain is None
                else screen_fit.warm_start(
                    chain=(chain - 1) % screen_fit.n_chains,
                    draw=screen_fit.draws_per_chain - 1,
                )
            )
        fit = fit_uccle_hierarchical(
            self.config.data_dir,
            start=self.config.start,
            end=self.config.end,
            priors=componentwise_hierarchical_prior(pool),
            engine=engine,
            parameterization="fruehwirth_schnatter",
            asis=False,
            init=initial,
            mcmc=self._mcmc(chain=chain, seed_offset=1100),
            particles=self._particles(),
            hierarchical_sampler=HierarchicalSampler(
                initializer="laplace",
                channel_workers=self.config.runtime.channel_workers,
            ),
        )
        fit.save(target)
        artifacts: list[Path] = [target]
        if chain is None:
            artifacts.extend(
                self._export(
                    fit,
                    table_name=f"07_pooling_sensitivity/{pool}_{engine}",
                    source=target,
                    figures=figures,
                )
            )
        self._record(
            f"sensitivity-{pool}-{engine}",
            status="complete",
            artifacts=artifacts,
            details={"pool": pool, "engine": engine, "chain": chain},
        )
        return target

    def _combine_paths(
        self, sources: list[Path], target: Path, *, overwrite: bool
    ) -> FitResult:
        missing = [path for path in sources if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing independent chain archives: {missing}")
        self._prepare_target(target, overwrite=overwrite)
        fit = combine_fits(FitResult.load(path) for path in sources)
        fit.save(target)
        return fit

    def combine(
        self,
        target_name: str,
        *,
        benchmark: str | None = None,
        series: str | None = None,
        pool: str | None = None,
        engine: str = "pgas",
        overwrite: bool = False,
        figures: bool = False,
        diagnostic_plots: bool = True,
    ) -> Path:
        """Combine independently scheduled chains into one diagnostic fit."""

        n = self.config.runtime.chains
        key = target_name.lower().replace("_", "-")
        if key == "txx-benchmarks":
            if benchmark is None:
                raise ValueError("benchmark= is required.")
            sources = [self.paths.benchmark_fit(benchmark, i) for i in range(1, n + 1)]
            target = self.paths.benchmark_fit(benchmark, None)
            table_name = f"01_txx_benchmarks/{benchmark}"
        elif key == "txx-ssvs":
            sources = [self.paths.txx_ssvs_fit(i) for i in range(1, n + 1)]
            target = self.paths.txx_ssvs_fit(None)
            table_name = "02_txx_componentwise_ssvs"
        elif key == "six-univariate":
            if series is None:
                raise ValueError("series= is required.")
            sources = [self.paths.univariate_fit(series, i) for i in range(1, n + 1)]
            target = self.paths.univariate_fit(series, None)
            table_name = f"04_six_univariate/{series}"
        elif key == "hierarchy-pgas":
            mode = self.config.pool if pool is None else pool
            sources = [self.paths.hierarchy_fit(mode, i) for i in range(1, n + 1)]
            target = self.paths.hierarchy_fit(mode, None)
            table_name = f"06_hierarchy_pgas/{mode}"
        elif key == "sensitivity":
            if pool is None:
                raise ValueError("pool= is required for sensitivity.")
            sources = [
                self.paths.sensitivity_fit(pool, engine, i) for i in range(1, n + 1)
            ]
            target = self.paths.sensitivity_fit(pool, engine, None)
            table_name = f"07_pooling_sensitivity/{pool}_{engine}"
        else:
            raise ValueError(
                "target_name must be txx-benchmarks, txx-ssvs, six-univariate, "
                "hierarchy-pgas, or sensitivity."
            )
        fit = self._combine_paths(sources, target, overwrite=overwrite)
        artifacts = [target, *self._export(
            fit,
            table_name=table_name,
            source=target,
            figures=figures,
            diagnostic_plots=diagnostic_plots,
        )]
        self._record(
            f"combine-{key}", status="complete", artifacts=artifacts
        )
        return target

    def report(
        self, *, figures: bool = True, diagnostic_plots: bool = False
    ) -> list[Path]:
        """Regenerate all tables/figures from completed fits without refitting."""

        artifacts: list[Path] = []
        for source in sorted(self.paths.fits.rglob("*.bucex")):
            if source.name.startswith("chain_"):
                continue
            relative = source.relative_to(self.paths.fits).with_suffix("")
            if relative.name == "combined":
                relative = relative.parent
            fit = FitResult.load(source)
            artifacts.extend(
                self._export(
                    fit,
                    table_name=str(relative),
                    source=source,
                    figures=figures,
                    diagnostic_plots=diagnostic_plots,
                )
            )
        artifacts.extend(self._aggregate_txx_validation())
        self._record("report", status="complete", artifacts=artifacts)
        return artifacts

    def run_all(
        self,
        *,
        overwrite: bool = False,
        figures: bool = False,
        include_sensitivity: bool = False,
    ) -> None:
        """Sequential local run; publication profiles are better submitted to PBS."""

        self.validate_data()
        self.run_txx_benchmarks(overwrite=overwrite, figures=figures)
        self.run_txx_ssvs(overwrite=overwrite, figures=figures)
        self.run_txx_validation(overwrite=overwrite)
        self.run_six_univariate(overwrite=overwrite, figures=figures)
        self.run_hierarchy_screen(overwrite=overwrite, figures=figures)
        self.run_hierarchy_pgas(overwrite=overwrite, figures=figures)
        if include_sensitivity:
            for pool in ("slab", "both"):
                self.run_sensitivity(
                    pool=pool,
                    engine="laplace",
                    overwrite=overwrite,
                    figures=figures,
                )
        self.report(figures=figures, diagnostic_plots=False)
