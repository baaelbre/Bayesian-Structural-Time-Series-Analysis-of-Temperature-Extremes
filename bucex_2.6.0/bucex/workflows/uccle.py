"""Focused simulation and Uccle workflow for the COMPSTAT presentation.

The workflow follows the scientific story rather than the implementation
history of the package:

1. describe the Uccle record, led by TXx;
2. illustrate GEV shape and observation-scale effects under matched local-level
   signals;
3. illustrate absent, fixed, and visibly stochastic structural components;
4. assess componentwise SSVS recovery with a Laplace state update;
5. repeat the same fits with PGAS, initialized from the Laplace posterior;
6. analyse TXx, TXn, TNx, and TNn in that order.

Every fit still enters through :func:`bucex.fit`. This module only fixes the
experimental design, deterministic file names, and local/HPC orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..api import fit
from ..core import FitResult, combine_fits
from ..datasets import UCCLE_INFO, load_uccle_series
from ..inference import MCMC, Particles
from ..priors import ssvs_gev_priors
from .config import PresentationConfig, WorkflowPaths
from .results import (
    collect_selection_probabilities,
    export_fit_results,
    plot_fit_results,
    plot_scale_simulations,
    plot_selection_recovery,
    plot_structural_simulations,
    plot_tail_simulations,
    plot_uccle_record_figures,
    plot_uccle_selection_comparison,
)
from .scenarios import (
    ALL_SCENARIOS,
    SCALE_SCENARIOS,
    STRUCTURAL_SCENARIOS,
    TAIL_SCENARIOS,
    GEVScenario,
    scenario_by_name,
    simulate_scenario,
)


EXTREME_SERIES = ("TXx", "TXn", "TNx", "TNn")
INFERENCE_ENGINES = ("laplace", "pgas")
PRESENTATION_STAGES = (
    "data",
    "tail-simulations",
    "structural-simulations",
    "simulation-fit",
    "uccle-fit",
    "report",
)


def presentation_gev_prior(*, alpha_mean: float, period: int = 12):
    """Return the common componentwise SSVS prior used in all fitted examples.

    Level is fixed or dynamic. Slope and seasonality are independently zero,
    fixed, or dynamic. The signed innovation coefficients have zero-centred
    normal slabs, avoiding inverse-gamma process-variance priors and retaining
    the non-centred Fruehwirth--Schnatter parameterization.
    """

    return ssvs_gev_priors(
        period=int(period),
        alpha_mean=float(alpha_mean),
        beta_mean=0.0,
        beta_sd=0.01,
        innovation_slab_sd={"level": 0.12, "trend": 0.0015, "season": 0.10},
        level_dynamic_probability=0.5,
        trend_probabilities=(1.0 / 3.0,) * 3,
        season_probabilities=(1.0 / 3.0,) * 3,
    )


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _read_truth(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing simulation truth file {path}. Run the simulation stage first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class PresentationWorkflow:
    """Run, combine, and report the v2.6 presentation analyses."""

    config: PresentationConfig

    def __post_init__(self) -> None:
        self.paths = WorkflowPaths.from_config(self.config)
        self.paths.create()
        self._write_config()

    # ------------------------------------------------------------------
    # Reproducibility and artifact bookkeeping
    # ------------------------------------------------------------------
    def _write_config(self) -> None:
        temporary = self.paths.config.with_name(f".config.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                self.config.to_dict(),
                indent=2,
                sort_keys=True,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.paths.config)

    def _manifest(self) -> dict[str, Any]:
        if self.paths.manifest.is_file():
            payload = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
            payload.setdefault("tasks", {})
            return payload
        return {"schema": "bucex-presentation-2", "tasks": {}}

    def _record(
        self,
        task: str,
        *,
        status: str,
        artifacts: Iterable[str | Path] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically update the manifest, including from PBS array tasks."""

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
                manifest["tasks"][task] = {
                    "status": str(status),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "artifacts": [str(Path(path)) for path in artifacts],
                    "details": dict(details or {}),
                }
                temporary = self.paths.manifest.with_name(
                    f".manifest.{os.getpid()}.tmp"
                )
                temporary.write_text(
                    json.dumps(
                        manifest,
                        indent=2,
                        sort_keys=True,
                        default=_json_default,
                    ),
                    encoding="utf-8",
                )
                temporary.replace(self.paths.manifest)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _prepare_target(path: Path, *, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {path}. Pass overwrite=True explicitly."
            )
        path.parent.mkdir(parents=True, exist_ok=True)

    def _require_chain(self, chain: int | None) -> int | None:
        if chain is None:
            return None
        value = int(chain)
        if not 1 <= value <= self.config.runtime.chains:
            raise ValueError(
                f"chain is one-based and must be in 1..{self.config.runtime.chains}."
            )
        return value

    def _mcmc(self, *, chain: int | None, seed_offset: int) -> MCMC:
        runtime = self.config.runtime
        base_seed = self.config.seed + int(seed_offset)
        if chain is None or runtime.chains == 1:
            resolved_seed = base_seed
        else:
            # Match api.fit._chain_seeds exactly so chain k has the same RNG
            # stream whether all chains run locally or as separate PBS tasks.
            streams = np.random.SeedSequence(base_seed).spawn(runtime.chains)
            resolved_seed = int(streams[chain - 1].generate_state(1)[0])
        return MCMC(
            draws=runtime.draws,
            warmup=runtime.warmup,
            chains=runtime.chains if chain is None else 1,
            seed=resolved_seed,
            progress=self.config.progress,
        )

    def _particles(self) -> Particles:
        return Particles(n=self.config.runtime.particles, proposal="guided")

    def _figure_options(self) -> dict[str, Any]:
        return {
            "formats": self.config.figure_formats,
            "dpi": self.config.figure_dpi,
        }

    def _export_fit(
        self,
        fit_result: FitResult,
        *,
        study: str,
        engine: str,
        name: str,
        source: Path,
        truth_table: pd.DataFrame | None = None,
        truth: Mapping[str, Any] | None = None,
        figures: bool = True,
        diagnostics: bool = False,
    ) -> list[Path]:
        artifacts = export_fit_results(
            fit_result,
            self.paths.fit_table_dir(study, engine, name),
            source=source,
            truth=truth,
        )
        if figures:
            artifacts.extend(
                plot_fit_results(
                    fit_result,
                    self.paths.fit_figure_dir(study, engine, name),
                    truth_table=truth_table,
                    truth=truth,
                    diagnostics=diagnostics,
                    **self._figure_options(),
                )
            )
        return artifacts

    # ------------------------------------------------------------------
    # Stage 0: observed record
    # ------------------------------------------------------------------
    def _load_uccle_frame(self) -> pd.DataFrame:
        values = [
            load_uccle_series(
                name,
                self.config.data_dir,
                start=self.config.start,
                end=self.config.end,
            )
            for name in EXTREME_SERIES
        ]
        frame = pd.concat(values, axis=1, join="inner")
        if list(frame.columns) != list(EXTREME_SERIES) or frame.isna().any().any():
            raise ValueError("The four Uccle extreme series are not completely aligned.")
        return frame

    def run_data(
        self,
        *,
        overwrite: bool = False,
        figures: bool = True,
    ) -> list[Path]:
        """Write the aligned record, integrity table, and descriptive figures."""

        frame = self._load_uccle_frame()
        data_target = self.paths.tables / "00_data" / "uccle_extremes.csv"
        integrity_target = self.paths.tables / "00_data" / "uccle_integrity.csv"
        self._prepare_target(data_target, overwrite=overwrite)
        self._prepare_target(integrity_target, overwrite=overwrite)
        data_target.parent.mkdir(parents=True, exist_ok=True)
        frame.rename_axis("date").reset_index().to_csv(data_target, index=False)
        rows = []
        for name in EXTREME_SERIES:
            series = frame[name]
            rows.append(
                {
                    "series": name,
                    "description": UCCLE_INFO[name]["description"],
                    "tail": UCCLE_INFO[name]["tail"],
                    "n": int(series.size),
                    "start": series.index.min(),
                    "end": series.index.max(),
                    "minimum": float(series.min()),
                    "maximum": float(series.max()),
                    "mean": float(series.mean()),
                    "sd": float(series.std()),
                }
            )
        pd.DataFrame(rows).to_csv(integrity_target, index=False)
        artifacts: list[Path] = [data_target, integrity_target]
        if figures:
            artifacts.extend(
                plot_uccle_record_figures(
                    frame,
                    self.paths.figures / "00_data",
                    **self._figure_options(),
                )
            )
        self._record(
            "data",
            status="complete",
            artifacts=artifacts,
            details={"series": list(EXTREME_SERIES), "n_time": len(frame)},
        )
        return artifacts

    # ------------------------------------------------------------------
    # Stages 1 and 2: pedagogical and selection simulations
    # ------------------------------------------------------------------
    def _selected_scenarios(
        self,
        *,
        kind: str,
        scenario: str | None,
    ) -> tuple[GEVScenario, ...]:
        group = str(kind).lower().replace("_", "-")
        if group not in {"all", "tail", "scale", "structure", "structural"}:
            raise ValueError("kind must be all, tail, scale, or structure.")
        if scenario is not None:
            selected = scenario_by_name(scenario)
            if group in {"tail", "scale", "structure", "structural"}:
                expected = (
                    {"structure"}
                    if group in {"structure", "structural"}
                    else ({"tail", "scale"} if group == "tail" else {"scale"})
                )
                if selected.group not in expected:
                    raise ValueError(
                        f"Scenario {selected.name!r} belongs to {selected.group!r}, "
                        f"not one of {sorted(expected)!r}."
                    )
            return (selected,)
        if group == "tail":
            return (*TAIL_SCENARIOS, *SCALE_SCENARIOS)
        if group == "scale":
            return SCALE_SCENARIOS
        if group in {"structure", "structural"}:
            return STRUCTURAL_SCENARIOS
        return ALL_SCENARIOS

    def run_simulations(
        self,
        *,
        kind: str = "all",
        scenario: str | None = None,
        overwrite: bool = False,
        figures: bool = True,
    ) -> list[Path]:
        """Generate reproducible shape, scale, and structural scenarios."""

        selected = self._selected_scenarios(kind=kind, scenario=scenario)
        tables: dict[str, dict[str, pd.DataFrame]] = {
            "tail": {},
            "scale": {},
            "structure": {},
        }
        truths: dict[str, dict[str, dict[str, Any]]] = {
            "tail": {},
            "scale": {},
            "structure": {},
        }
        artifacts: list[Path] = []
        catalog_rows = []
        for base in selected:
            current = base.resized(self.config.runtime.simulation_months)
            data_target = self.paths.simulation_data(current.group, current.name)
            truth_target = self.paths.simulation_truth(current.group, current.name)
            self._prepare_target(data_target, overwrite=overwrite)
            self._prepare_target(truth_target, overwrite=overwrite)
            _, table = simulate_scenario(current)
            truth = current.to_dict()
            data_target.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(data_target, index=False)
            truth_target.write_text(
                json.dumps(truth, indent=2, sort_keys=True, default=_json_default),
                encoding="utf-8",
            )
            tables[current.group][current.name] = table
            truths[current.group][current.name] = truth
            catalog_rows.append(truth)
            artifacts.extend((data_target, truth_target))

        catalog_label = (
            "all"
            if scenario is None and str(kind).lower() == "all"
            else (
                selected[0].group
                if scenario is None
                else f"{selected[0].group}_{selected[0].name}"
            )
        )
        catalog_target = (
            self.paths.tables
            / "10_simulations"
            / f"scenario_catalog_{catalog_label}.csv"
        )
        catalog_target.parent.mkdir(parents=True, exist_ok=True)
        catalog = pd.json_normalize(catalog_rows, sep=".")
        self._prepare_target(catalog_target, overwrite=overwrite)
        catalog.to_csv(catalog_target, index=False)
        artifacts.append(catalog_target)

        if figures:
            if set(tables["tail"]) == {item.name for item in TAIL_SCENARIOS}:
                artifacts.extend(
                    plot_tail_simulations(
                        tables["tail"],
                        truths["tail"],
                        self.paths.figures / "10_simulations" / "tail",
                        **self._figure_options(),
                    )
                )
            if set(tables["scale"]) == {item.name for item in SCALE_SCENARIOS}:
                artifacts.extend(
                    plot_scale_simulations(
                        tables["scale"],
                        truths["scale"],
                        self.paths.figures / "10_simulations" / "scale",
                        **self._figure_options(),
                    )
                )
            if tables["structure"]:
                artifacts.extend(
                    plot_structural_simulations(
                        tables["structure"],
                        truths["structure"],
                        self.paths.figures / "10_simulations" / "structure",
                        **self._figure_options(),
                    )
                )
        task = f"simulations:{kind}:{scenario or 'all'}"
        self._record(
            task,
            status="complete",
            artifacts=artifacts,
            details={"scenarios": [item.name for item in selected]},
        )
        return artifacts

    # ------------------------------------------------------------------
    # Stages 3 and 4: identical simulated fits under Laplace and PGAS
    # ------------------------------------------------------------------
    def _load_simulation(
        self,
        scenario: GEVScenario,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        data_path = self.paths.simulation_data("structure", scenario.name)
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Missing {data_path}. Run structural-simulations first."
            )
        table = pd.read_csv(data_path, parse_dates=["date"])
        truth = _read_truth(self.paths.simulation_truth("structure", scenario.name))
        return table, truth

    def fit_simulations(
        self,
        *,
        engine: str,
        scenario: str | None = None,
        chain: int | None = None,
        overwrite: bool = False,
        figures: bool = False,
        diagnostics: bool = False,
    ) -> list[Path]:
        """Fit structural scenarios with componentwise SSVS.

        PGAS requires the corresponding combined Laplace fit. Its selected
        posterior draw initializes both the static parameters and the complete
        centred state path; this affects burn-in only, not the PGAS target.
        """

        engine = str(engine).lower()
        if engine not in INFERENCE_ENGINES:
            raise ValueError(f"engine must be one of {INFERENCE_ENGINES}.")
        chain = self._require_chain(chain)
        selected = (
            STRUCTURAL_SCENARIOS
            if scenario is None
            else (scenario_by_name(scenario, group="structure"),)
        )
        artifacts: list[Path] = []
        scenario_order = {item.name: index for index, item in enumerate(STRUCTURAL_SCENARIOS)}
        for design in selected:
            table, truth = self._load_simulation(design)
            target = self.paths.simulation_fit(engine, design.name, chain)
            self._prepare_target(target, overwrite=overwrite)
            warm_start = None
            if engine == "pgas":
                laplace_path = self.paths.simulation_fit(
                    "laplace", design.name, None
                )
                if not laplace_path.is_file():
                    raise FileNotFoundError(
                        f"PGAS warm start is missing: {laplace_path}. "
                        "Fit or combine the Laplace result first."
                    )
                warm_start = FitResult.load(laplace_path)
            y = table["y"].to_numpy(float)
            fit_result = fit(
                y,
                family="gev",
                period=12,
                priors=presentation_gev_prior(alpha_mean=float(np.median(y))),
                engine=engine,
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=self._mcmc(
                    chain=chain,
                    seed_offset=(
                        1_000
                        + 100 * scenario_order[design.name]
                        + (10_000 if engine == "pgas" else 0)
                    ),
                ),
                particles=self._particles(),
                dates=table["date"].to_numpy(),
                name=design.name,
                init=warm_start,
            )
            fit_result.metadata.update(
                {
                    "workflow": "presentation-2.6",
                    "study": "simulation",
                    "scenario": design.name,
                    "truth": truth,
                    "warm_start_source_engine": (
                        None if warm_start is None else warm_start.plan.engine
                    ),
                }
            )
            fit_result.save(target)
            current_artifacts: list[Path] = [target]
            # Direct local fits contain all chains and are immediately reportable.
            if chain is None:
                current_artifacts.extend(
                    self._export_fit(
                        fit_result,
                        study="simulations",
                        engine=engine,
                        name=design.name,
                        source=target,
                        truth_table=table,
                        truth=truth,
                        figures=figures,
                        diagnostics=diagnostics,
                    )
                )
            artifacts.extend(current_artifacts)
            self._record(
                f"simulation-fit:{engine}:{design.name}:{chain or 'combined'}",
                status="complete",
                artifacts=current_artifacts,
                details={
                    "engine": engine,
                    "scenario": design.name,
                    "chain": chain,
                    "laplace_warm_start": engine == "pgas",
                },
            )
        return artifacts

    def combine_simulation_fits(
        self,
        *,
        engine: str,
        scenario: str | None = None,
        overwrite: bool = False,
        figures: bool = True,
        diagnostics: bool = True,
    ) -> list[Path]:
        """Combine independently submitted simulation chains."""

        engine = str(engine).lower()
        if engine not in INFERENCE_ENGINES:
            raise ValueError(f"engine must be one of {INFERENCE_ENGINES}.")
        selected = (
            STRUCTURAL_SCENARIOS
            if scenario is None
            else (scenario_by_name(scenario, group="structure"),)
        )
        artifacts: list[Path] = []
        for design in selected:
            sources = [
                self.paths.simulation_fit(engine, design.name, chain)
                for chain in range(1, self.config.runtime.chains + 1)
            ]
            missing = [path for path in sources if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine simulation chains; missing: "
                    + ", ".join(str(path) for path in missing)
                )
            target = self.paths.simulation_fit(engine, design.name, None)
            self._prepare_target(target, overwrite=overwrite)
            combined = combine_fits(FitResult.load(path) for path in sources)
            combined.metadata.update(
                {
                    "workflow": "presentation-2.6",
                    "study": "simulation",
                    "scenario": design.name,
                    "combined_sources": [str(path) for path in sources],
                }
            )
            combined.save(target)
            table, truth = self._load_simulation(design)
            current = [target]
            current.extend(
                self._export_fit(
                    combined,
                    study="simulations",
                    engine=engine,
                    name=design.name,
                    source=target,
                    truth_table=table,
                    truth=truth,
                    figures=figures,
                    diagnostics=diagnostics,
                )
            )
            artifacts.extend(current)
            self._record(
                f"simulation-combine:{engine}:{design.name}",
                status="complete",
                artifacts=current,
            )
        return artifacts

    # ------------------------------------------------------------------
    # Stages 5 and 6: four observed temperature-extreme series
    # ------------------------------------------------------------------
    def fit_uccle(
        self,
        *,
        engine: str,
        series: str | None = None,
        chain: int | None = None,
        overwrite: bool = False,
        figures: bool = False,
        diagnostics: bool = False,
    ) -> list[Path]:
        """Fit TXx, TXn, TNx, and TNn with the same model and prior."""

        engine = str(engine).lower()
        if engine not in INFERENCE_ENGINES:
            raise ValueError(f"engine must be one of {INFERENCE_ENGINES}.")
        if series is not None and series not in EXTREME_SERIES:
            raise ValueError(f"series must be one of {EXTREME_SERIES}.")
        chain = self._require_chain(chain)
        selected = EXTREME_SERIES if series is None else (series,)
        artifacts: list[Path] = []
        series_order = {name: index for index, name in enumerate(EXTREME_SERIES)}
        for name in selected:
            values = load_uccle_series(
                name,
                self.config.data_dir,
                start=self.config.start,
                end=self.config.end,
            )
            target = self.paths.uccle_fit(engine, name, chain)
            self._prepare_target(target, overwrite=overwrite)
            warm_start = None
            if engine == "pgas":
                laplace_path = self.paths.uccle_fit("laplace", name, None)
                if not laplace_path.is_file():
                    raise FileNotFoundError(
                        f"PGAS warm start is missing: {laplace_path}. "
                        "Fit or combine the Laplace result first."
                    )
                warm_start = FitResult.load(laplace_path)
            sign = -1.0 if UCCLE_INFO[name]["tail"] == "min" else 1.0
            fit_result = fit(
                values,
                family="gev",
                period=12,
                priors=presentation_gev_prior(
                    alpha_mean=float(np.median(sign * values.to_numpy(float)))
                ),
                engine=engine,
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=self._mcmc(
                    chain=chain,
                    seed_offset=(
                        30_000
                        + 100 * series_order[name]
                        + (10_000 if engine == "pgas" else 0)
                    ),
                ),
                particles=self._particles(),
                name=name,
                tail=UCCLE_INFO[name]["tail"],
                init=warm_start,
            )
            fit_result.metadata.update(
                {
                    "workflow": "presentation-2.6",
                    "study": "uccle",
                    "series": name,
                    "description": UCCLE_INFO[name]["description"],
                    "warm_start_source_engine": (
                        None if warm_start is None else warm_start.plan.engine
                    ),
                }
            )
            fit_result.save(target)
            current_artifacts: list[Path] = [target]
            if chain is None:
                current_artifacts.extend(
                    self._export_fit(
                        fit_result,
                        study="uccle",
                        engine=engine,
                        name=name,
                        source=target,
                        figures=figures,
                        diagnostics=diagnostics,
                    )
                )
            artifacts.extend(current_artifacts)
            self._record(
                f"uccle-fit:{engine}:{name}:{chain or 'combined'}",
                status="complete",
                artifacts=current_artifacts,
                details={
                    "engine": engine,
                    "series": name,
                    "chain": chain,
                    "laplace_warm_start": engine == "pgas",
                },
            )
        return artifacts

    def combine_uccle_fits(
        self,
        *,
        engine: str,
        series: str | None = None,
        overwrite: bool = False,
        figures: bool = True,
        diagnostics: bool = True,
    ) -> list[Path]:
        """Combine independently submitted Uccle chains."""

        engine = str(engine).lower()
        if engine not in INFERENCE_ENGINES:
            raise ValueError(f"engine must be one of {INFERENCE_ENGINES}.")
        if series is not None and series not in EXTREME_SERIES:
            raise ValueError(f"series must be one of {EXTREME_SERIES}.")
        selected = EXTREME_SERIES if series is None else (series,)
        artifacts: list[Path] = []
        for name in selected:
            sources = [
                self.paths.uccle_fit(engine, name, chain)
                for chain in range(1, self.config.runtime.chains + 1)
            ]
            missing = [path for path in sources if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine Uccle chains; missing: "
                    + ", ".join(str(path) for path in missing)
                )
            target = self.paths.uccle_fit(engine, name, None)
            self._prepare_target(target, overwrite=overwrite)
            combined = combine_fits(FitResult.load(path) for path in sources)
            combined.metadata.update(
                {
                    "workflow": "presentation-2.6",
                    "study": "uccle",
                    "series": name,
                    "combined_sources": [str(path) for path in sources],
                }
            )
            combined.save(target)
            current = [target]
            current.extend(
                self._export_fit(
                    combined,
                    study="uccle",
                    engine=engine,
                    name=name,
                    source=target,
                    figures=figures,
                    diagnostics=diagnostics,
                )
            )
            artifacts.extend(current)
            self._record(
                f"uccle-combine:{engine}:{name}",
                status="complete",
                artifacts=current,
            )
        return artifacts

    # ------------------------------------------------------------------
    # Final report regeneration
    # ------------------------------------------------------------------
    def report(
        self,
        *,
        strict: bool = False,
        diagnostics: bool = False,
    ) -> list[Path]:
        """Regenerate all result tables and figures from combined fit files."""

        artifacts: list[Path] = []
        frame = self._load_uccle_frame()
        artifacts.extend(
            plot_uccle_record_figures(
                frame,
                self.paths.figures / "00_data",
                **self._figure_options(),
            )
        )

        truths: dict[str, dict[str, Any]] = {}
        simulated_fits: dict[tuple[str, str], FitResult] = {}
        observed_fits: dict[tuple[str, str], FitResult] = {}
        missing: list[Path] = []

        for design in STRUCTURAL_SCENARIOS:
            simulation_data = self.paths.simulation_data("structure", design.name)
            simulation_truth = self.paths.simulation_truth("structure", design.name)
            if not simulation_data.is_file() or not simulation_truth.is_file():
                missing.extend(
                    path
                    for path in (simulation_data, simulation_truth)
                    if not path.is_file()
                )
                continue
            table, truth = self._load_simulation(design)
            truths[design.name] = truth
            for engine in INFERENCE_ENGINES:
                source = self.paths.simulation_fit(engine, design.name, None)
                if not source.is_file():
                    missing.append(source)
                    continue
                fit_result = FitResult.load(source)
                simulated_fits[(engine, design.name)] = fit_result
                artifacts.extend(
                    self._export_fit(
                        fit_result,
                        study="simulations",
                        engine=engine,
                        name=design.name,
                        source=source,
                        truth_table=table,
                        truth=truth,
                        figures=True,
                        diagnostics=diagnostics,
                    )
                )

        simulation_selection = collect_selection_probabilities(
            simulated_fits, truths
        )
        selection_target = (
            self.paths.tables / "30_results" / "simulation_selection_recovery.csv"
        )
        selection_target.parent.mkdir(parents=True, exist_ok=True)
        simulation_selection.to_csv(selection_target, index=False)
        artifacts.append(selection_target)
        artifacts.extend(
            plot_selection_recovery(
                simulation_selection,
                self.paths.figures / "30_results",
                **self._figure_options(),
            )
        )

        for name in EXTREME_SERIES:
            for engine in INFERENCE_ENGINES:
                source = self.paths.uccle_fit(engine, name, None)
                if not source.is_file():
                    missing.append(source)
                    continue
                fit_result = FitResult.load(source)
                observed_fits[(engine, name)] = fit_result
                artifacts.extend(
                    self._export_fit(
                        fit_result,
                        study="uccle",
                        engine=engine,
                        name=name,
                        source=source,
                        figures=True,
                        diagnostics=diagnostics,
                    )
                )

        observed_selection = collect_selection_probabilities(observed_fits)
        observed_target = (
            self.paths.tables / "40_uccle" / "uccle_selection_probabilities.csv"
        )
        observed_target.parent.mkdir(parents=True, exist_ok=True)
        observed_selection.to_csv(observed_target, index=False)
        artifacts.append(observed_target)
        artifacts.extend(
            plot_uccle_selection_comparison(
                observed_selection,
                self.paths.figures / "40_uccle",
                **self._figure_options(),
            )
        )

        missing_unique = list(dict.fromkeys(missing))
        if strict and missing_unique:
            raise FileNotFoundError(
                "The strict report is missing combined fits: "
                + ", ".join(str(path) for path in missing_unique)
            )
        self._record(
            "report",
            status="complete" if not missing_unique else "partial",
            artifacts=artifacts,
            details={"missing": [str(path) for path in missing_unique]},
        )
        return artifacts

    # ------------------------------------------------------------------
    # Compact orchestration helpers used by examples and the CLI
    # ------------------------------------------------------------------
    def run(self, stage: str, **kwargs) -> list[Path]:
        """Run one named presentation stage."""

        key = str(stage).lower().replace("_", "-")
        if key == "data":
            return self.run_data(**kwargs)
        if key == "tail-simulations":
            return self.run_simulations(kind="tail", **kwargs)
        if key == "structural-simulations":
            return self.run_simulations(kind="structure", **kwargs)
        if key == "simulation-fit":
            return self.fit_simulations(**kwargs)
        if key == "uccle-fit":
            return self.fit_uccle(**kwargs)
        if key == "report":
            return self.report(**kwargs)
        raise ValueError(f"stage must be one of {PRESENTATION_STAGES}.")

    def run_all(
        self,
        *,
        overwrite: bool = False,
        diagnostics: bool = False,
    ) -> list[Path]:
        """Run the complete sequence locally (use the smoke profile first)."""

        artifacts = self.run_data(overwrite=overwrite, figures=True)
        artifacts.extend(
            self.run_simulations(kind="all", overwrite=overwrite, figures=True)
        )
        artifacts.extend(
            self.fit_simulations(
                engine="laplace", overwrite=overwrite, figures=False
            )
        )
        artifacts.extend(
            self.fit_simulations(engine="pgas", overwrite=overwrite, figures=False)
        )
        artifacts.extend(
            self.fit_uccle(engine="laplace", overwrite=overwrite, figures=False)
        )
        artifacts.extend(
            self.fit_uccle(engine="pgas", overwrite=overwrite, figures=False)
        )
        artifacts.extend(self.report(strict=True, diagnostics=diagnostics))
        return artifacts


__all__ = [
    "EXTREME_SERIES",
    "INFERENCE_ENGINES",
    "PRESENTATION_STAGES",
    "PresentationWorkflow",
    "presentation_gev_prior",
]
