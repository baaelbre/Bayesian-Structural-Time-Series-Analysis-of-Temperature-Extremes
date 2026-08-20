"""Reproducible simulation and Uccle presentation workflow."""
from __future__ import annotations

from .config import (
    PROFILE_SETTINGS,
    PROFILE_WINDOWS,
    PresentationConfig,
    RuntimeSettings,
    WorkflowPaths,
)
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
    scenario_catalog,
    simulate_scenario,
)
from .uccle import (
    EXTREME_SERIES,
    INFERENCE_ENGINES,
    PRESENTATION_STAGES,
    PresentationWorkflow,
    presentation_gev_prior,
)

__all__ = [
    "ALL_SCENARIOS",
    "EXTREME_SERIES",
    "GEVScenario",
    "INFERENCE_ENGINES",
    "PRESENTATION_STAGES",
    "PROFILE_SETTINGS",
    "PROFILE_WINDOWS",
    "PresentationConfig",
    "PresentationWorkflow",
    "RuntimeSettings",
    "SCALE_SCENARIOS",
    "STRUCTURAL_SCENARIOS",
    "TAIL_SCENARIOS",
    "WorkflowPaths",
    "collect_selection_probabilities",
    "export_fit_results",
    "plot_fit_results",
    "plot_scale_simulations",
    "plot_selection_recovery",
    "plot_structural_simulations",
    "plot_tail_simulations",
    "plot_uccle_record_figures",
    "plot_uccle_selection_comparison",
    "presentation_gev_prior",
    "scenario_by_name",
    "scenario_catalog",
    "simulate_scenario",
]
