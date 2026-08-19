"""Reproducible analysis workflows shipped with :mod:`bucex`.

The presentation workflow is intentionally a thin orchestration layer.  All
inference still goes through :func:`bucex.fit`; the workflow only fixes the
scientific sequence, names artifacts consistently, and makes local and HPC
runs interchangeable.
"""
from __future__ import annotations

from .config import PresentationConfig, RuntimeSettings, WorkflowPaths
from .results import export_fit_results, plot_fit_results
from .uccle import (
    BENCHMARK_MODELS,
    PRESENTATION_STAGES,
    VALIDATION_MODELS,
    PresentationWorkflow,
    componentwise_hierarchical_prior,
    componentwise_univariate_prior,
    txx_benchmark_prior,
)

__all__ = [
    "BENCHMARK_MODELS",
    "PRESENTATION_STAGES",
    "PresentationConfig",
    "PresentationWorkflow",
    "RuntimeSettings",
    "VALIDATION_MODELS",
    "WorkflowPaths",
    "componentwise_hierarchical_prior",
    "componentwise_univariate_prior",
    "export_fit_results",
    "plot_fit_results",
    "txx_benchmark_prior",
]
