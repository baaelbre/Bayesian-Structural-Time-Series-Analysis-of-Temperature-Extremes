"""Resolve one public prior choice for the selected parameterization."""
from __future__ import annotations

from typing import Any

from ..models.compiler import CompiledModel
from .process import Priors, resolve_priors as resolve_process_priors
from .structural import (
    FSGaussianPriors,
    FSGEVPriors,
    manuscript_gaussian_priors,
    manuscript_gev_priors,
    normal_gaussian_priors,
    normal_gev_priors,
    pc_gaussian_priors,
    pc_gev_priors,
    regularized_gaussian_priors,
    regularized_gev_priors,
    regularized_horseshoe_gaussian_priors,
    regularized_horseshoe_gev_priors,
    ssvs_gaussian_priors,
    ssvs_gev_priors,
)


STRUCTURAL_PRIORS = {
    "manuscript_lasso",
    "regularized_lasso",
    "regularized_horseshoe",
    "pc",
    "normal",
    "ssvs",
}


def normalize_prior_profile(value: str | None) -> str:
    key = "regularized_horseshoe" if value is None else str(value).lower().replace("-", "_")
    aliases = {
        "manuscript": "manuscript_lasso",
        "bayesian_lasso": "manuscript_lasso",
        "lasso": "regularized_lasso",
        "regularized": "regularized_lasso",
        "horseshoe": "regularized_horseshoe",
        "half_normal": "normal",
        "spike_slab": "ssvs",
    }
    return aliases.get(key, key)


def resolve_structural_priors(
    compiled: CompiledModel,
    priors: Any,
):
    family = compiled.family
    expected = FSGaussianPriors if family == "gaussian" else FSGEVPriors
    if isinstance(priors, expected):
        return priors
    if not isinstance(priors, (str, type(None))):
        raise TypeError(
            "Fruehwirth-Schnatter fits require a structural prior profile or "
            f"{expected.__name__}."
        )
    profile = normalize_prior_profile(priors)
    builders = {
        ("gaussian", "manuscript_lasso"): manuscript_gaussian_priors,
        ("gev", "manuscript_lasso"): manuscript_gev_priors,
        ("gaussian", "regularized_lasso"): regularized_gaussian_priors,
        ("gev", "regularized_lasso"): regularized_gev_priors,
        ("gaussian", "regularized_horseshoe"): regularized_horseshoe_gaussian_priors,
        ("gev", "regularized_horseshoe"): regularized_horseshoe_gev_priors,
        ("gaussian", "pc"): pc_gaussian_priors,
        ("gev", "pc"): pc_gev_priors,
        ("gaussian", "normal"): normal_gaussian_priors,
        ("gev", "normal"): normal_gev_priors,
        ("gaussian", "ssvs"): ssvs_gaussian_priors,
        ("gev", "ssvs"): ssvs_gev_priors,
    }
    try:
        builder = builders[(family, profile)]
    except KeyError as exc:
        raise ValueError(
            "FS prior must be manuscript_lasso, regularized_lasso, "
            "regularized_horseshoe, pc, normal, or ssvs."
        ) from exc
    return builder(period=int(compiled.model.period or 1))


def resolve_prior_spec(
    compiled: CompiledModel,
    *,
    parameterization: str,
    priors: Any,
):
    """Return the kernel prior object without changing its statistical meaning."""

    if parameterization == "fruehwirth_schnatter":
        return resolve_structural_priors(compiled, priors)
    if isinstance(priors, (FSGaussianPriors, FSGEVPriors)):
        raise ValueError(
            "Hierarchical signed-scale priors require "
            "parameterization='fruehwirth_schnatter'."
        )
    profile = priors
    if isinstance(priors, str) or priors is None:
        # The FS default is the regularized horseshoe because its signed
        # innovation scales are regression coefficients.  Centered and
        # disturbance strategies instead need a prior directly on positive
        # process SDs; use the calibrated PC/exponential profile there.
        normalized = "pc" if priors is None else normalize_prior_profile(priors)
        mapping = {
            "pc": "regularized",
            "normal": "half_normal",
            "ssvs": "spike_slab",
        }
        if normalized in {"manuscript_lasso", "regularized_lasso", "regularized_horseshoe"}:
            raise ValueError(
                f"Prior '{normalized}' is a signed-scale hierarchy and requires "
                "parameterization='fruehwirth_schnatter'."
            )
        profile = mapping.get(normalized, normalized)
    return resolve_process_priors(compiled, profile)
