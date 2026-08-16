"""Regularized-horseshoe utilities for namespaced factor innovations."""
from __future__ import annotations

from math import lgamma
from typing import Any, Mapping

import numpy as np

from ...priors.factor import FactorPriors
from .fs_utils import _slice_sample_real


def initialise_factor_horseshoe(priors: FactorPriors) -> dict[str, Any]:
    hp = priors.horseshoe
    if hp is None:
        return {}
    return {
        "local": {
            name: float(hp.initial_local) for name in priors.horseshoe_processes
        },
        "global": float(hp.initial_global_value()),
        "slab2": float(hp.initial_slab2_value()),
    }


def copy_factor_horseshoe(state: Mapping[str, Any]) -> dict[str, Any]:
    if not state:
        return {}
    return {
        "local": {name: float(value) for name, value in state["local"].items()},
        "global": float(state["global"]),
        "slab2": float(state["slab2"]),
    }


def factor_horseshoe_variance(
    priors: FactorPriors,
    state: Mapping[str, Any],
    process: str,
) -> float:
    hp = priors.horseshoe
    if hp is None or process not in priors.horseshoe_processes:
        raise ValueError(f"No factor regularized-horseshoe prior for '{process}'.")
    return float(
        hp.conditional_variance(
            process,
            local=float(state["local"][process]),
            global_scale=float(state["global"]),
            slab2=float(state["slab2"]),
        )
    )


def _normal_zero_logpdf(value: float, variance: float) -> float:
    variance = max(float(variance), 1e-300)
    return float(
        -0.5 * (np.log(2.0 * np.pi * variance) + float(value) ** 2 / variance)
    )


def _half_cauchy_logpdf(value: float, scale: float) -> float:
    value = float(value)
    scale = float(scale)
    if value <= 0.0 or scale <= 0.0:
        return -np.inf
    return float(np.log(2.0 / (np.pi * scale)) - np.log1p((value / scale) ** 2))


def _inverse_gamma_logpdf(value: float, shape: float, scale: float) -> float:
    value = float(value)
    if value <= 0.0:
        return -np.inf
    return float(
        shape * np.log(scale)
        - lgamma(shape)
        - (shape + 1.0) * np.log(value)
        - scale / value
    )


def factor_horseshoe_coefficient_logpdf(
    value: float,
    process: str,
    priors: FactorPriors,
    state: Mapping[str, Any],
) -> float:
    """Conditional Gaussian density for signed or absolute FS coefficients.

    For a positive disturbance SD this differs from its half-Normal density by
    the constant ``log(2)``, which cancels from every scale/hyperparameter MH
    ratio.
    """

    return _normal_zero_logpdf(
        float(value), factor_horseshoe_variance(priors, state, process)
    )


def factor_horseshoe_logpdf(
    coefficients: Mapping[str, float],
    priors: FactorPriors,
    state: Mapping[str, Any],
) -> float:
    hp = priors.horseshoe
    if hp is None:
        return 0.0
    total = sum(
        factor_horseshoe_coefficient_logpdf(
            coefficients[name], name, priors, state
        )
        for name in priors.horseshoe_processes
    )
    total += sum(
        _half_cauchy_logpdf(value, 1.0)
        for value in state["local"].values()
    )
    total += _half_cauchy_logpdf(state["global"], hp.global_scale)
    total += _inverse_gamma_logpdf(
        state["slab2"],
        0.5 * hp.slab_df,
        0.5 * hp.slab_df * hp.slab_scale**2,
    )
    return float(total)


def update_factor_horseshoe(
    coefficients: Mapping[str, float],
    state: Mapping[str, Any],
    priors: FactorPriors,
    rng: np.random.Generator,
    *,
    steps: Mapping[str, float] | None = None,
    step_local: float = 0.35,
    step_global: float = 0.25,
    step_slab: float = 0.20,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Exact stepping-out slice updates for the factor horseshoe hierarchy."""

    hp = priors.horseshoe
    if hp is None:
        return copy_factor_horseshoe(state), {}
    out = copy_factor_horseshoe(state)
    moved: dict[str, bool] = {}
    proposal_steps = {} if steps is None else steps

    def coefficient_density(
        candidate: Mapping[str, Any], subset: tuple[str, ...] | None = None
    ) -> float:
        selected = priors.horseshoe_processes if subset is None else subset
        return float(
            sum(
                _normal_zero_logpdf(
                    coefficients[name],
                    hp.conditional_variance(
                        name,
                        local=candidate["local"][name],
                        global_scale=candidate["global"],
                        slab2=candidate["slab2"],
                    ),
                )
                for name in selected
            )
        )

    for name in priors.horseshoe_processes:
        step = float(
            proposal_steps.get(f"horseshoe.local.{name}", step_local)
        )
        current = float(out["local"][name])
        current_log = np.log(current)

        def target(log_value):
            candidate = copy_factor_horseshoe(out)
            value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
            candidate["local"][name] = value
            return (
                coefficient_density(candidate, (name,))
                + _half_cauchy_logpdf(value, 1.0)
                + log_value
            )

        proposed_log, _ = _slice_sample_real(
            current_log, target, rng, width=step
        )
        out["local"][name] = float(np.exp(proposed_log))
        moved[f"horseshoe.local.{name}"] = not np.isclose(
            proposed_log, current_log
        )

    current = float(out["global"])
    current_log = np.log(current)

    def global_target(log_value):
        candidate = copy_factor_horseshoe(out)
        value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
        candidate["global"] = value
        return (
            coefficient_density(candidate)
            + _half_cauchy_logpdf(value, hp.global_scale)
            + log_value
        )

    proposed_log, _ = _slice_sample_real(
        current_log,
        global_target,
        rng,
        width=float(proposal_steps.get("horseshoe.global", step_global)),
    )
    out["global"] = float(np.exp(proposed_log))
    moved["horseshoe.global"] = not np.isclose(proposed_log, current_log)

    current = float(out["slab2"])
    current_log = np.log(current)
    shape = 0.5 * hp.slab_df
    scale = 0.5 * hp.slab_df * hp.slab_scale**2
    def slab_target(log_value):
        candidate = copy_factor_horseshoe(out)
        value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
        candidate["slab2"] = value
        return (
            coefficient_density(candidate)
            + _inverse_gamma_logpdf(value, shape, scale)
            + log_value
        )

    proposed_log, _ = _slice_sample_real(
        current_log,
        slab_target,
        rng,
        width=float(proposal_steps.get("horseshoe.slab2", step_slab)),
    )
    out["slab2"] = float(np.exp(proposed_log))
    moved["horseshoe.slab2"] = not np.isclose(proposed_log, current_log)
    return out, moved


__all__ = [
    "copy_factor_horseshoe",
    "factor_horseshoe_coefficient_logpdf",
    "factor_horseshoe_logpdf",
    "factor_horseshoe_variance",
    "initialise_factor_horseshoe",
    "update_factor_horseshoe",
]
