"""Triple-gamma utilities for namespaced factor innovations."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...priors.factor import FactorPriors
from .fs_utils import (
    _beta_logpdf,
    _beta_prime_logpdf,
    _gamma_logpdf,
    _inverse_gamma_logpdf,
    _normal_zero_logpdf,
    _slice_sample_real,
)


def initialise_factor_triple_gamma(priors: FactorPriors) -> dict[str, Any]:
    prior = priors.triple_gamma
    if prior is None:
        return {}
    return {
        "numerator": {
            name: float(prior.initial_numerator)
            for name in priors.triple_gamma_processes
        },
        "denominator": {
            name: float(prior.initial_denominator)
            for name in priors.triple_gamma_processes
        },
        "global": float(prior.global_scale),
        "a": float(prior.spike_shape),
        "c": float(prior.tail_shape),
        "slab2": (
            float(prior.initial_slab2_value()) if prior.regularized else None
        ),
    }


def copy_factor_triple_gamma(state: Mapping[str, Any]) -> dict[str, Any]:
    if not state:
        return {}
    return {
        "numerator": {
            name: float(value) for name, value in state["numerator"].items()
        },
        "denominator": {
            name: float(value) for name, value in state["denominator"].items()
        },
        "global": float(state["global"]),
        "a": float(state["a"]),
        "c": float(state["c"]),
        "slab2": (
            None if state.get("slab2") is None else float(state["slab2"])
        ),
    }


def factor_triple_gamma_variance(
    priors: FactorPriors,
    state: Mapping[str, Any],
    process: str,
) -> float:
    prior = priors.triple_gamma
    if prior is None or process not in priors.triple_gamma_processes:
        raise ValueError(f"No factor triple-gamma prior for '{process}'.")
    return float(
        prior.conditional_variance(
            process,
            numerator=state["numerator"][process],
            denominator=state["denominator"][process],
            global_scale=state["global"],
            slab2=state.get("slab2"),
        )
    )


def factor_triple_gamma_coefficient_logpdf(
    value: float,
    process: str,
    priors: FactorPriors,
    state: Mapping[str, Any],
) -> float:
    return _normal_zero_logpdf(
        float(value), factor_triple_gamma_variance(priors, state, process)
    )


def factor_triple_gamma_logpdf(
    coefficients: Mapping[str, float],
    priors: FactorPriors,
    state: Mapping[str, Any],
) -> float:
    prior = priors.triple_gamma
    if prior is None:
        return 0.0
    total = sum(
        factor_triple_gamma_coefficient_logpdf(
            coefficients[name], name, priors, state
        )
        for name in priors.triple_gamma_processes
    )
    total += sum(
        _gamma_logpdf(value, state["a"], 1.0)
        for value in state["numerator"].values()
    )
    total += sum(
        _gamma_logpdf(value, state["c"], 1.0)
        for value in state["denominator"].values()
    )
    if prior.learn_global:
        total += _beta_prime_logpdf(
            state["global"], state["c"], state["a"]
        )
    if prior.learn_shapes:
        total += _beta_logpdf(
            2.0 * state["a"], *prior.spike_shape_prior
        )
        total += _beta_logpdf(
            2.0 * state["c"], *prior.tail_shape_prior
        )
    if prior.regularized:
        total += _inverse_gamma_logpdf(
            state["slab2"],
            0.5 * prior.slab_df,
            0.5 * prior.slab_df * prior.slab_scale**2,
        )
    return float(total)


def update_factor_triple_gamma(
    coefficients: Mapping[str, float],
    state: Mapping[str, Any],
    priors: FactorPriors,
    rng: np.random.Generator,
    *,
    widths: Mapping[str, float] | None = None,
    width_local: float = 1.0,
    width_global: float = 1.0,
    width_shape: float = 0.8,
    width_slab: float = 0.8,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Exact slice updates for the factor triple-gamma hierarchy."""

    prior = priors.triple_gamma
    if prior is None:
        return copy_factor_triple_gamma(state), {}
    out = copy_factor_triple_gamma(state)
    names = priors.triple_gamma_processes
    widths = {} if widths is None else widths
    moved: dict[str, bool] = {}

    def coefficient_density(candidate, selected=None):
        selected = names if selected is None else selected
        return float(
            sum(
                _normal_zero_logpdf(
                    coefficients[name],
                    prior.conditional_variance(
                        name,
                        numerator=candidate["numerator"][name],
                        denominator=candidate["denominator"][name],
                        global_scale=candidate["global"],
                        slab2=candidate.get("slab2"),
                    ),
                )
                for name in selected
            )
        )

    for name in names:
        for field_name, shape_name in (
            ("numerator", "a"),
            ("denominator", "c"),
        ):
            label = f"triple_gamma.{field_name}.{name}"
            current_log = np.log(out[field_name][name])

            def target(log_value, *, field_name=field_name, shape_name=shape_name):
                candidate = copy_factor_triple_gamma(out)
                value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
                candidate[field_name][name] = value
                return (
                    coefficient_density(candidate, (name,))
                    + _gamma_logpdf(value, candidate[shape_name], 1.0)
                    + log_value
                )

            proposed_log, _ = _slice_sample_real(
                current_log,
                target,
                rng,
                width=float(widths.get(label, width_local)),
            )
            out[field_name][name] = float(np.exp(proposed_log))
            moved[label] = not np.isclose(proposed_log, current_log)

    if prior.learn_global:
        current_log = np.log(out["global"])

        def global_target(log_value):
            candidate = copy_factor_triple_gamma(out)
            value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
            candidate["global"] = value
            return (
                coefficient_density(candidate)
                + _beta_prime_logpdf(value, candidate["c"], candidate["a"])
                + log_value
            )

        proposed_log, _ = _slice_sample_real(
            current_log,
            global_target,
            rng,
            width=float(widths.get("triple_gamma.global", width_global)),
        )
        out["global"] = float(np.exp(proposed_log))
        moved["triple_gamma.global"] = not np.isclose(
            proposed_log, current_log
        )

    if prior.learn_shapes:
        for shape_name, hyperprior, field_name in (
            ("a", prior.spike_shape_prior, "numerator"),
            ("c", prior.tail_shape_prior, "denominator"),
        ):
            current = out[shape_name]
            current_logit = np.log(current / (0.5 - current))

            def shape_target(logit_value, *, shape_name=shape_name,
                             hyperprior=hyperprior, field_name=field_name):
                probability = 1.0 / (
                    1.0 + np.exp(-np.clip(logit_value, -700.0, 700.0))
                )
                shape = 0.5 * probability
                candidate = copy_factor_triple_gamma(out)
                candidate[shape_name] = shape
                total = sum(
                    _gamma_logpdf(candidate[field_name][name], shape, 1.0)
                    for name in names
                )
                if prior.learn_global:
                    total += _beta_prime_logpdf(
                        candidate["global"], candidate["c"], candidate["a"]
                    )
                total += _beta_logpdf(probability, *hyperprior)
                total += np.log(max(probability * (1.0 - probability), 1e-300))
                return float(total)

            proposed_logit, _ = _slice_sample_real(
                current_logit,
                shape_target,
                rng,
                width=float(
                    widths.get(f"triple_gamma.{shape_name}", width_shape)
                ),
            )
            out[shape_name] = float(
                0.5 / (1.0 + np.exp(-proposed_logit))
            )
            moved[f"triple_gamma.{shape_name}"] = not np.isclose(
                proposed_logit, current_logit
            )

    if prior.regularized:
        current_log = np.log(out["slab2"])
        shape = 0.5 * prior.slab_df
        scale = 0.5 * prior.slab_df * prior.slab_scale**2

        def slab_target(log_value):
            candidate = copy_factor_triple_gamma(out)
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
            width=float(widths.get("triple_gamma.slab2", width_slab)),
        )
        out["slab2"] = float(np.exp(proposed_log))
        moved["triple_gamma.slab2"] = not np.isclose(
            proposed_log, current_log
        )

    return out, moved


__all__ = [
    "copy_factor_triple_gamma",
    "factor_triple_gamma_coefficient_logpdf",
    "factor_triple_gamma_logpdf",
    "factor_triple_gamma_variance",
    "initialise_factor_triple_gamma",
    "update_factor_triple_gamma",
]
