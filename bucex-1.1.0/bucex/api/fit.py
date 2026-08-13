from __future__ import annotations

import copy
from dataclasses import asdict, replace
from typing import Any, Optional

import numpy as np

from ..components import DummySeasonal, LocalLinearTrend
from ..__about__ import __version__
from ..core.results import PosteriorBundle
from ..inference.fit.base import GibbsConfig
from ..inference.fit.centered_gaussian import CenteredGaussianGibbs
from ..inference.fit.centered_gev import CenteredGEVGibbs
from ..inference.fit.noncentered_gaussian import NonCenteredGaussianGibbs
from ..inference.fit.noncentered_gev import NonCenteredGEVGibbs
from ..inference.fit.priors import (
    NonCenteredGaussianPriors,
    NonCenteredGEVPriors,
    manuscript_gaussian_priors,
    manuscript_gev_priors,
    normal_gaussian_priors,
    normal_gev_priors,
    regularized_gaussian_priors,
    regularized_gev_priors,
    regularized_horseshoe_gaussian_priors,
    regularized_horseshoe_gev_priors,
    pc_gaussian_priors,
    pc_gev_priors,
    ssvs_gaussian_priors,
    ssvs_gev_priors,
)
from ..models.base import StateSpaceModel
from ..models.structural import StructuralModel
from ..observation.gaussian import GaussianObs
from ..observation.gev import GEVObs
from .._general.plan import InferencePlan


def _obs_name(model: StateSpaceModel) -> str | None:
    spec = getattr(getattr(model, "obs", None), "spec", None)
    return getattr(spec, "name", None) if spec is not None else None


def _normalize_fs_parameterization(parameterization: str) -> str:
    key = str(parameterization).lower().replace("-", "_")
    if key in {"noncentered", "noncentred", "ncp", "fs", "fruehwirth_schnatter"}:
        return "noncentered"
    if key in {"centered", "centred"}:
        return "centered"
    raise ValueError(
        "parameterization must be 'fruehwirth_schnatter' (alias 'noncentered') "
        "or 'centered'."
    )


def _combine_fs_chains(fits: list[PosteriorBundle], seeds: list[int]) -> PosteriorBundle:
    """Combine independent FS chains while retaining the legacy flat draws."""
    if not fits:
        raise ValueError("At least one fitted chain is required.")
    if len(fits) == 1:
        result = fits[0]
    else:
        result = copy.deepcopy(fits[0])
        keys = set(result.draws_static)
        if any(set(item.draws_static) != keys for item in fits[1:]):
            raise ValueError("FS chains produced incompatible static draw blocks.")
        result.draws_static = {
            key: np.concatenate([np.asarray(item.draws_static[key]) for item in fits], axis=0)
            for key in keys
        }
        result.draws_states = np.concatenate(
            [np.asarray(item.draws_states) for item in fits], axis=0
        )
        result.logpost = np.concatenate(
            [np.asarray(item.logpost) for item in fits], axis=0
        )
        acceptance_keys = set().union(*(item.acceptance for item in fits))
        result.acceptance = {
            key: float(np.mean([item.acceptance.get(key, np.nan) for item in fits]))
            for key in acceptance_keys
        }
        if all("draws_states_ncp" in item.meta for item in fits):
            result.meta["draws_states_ncp"] = np.concatenate(
                [np.asarray(item.meta["draws_states_ncp"]) for item in fits], axis=0
            )
        diagnostic_names = set.intersection(
            *[
                set(item.meta.get("engine_diagnostics", {}))
                for item in fits
            ]
        ) if fits else set()
        if diagnostic_names:
            result.meta["engine_diagnostics_by_chain"] = {
                name: np.stack(
                    [np.asarray(item.meta["engine_diagnostics"][name]) for item in fits]
                )
                for name in diagnostic_names
            }
            result.meta["engine_diagnostics"] = {
                name: np.concatenate(
                    [np.asarray(item.meta["engine_diagnostics"][name]) for item in fits]
                )
                for name in diagnostic_names
            }
    per_chain = [item.n_draws for item in fits]
    offsets = np.cumsum([0, *per_chain])
    result.meta.update(
        {
            "n_chains": len(fits),
            "draws_per_chain": per_chain[0] if len(set(per_chain)) == 1 else None,
            "chain_ids": np.concatenate(
                [np.full(count, index, dtype=int) for index, count in enumerate(per_chain)]
            ),
            "chain_slices": [
                (int(offsets[index]), int(offsets[index + 1]))
                for index in range(len(fits))
            ],
            "chain_seeds": [int(value) for value in seeds],
        }
    )
    return result


def combine_fs_fits(fits) -> PosteriorBundle:
    """Combine compatible independently saved single-chain FS fits."""
    items = list(fits)
    if not items:
        raise ValueError("At least one FS fit is required.")
    first = items[0]
    for index, item in enumerate(items, start=1):
        if item.n_chains != 1:
            raise ValueError(
                f"Fit {index} contains {item.n_chains} chains; combine_fs_fits "
                "expects independently saved single-chain fits."
            )
        if item.state_names != first.state_names:
            raise ValueError(f"Fit {index} has different state names.")
        if item.meta.get("prior_profile") != first.meta.get("prior_profile"):
            raise ValueError(f"Fit {index} has a different prior profile.")
        if item.meta.get("state_method") != first.meta.get("state_method"):
            raise ValueError(f"Fit {index} has a different state engine.")
        if not np.array_equal(np.asarray(item.y), np.asarray(first.y), equal_nan=True):
            raise ValueError(f"Fit {index} uses different observations.")
        if (item.dates is None) != (first.dates is None) or (
            item.dates is not None
            and not np.array_equal(np.asarray(item.dates), np.asarray(first.dates))
        ):
            raise ValueError(f"Fit {index} uses different dates.")
    seeds = [
        int(item.meta.get("chain_seeds", [index])[0])
        for index, item in enumerate(items)
    ]
    combined = _combine_fs_chains(items, seeds)
    combined.initial_values = copy.deepcopy(first.initial_values)
    combined.initial_values["parameters_by_chain"] = [
        values
        for item in items
        for values in item.initial_values.get("parameters_by_chain", [])
    ]
    combined.initial_values["indicators_by_chain"] = [
        values
        for item in items
        for values in item.initial_values.get("indicators_by_chain", [])
    ]
    combined.config = copy.deepcopy(first.config)
    combined.config["chains"] = len(items)
    combined.meta["combined_independent_runs"] = len(items)
    return combined


def _default_model(family: str, period: Optional[int]) -> StructuralModel:
    family = family.lower()
    if family not in {"gaussian", "gev"}:
        raise ValueError("family must be 'gaussian' or 'gev'.")
    obs = GaussianObs() if family == "gaussian" else GEVObs()
    components = [LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic")]
    if period is not None:
        components.append(DummySeasonal(period=int(period), mode="dynamic"))
    return StructuralModel(components=components, obs=obs)


def _seasonal_initial(y: np.ndarray, period: Optional[int]) -> np.ndarray:
    if period is None:
        return np.zeros(0, dtype=float)
    overall = float(np.nanmean(y))
    full = np.asarray(
        [np.nanmean(y[np.arange(y.size) % period == month]) - overall for month in range(period)],
        dtype=float,
    )
    full = np.nan_to_num(full, nan=0.0)
    full -= full.mean()
    return full[: period - 1]


def _default_initial_values(
    y_model: np.ndarray,
    model: StateSpaceModel,
    *,
    period: Optional[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = _obs_name(model)
    gamma0 = _seasonal_initial(y_model, period)
    if period is None:
        residual = y_model - float(np.mean(y_model))
    else:
        month = np.arange(y_model.size) % period
        full = np.r_[gamma0, -gamma0.sum()]
        residual = y_model - float(np.mean(y_model)) - full[month]
    scale = max(float(np.std(residual, ddof=1)), 0.25)

    params_state = {
        "alpha0": float(np.mean(y_model)),
        "beta0": 0.0,
        "gamma0_season": gamma0,
        "s_level": 0.02,
        "s_trend": 0.001,
        "s_season": 0.02,
        "q_level": 0.02**2,
        "q_trend": 0.001**2,
        "q_season": 0.02**2,
    }
    if family == "gaussian":
        params_obs = {"sigma": scale, "sigma2": scale * scale}
    elif family == "gev":
        params_obs = {"sigma": scale, "xi": -0.10}
    else:
        raise NotImplementedError(f"No automatic initialisation for obs='{family}'.")
    return params_state, params_obs


def fit_gaussian_structural(
    y: np.ndarray,
    model: StateSpaceModel,
    priors: Any,
    init_params_state: dict[str, Any],
    init_params_obs: dict[str, Any],
    *,
    exog: Optional[np.ndarray] = None,
    config: GibbsConfig | None = None,
    parameterization: str = "noncentered",
    state_method: str = "ffbs",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    cfg = config or GibbsConfig()
    parameterization = _normalize_fs_parameterization(parameterization)
    if parameterization == "centered":
        fitter = CenteredGaussianGibbs(model=model, priors=priors, config=cfg)
    elif parameterization == "noncentered":
        fitter = NonCenteredGaussianGibbs(model=model, priors=priors, config=cfg)
    else:
        raise ValueError("parameterization must be 'centered' or 'noncentered'.")
    return fitter.fit(
        y=y,
        init_params_state=init_params_state,
        init_params_obs=init_params_obs,
        exog=exog,
        state_method=state_method,
        state_kwargs=state_kwargs,
    )


def fit_gev_structural(
    y: np.ndarray,
    model: StateSpaceModel,
    priors: Any,
    init_params_state: dict[str, Any],
    init_params_obs: dict[str, Any],
    *,
    exog: Optional[np.ndarray] = None,
    config: GibbsConfig | None = None,
    parameterization: str = "noncentered",
    state_method: str = "laplace",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    cfg = config or GibbsConfig()
    kwargs = {} if state_kwargs is None else dict(state_kwargs)
    parameterization = _normalize_fs_parameterization(parameterization)
    if parameterization == "centered":
        fitter = CenteredGEVGibbs(model=model, priors=priors, config=cfg)
    elif parameterization == "noncentered":
        fitter = NonCenteredGEVGibbs(
            model=model,
            priors=priors,
            config=cfg,
            step_log_sigma=float(kwargs.pop("step_log_sigma", 0.05)),
            step_xi=float(kwargs.pop("step_xi", 0.05)),
        )
    else:
        raise ValueError("parameterization must be 'centered' or 'noncentered'.")
    return fitter.fit(
        y=y,
        init_params_state=init_params_state,
        init_params_obs=init_params_obs,
        exog=exog,
        state_method=state_method,
        state_kwargs=kwargs,
    )


def fit_bayes(
    y: np.ndarray,
    model: StateSpaceModel | str | None = None,
    priors: Any = "manuscript",
    init_params_state: Optional[dict[str, Any]] = None,
    init_params_obs: Optional[dict[str, Any]] = None,
    *,
    family: Optional[str] = None,
    period: Optional[int] = 12,
    dates: Optional[np.ndarray] = None,
    name: Optional[str] = None,
    transform_sign: float = 1.0,
    tail: Optional[str] = None,
    exog: Optional[np.ndarray] = None,
    config: GibbsConfig | None = None,
    n_iter: Optional[int] = None,
    burn: Optional[int] = None,
    thin: Optional[int] = None,
    seed: Optional[int] = None,
    progress: Optional[bool] = None,
    chains: int = 1,
    asis: Optional[bool] = None,
    method: str = "gibbs",
    state_method: str = "auto",
    parameterization: str = "noncentered",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    """Fit a Bayesian structural Gaussian or DGEV model.

    This is the v1.1 high-level FS API. A complete manuscript-style fit can be
    requested without manually constructing priors or initial values::

        fit = fit_bayes(y, family="gev", dates=dates, name="TXx")

    For block minima, pass ``tail='min'``; the function fits ``-y`` internally
    and all high-level plots and risk calculations are transformed back.
    """
    if str(method).lower() not in {"gibbs", "mcmc"}:
        raise NotImplementedError("fit_bayes exposes Gibbs / MH-within-Gibbs fitting.")
    if int(chains) < 1:
        raise ValueError("chains must be at least one.")

    if tail is not None:
        if tail.lower() in {"min", "minimum", "lower"}:
            transform_sign = -1.0
        elif tail.lower() in {"max", "maximum", "upper"}:
            transform_sign = 1.0
        else:
            raise ValueError("tail must be 'min' or 'max'.")
    if transform_sign not in {-1, 1, -1.0, 1.0}:
        raise ValueError("transform_sign must be +1 or -1.")

    if isinstance(model, str):
        family = model
        model = None
    if model is None:
        if family is None:
            raise ValueError("Provide model=... or family='gaussian'/'gev'.")
        model = _default_model(family, period)
    obs_name = _obs_name(model)
    if obs_name not in {"gaussian", "gev"}:
        raise NotImplementedError(f"No high-level fitter registered for obs='{obs_name}'.")

    y_original = np.asarray(y, dtype=float).reshape(-1)
    if not np.all(np.isfinite(y_original)):
        raise ValueError("y must contain only finite observations.")
    y_model = float(transform_sign) * y_original

    if config is None:
        config = GibbsConfig()
    overrides = {
        "n_iter": n_iter,
        "burn": burn,
        "thin": thin,
        "seed": seed,
        "progress": progress,
    }
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})
    if not (0 <= config.burn < config.n_iter):
        raise ValueError("Require 0 <= burn < n_iter.")
    if config.thin < 1:
        raise ValueError("thin must be >= 1.")

    prior_profile = "manuscript" if priors is None else priors
    built_in_priors = isinstance(prior_profile, str)
    if built_in_priors:
        profile = str(prior_profile).lower().replace("-", "_")
        builders = {
            ("gaussian", "manuscript"): manuscript_gaussian_priors,
            ("gev", "manuscript"): manuscript_gev_priors,
            ("gaussian", "normal"): normal_gaussian_priors,
            ("gev", "normal"): normal_gev_priors,
            ("gaussian", "regularized"): regularized_gaussian_priors,
            ("gev", "regularized"): regularized_gev_priors,
            ("gaussian", "regularized_lasso"): regularized_gaussian_priors,
            ("gev", "regularized_lasso"): regularized_gev_priors,
            ("gaussian", "horseshoe"): regularized_horseshoe_gaussian_priors,
            ("gev", "horseshoe"): regularized_horseshoe_gev_priors,
            ("gaussian", "regularized_horseshoe"): regularized_horseshoe_gaussian_priors,
            ("gev", "regularized_horseshoe"): regularized_horseshoe_gev_priors,
            ("gaussian", "pc"): pc_gaussian_priors,
            ("gev", "pc"): pc_gev_priors,
            ("gaussian", "ssvs"): ssvs_gaussian_priors,
            ("gev", "ssvs"): ssvs_gev_priors,
        }
        key = (obs_name, profile)
        if key not in builders:
            raise ValueError(
                "Built-in prior profiles are 'manuscript', 'regularized_lasso' "
                "(alias 'regularized'), 'horseshoe', 'pc', 'normal', and 'ssvs'."
            )
        priors = builders[key](1 if period is None else int(period))
        prior_profile = profile
    else:
        prior_profile = "custom"

    auto_state, auto_obs = _default_initial_values(y_model, model, period=period)
    if init_params_state is not None:
        auto_state.update(init_params_state)
    if init_params_obs is not None:
        auto_obs.update(init_params_obs)

    for component in model.components:
        if isinstance(component, LocalLinearTrend):
            component.initial_level = float(auto_state["alpha0"])
            if component.trend_mode == "dynamic":
                component.initial_slope = float(auto_state["beta0"])
        elif isinstance(component, DummySeasonal) and component.mode == "dynamic":
            component.initial_mean = tuple(
                np.asarray(auto_state["gamma0_season"], dtype=float)
            )

    resolved_state_method = (
        "ffbs" if obs_name == "gaussian" else "laplace"
    ) if state_method == "auto" else state_method
    normalized_parameterization = _normalize_fs_parameterization(parameterization)
    resolved_state_kwargs = {} if state_kwargs is None else dict(state_kwargs)
    if asis is not None:
        resolved_state_kwargs["asis"] = bool(asis)
    resolved_state_kwargs.setdefault("asis", False)

    if int(chains) == 1 and config.seed is not None:
        chain_seeds = [int(config.seed)]
    else:
        seed_sequences = np.random.SeedSequence(config.seed).spawn(int(chains))
        chain_seeds = [int(sequence.generate_state(1)[0]) for sequence in seed_sequences]
    chain_fits: list[PosteriorBundle] = []
    for chain_index, chain_seed in enumerate(chain_seeds):
        chain_config = replace(config, seed=chain_seed)
        if config.progress and int(chains) > 1:
            print(f"[chain {chain_index + 1}/{chains}] seed={chain_seed}", flush=True)
        if obs_name == "gaussian":
            if (
                not isinstance(priors, NonCenteredGaussianPriors)
                and normalized_parameterization == "noncentered"
            ):
                raise TypeError("An FS Gaussian fit requires NonCenteredGaussianPriors.")
            chain_fit = fit_gaussian_structural(
                y_model,
                model,
                priors,
                auto_state,
                auto_obs,
                exog=exog,
                config=chain_config,
                parameterization=normalized_parameterization,
                state_method=resolved_state_method,
                state_kwargs=resolved_state_kwargs,
            )
        else:
            if (
                not isinstance(priors, NonCenteredGEVPriors)
                and normalized_parameterization == "noncentered"
            ):
                raise TypeError("An FS GEV fit requires NonCenteredGEVPriors.")
            chain_fit = fit_gev_structural(
                y_model,
                model,
                priors,
                auto_state,
                auto_obs,
                exog=exog,
                config=chain_config,
                parameterization=normalized_parameterization,
                state_method=resolved_state_method,
                state_kwargs=resolved_state_kwargs,
            )
        chain_fits.append(chain_fit)
    fit = _combine_fs_chains(chain_fits, chain_seeds)

    fit.y = y_original
    fit.dates = None if dates is None else np.asarray(dates)
    if fit.dates is not None and len(fit.dates) != len(y_original):
        raise ValueError("dates must have the same length as y.")
    fit.model = model
    fit.state_names = tuple(model.state_names)
    fit.series_name = name
    fit.transform_sign = float(transform_sign)
    fit.priors = priors
    fit.config = {
        "gibbs": asdict(config),
        "chains": int(chains),
        "state_method": resolved_state_method,
        "parameterization": (
            "fruehwirth_schnatter"
            if normalized_parameterization == "noncentered"
            else "centered"
        ),
        "state_kwargs": copy.deepcopy(resolved_state_kwargs),
    }
    fit.initial_values = {
        "params_state": copy.deepcopy(auto_state),
        "params_obs": copy.deepcopy(auto_obs),
        "parameters_by_chain": [
            {
                "sd.level": abs(float(auto_state.get("s_level", 0.0))),
                **(
                    {"sd.slope": abs(float(auto_state.get("s_trend", 0.0)))}
                    if "beta" in model.state_names
                    else {}
                ),
                **(
                    {"sd.seasonal": abs(float(auto_state.get("s_season", 0.0)))}
                    if any(name.startswith("g") for name in model.state_names)
                    else {}
                ),
                "sigma": float(auto_obs["sigma"]),
                **({"xi": float(auto_obs["xi"])} if "xi" in auto_obs else {}),
            }
            for _ in range(int(chains))
        ],
        "indicators_by_chain": [{} for _ in range(int(chains))],
    }
    fit.exog = None if exog is None else np.asarray(exog).copy()
    exact_target = obs_name == "gaussian" or resolved_state_method == "pgas"
    fit.plan = InferencePlan(
        family=obs_name,
        engine=resolved_state_method,
        parameterization=(
            "fruehwirth_schnatter"
            if normalized_parameterization == "noncentered"
            else "centered"
        ),
        asis=bool(resolved_state_kwargs["asis"]),
        state_update=(
            "exact Gaussian FS FFBS"
            if obs_name == "gaussian"
            else (
                "exact-invariant FS PGAS"
                if resolved_state_method == "pgas"
                else "iterated FS Laplace approximation"
            )
        ),
        targets_exact_posterior=exact_target,
        approximation=(
            None
            if exact_target
            else "Iterated Laplace approximation to the GEV state block"
        ),
        warnings=(),
    )
    fit.meta.update(
        {
            "period": None if period is None else int(period),
            "family": obs_name,
            "series_name": name,
            "transform_sign": float(transform_sign),
            "prior_profile": str(prior_profile),
            "bucex_version": __version__,
            "inference_contract": (
                "fruehwirth_schnatter_augmented_noncentering"
                if normalized_parameterization == "noncentered"
                else "centered"
            ),
            "asis": bool(resolved_state_kwargs["asis"]),
        }
    )
    return fit
