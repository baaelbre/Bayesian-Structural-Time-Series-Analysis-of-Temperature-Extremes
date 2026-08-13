"""Stable public fitting API.

``fit`` is the flexible compiled-model interface introduced in 1.0.
``fit_bayes`` deliberately stays close to the 0.3 research code and uses the
Fruehwirth--Schnatter sampler.  The family-specific wrappers dispatch by model
type, so existing explicit 0.3 calls and the concise 1.x calls coexist.
"""

from __future__ import annotations

from .._general.api import (
    fit as _general_fit,
    fit_bulk_tail,
    fit_gaussian_structural as _general_fit_gaussian,
    fit_gev_structural as _general_fit_gev,
    make_gaussian_model as _general_make_gaussian,
    make_gev_model as _general_make_gev,
    plan,
)
from ..components.base import Component as LegacyComponent
from ..models.base import StateSpaceModel
from .fit import (
    combine_fs_fits,
    fit_bayes,
    fit_gaussian_structural as _fs_fit_gaussian,
    fit_gev_structural as _fs_fit_gev,
)
from .model_builders import (
    make_gaussian_model as _legacy_make_gaussian,
    make_gev_model as _legacy_make_gev,
)

# Importing the legacy ``api.fit`` submodule assigns that module to the package
# attribute ``fit``.  Rebind the intended public callable after the imports.
fit = _general_fit


def _legacy_model_argument(args, kwargs) -> bool:
    candidate = args[0] if args else kwargs.get("model")
    return isinstance(candidate, StateSpaceModel)


def fit_gaussian_structural(y, *args, **kwargs):
    if _legacy_model_argument(args, kwargs):
        return _fs_fit_gaussian(y, *args, **kwargs)
    return _general_fit_gaussian(y, *args, **kwargs)


def fit_gev_structural(y, *args, **kwargs):
    if _legacy_model_argument(args, kwargs):
        return _fs_fit_gev(y, *args, **kwargs)
    return _general_fit_gev(y, *args, **kwargs)


def make_gaussian_model(components, **kwargs):
    values = list(components)
    if values and isinstance(values[0], LegacyComponent):
        if kwargs:
            raise TypeError("The legacy model builder does not accept keyword options.")
        return _legacy_make_gaussian(values)
    return _general_make_gaussian(values, **kwargs)


def make_gev_model(components, **kwargs):
    values = list(components)
    if values and isinstance(values[0], LegacyComponent):
        if kwargs:
            raise TypeError("The legacy model builder does not accept keyword options.")
        return _legacy_make_gev(values)
    return _general_make_gev(values, **kwargs)

__all__ = [
    "fit_bayes",
    "fit",
    "fit_bulk_tail",
    "plan",
    "combine_fs_fits",
    "fit_gaussian_structural",
    "fit_gev_structural",
    "make_gaussian_model",
    "make_gev_model",
]
