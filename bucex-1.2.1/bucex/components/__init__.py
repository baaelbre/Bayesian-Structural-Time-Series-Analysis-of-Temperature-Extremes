"""Canonical structural components."""
from .base import Component, ComponentSpec
from .trend import LocalLevel, LocalLinearTrend
from .seasonal import DummySeasonal
from .regression import Regression, RegressionComponent

Seasonal = DummySeasonal


def component_from_dict(value):
    kind = str(value["type"])
    args = {key: item for key, item in value.items() if key != "type"}
    for name in ("feature_names", "initial_mean"):
        if name in args and args[name] is not None:
            args[name] = tuple(args[name])
    if kind == "local_level":
        return LocalLevel(**args)
    if kind == "local_linear_trend":
        return LocalLinearTrend(**args)
    if kind == "dummy_seasonal":
        return DummySeasonal(**args)
    if kind == "regression":
        return Regression(**args)
    raise ValueError(f"Unknown component type '{kind}'.")


__all__ = [
    "Component",
    "ComponentSpec",
    "LocalLevel",
    "LocalLinearTrend",
    "DummySeasonal",
    "Seasonal",
    "Regression",
    "RegressionComponent",
    "component_from_dict",
]
