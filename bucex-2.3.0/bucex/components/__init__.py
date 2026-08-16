"""Canonical structural components."""
from .base import Component, ComponentSpec
from .trend import LocalLevel, LocalLinearTrend
from .seasonal import DummySeasonal
from .regression import Regression, RegressionComponent

Seasonal = DummySeasonal


def component_from_dict(value):
    kind = str(value["type"])
    args = {key: item for key, item in value.items() if key != "type"}
    if "feature_names" in args and args["feature_names"] is not None:
        args["feature_names"] = tuple(args["feature_names"])
    if kind == "dummy_seasonal" and args.get("initial_mean") is not None:
        args["initial_mean"] = tuple(args["initial_mean"])
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
