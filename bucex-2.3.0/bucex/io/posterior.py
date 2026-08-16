"""Safe, checksummed serialization for the single :class:`FitResult` type."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from ..__about__ import __version__
from ..core.fit import FitResult
from ..inference.plan import InferencePlan
from ..models.compiler import compile_model
from ..models.factor import FactorModel
from ..models.multiseries import MultiSeriesModel
from ..models.structural import Model
from ..priors import factor as factor_priors
from ..priors import hierarchical as hierarchical_priors
from ..priors import process as process_priors
from ..priors import structural as structural_priors


FORMAT = "bucex-fit"
SCHEMA_VERSION = "2.3"
SUPPORTED_SCHEMA_VERSIONS = {"1.2", "2.0", "2.1", "2.3"}


def _prior_classes() -> dict[str, type]:
    classes: list[type] = []
    for module in (
        process_priors,
        structural_priors,
        factor_priors,
        hierarchical_priors,
    ):
        for name in dir(module):
            candidate = getattr(module, name)
            if isinstance(candidate, type) and is_dataclass(candidate):
                classes.append(candidate)
    return {
        f"{candidate.__module__}:{candidate.__qualname__}": candidate
        for candidate in classes
    }


_PRIOR_CLASSES = _prior_classes()


def _encode(value: Any, arrays: dict[str, np.ndarray], prefix: str) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            array = array.astype(str)
        key = f"array_{len(arrays):06d}_{prefix.replace('.', '_')}"
        arrays[key] = array
        return {"__array__": key}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays, prefix)
    if isinstance(value, float) and not np.isfinite(value):
        label = "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
        return {"__float__": label}
    if is_dataclass(value) and not isinstance(value, type):
        tag = f"{value.__class__.__module__}:{value.__class__.__qualname__}"
        if tag not in _PRIOR_CLASSES:
            raise TypeError(f"Unsupported dataclass in a fit archive: {tag}")
        payload = {
            item.name: _encode(getattr(value, item.name), arrays, f"{prefix}_{item.name}")
            for item in fields(value)
            if item.init
        }
        return {"__dataclass__": tag, "fields": payload}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item, arrays, prefix) for item in value]}
    if isinstance(value, list):
        return [_encode(item, arrays, prefix) for item in value]
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {
                key: _encode(item, arrays, f"{prefix}_{key}")
                for key, item in value.items()
            }
        return {
            "__mapping__": [
                [_encode(key, arrays, prefix), _encode(item, arrays, prefix)]
                for key, item in value.items()
            ]
        }
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported value in a fit archive: {type(value).__name__}")


def _decode(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    if not isinstance(value, dict):
        return value
    if "__array__" in value:
        key = value["__array__"]
        if key not in arrays:
            raise ValueError(f"Fit archive references a missing array: {key}")
        return np.asarray(arrays[key])
    if "__float__" in value:
        return {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[value["__float__"]]
    if "__tuple__" in value:
        return tuple(_decode(item, arrays) for item in value["__tuple__"])
    if "__mapping__" in value:
        return {
            _decode(pair[0], arrays): _decode(pair[1], arrays)
            for pair in value["__mapping__"]
        }
    if "__path__" in value:
        return Path(value["__path__"])
    if "__dataclass__" in value:
        tag = value["__dataclass__"]
        if tag not in _PRIOR_CLASSES:
            raise ValueError(f"Fit archive contains an unapproved class: {tag}")
        kwargs = {
            key: _decode(item, arrays)
            for key, item in value.get("fields", {}).items()
        }
        return _PRIOR_CLASSES[tag](**kwargs)
    return {key: _decode(item, arrays) for key, item in value.items()}


def save_fit(fit: FitResult, path: str | Path) -> None:
    """Write a non-pickle ``.bucex`` archive atomically."""

    if not isinstance(fit, FitResult):
        raise TypeError("save_fit expects a bucex.FitResult.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    payload = {
        "model": fit.model.to_dict(),
        "priors": fit.priors,
        "y": fit.y,
        "exog": fit.exog,
        "dates": fit.dates,
        "series_name": fit.series_name,
        "transform_sign": fit.transform_sign,
        "state_draws": fit.state_draws,
        "parameter_draws": fit.parameter_draws,
        "log_posterior": fit.log_posterior,
        "plan": fit.plan.to_dict(),
        "sampler_diagnostics": fit.sampler_diagnostics,
        "initial_values": fit.initial_values,
        "auxiliary_draws": fit.auxiliary_draws,
        "metadata": fit.metadata,
        "result_schema": fit.schema_version,
    }
    encoded = _encode(payload, arrays, "fit")
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    array_bytes = buffer.getvalue()
    metadata = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "producer_version": __version__,
        "arrays_sha256": sha256(array_bytes).hexdigest(),
        "payload": encoded,
    }
    metadata_bytes = json.dumps(
        metadata,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", metadata_bytes)
        archive.writestr("arrays.npz", array_bytes)
    temporary.replace(target)


def load_fit(path: str | Path) -> FitResult:
    """Load and validate a v1.2/v2 fit archive without executing serialized code."""

    source = Path(path)
    with zipfile.ZipFile(source, "r") as archive:
        if set(archive.namelist()) != {"metadata.json", "arrays.npz"}:
            raise ValueError("Invalid bucex fit archive members.")
        metadata = json.loads(archive.read("metadata.json"))
        array_bytes = archive.read("arrays.npz")
    if metadata.get("format") != FORMAT:
        raise ValueError("This is not a bucex fit archive.")
    if metadata.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported bucex archive schema {metadata.get('schema_version')!r}."
        )
    if sha256(array_bytes).hexdigest() != metadata.get("arrays_sha256"):
        raise ValueError("Fit archive integrity check failed.")
    with np.load(BytesIO(array_bytes), allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    payload = _decode(metadata["payload"], arrays)
    model_payload = payload["model"]
    kind = model_payload.get("kind")
    if kind == "factor":
        model = FactorModel.from_dict(model_payload)
    elif kind == "multiseries":
        model = MultiSeriesModel.from_dict(model_payload)
    else:
        model = Model.from_dict(model_payload)
    compiled = compile_model(model, payload["y"], exog=payload.get("exog"))
    return FitResult(
        model=model,
        compiled=compiled,
        priors=payload["priors"],
        y=payload["y"],
        exog=payload.get("exog"),
        dates=payload.get("dates"),
        series_name=payload.get("series_name"),
        transform_sign=payload.get("transform_sign", 1.0),
        state_draws=payload["state_draws"],
        parameter_draws=payload["parameter_draws"],
        log_posterior=payload["log_posterior"],
        plan=InferencePlan.from_dict(payload["plan"]),
        sampler_diagnostics=payload.get("sampler_diagnostics", {}),
        schema_version=payload.get("result_schema", SCHEMA_VERSION),
        initial_values=payload.get("initial_values", {}),
        auxiliary_draws=payload.get("auxiliary_draws", {}),
        metadata=payload.get("metadata", {}),
    )


# The old names remain aliases to the same format and result type.
save_posterior_bundle = save_fit
load_posterior_bundle = load_fit

__all__ = [
    "FORMAT",
    "SCHEMA_VERSION",
    "save_fit",
    "load_fit",
    "save_posterior_bundle",
    "load_posterior_bundle",
]
