"""Safe, checksummed archives for Fruehwirth--Schnatter fit objects."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from ..__about__ import __version__
from ..components.regression import RegressionComponent
from ..components.seasonal import DummySeasonal
from ..components.trend import LocalLinearTrend
from ..core.results import PosteriorBundle
from ..models.structural import StructuralSSM
from ..observation.base import ObsSpec
from ..observation.gaussian import GaussianObs
from ..observation.gev import GEVObs
from ..inference.fit import priors as prior_types
from .._general.plan import InferencePlan


FORMAT = "bucex-fs-fit"
SCHEMA_VERSION = "1.1"


def _allowed_classes() -> dict[str, type]:
    classes: list[type] = [
        RegressionComponent,
        DummySeasonal,
        LocalLinearTrend,
        StructuralSSM,
        ObsSpec,
        GaussianObs,
        GEVObs,
        InferencePlan,
    ]
    for name in dir(prior_types):
        candidate = getattr(prior_types, name)
        if isinstance(candidate, type) and is_dataclass(candidate):
            classes.append(candidate)
    return {
        f"{candidate.__module__}:{candidate.__qualname__}": candidate
        for candidate in classes
    }


_ALLOWED_CLASSES = _allowed_classes()


def _encode(value: Any, arrays: dict[str, np.ndarray], prefix: str) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            array = array.astype(str)
        key = f"array_{len(arrays):06d}_{prefix.replace('.', '_')}"
        arrays[key] = array
        return {"__array__": key}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        tag = f"{value.__class__.__module__}:{value.__class__.__qualname__}"
        if tag not in _ALLOWED_CLASSES:
            raise TypeError(f"Unsupported dataclass in safe fit archive: {tag}")
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
    if isinstance(value, dict):
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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__string__": str(value)}


def _decode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    if not isinstance(value, dict):
        return value
    if "__array__" in value:
        key = value["__array__"]
        if key not in arrays:
            raise ValueError(f"Fit archive references a missing array: {key}")
        return np.asarray(arrays[key])
    if "__tuple__" in value:
        return tuple(_decode(item, arrays) for item in value["__tuple__"])
    if "__mapping__" in value:
        return {
            _decode(pair[0], arrays): _decode(pair[1], arrays)
            for pair in value["__mapping__"]
        }
    if "__dataclass__" in value:
        tag = value["__dataclass__"]
        if tag not in _ALLOWED_CLASSES:
            raise ValueError(f"Fit archive contains an unapproved class: {tag}")
        kwargs = {
            key: _decode(item, arrays)
            for key, item in value.get("fields", {}).items()
        }
        return _ALLOWED_CLASSES[tag](**kwargs)
    if "__string__" in value:
        return value["__string__"]
    return {key: _decode(item, arrays) for key, item in value.items()}


def save_posterior_bundle(fit: PosteriorBundle, path: str | Path) -> None:
    """Write JSON metadata and non-pickled NumPy arrays to ``.bucex``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    payload = {
        "draws_static": fit.draws_static,
        "draws_states": fit.draws_states,
        "logpost": fit.logpost,
        "acceptance": fit.acceptance,
        "meta": fit.meta,
        "y": fit.y,
        "dates": fit.dates,
        "model": fit.model,
        "state_names": fit.state_names,
        "series_name": fit.series_name,
        "transform_sign": fit.transform_sign,
        "priors": getattr(fit, "priors", None),
        "config": getattr(fit, "config", {}),
        "initial_values": getattr(fit, "initial_values", {}),
        "exog": getattr(fit, "exog", None),
        "plan": getattr(fit, "plan", None),
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
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, sort_keys=True))
        archive.writestr("arrays.npz", array_bytes)


def load_posterior_bundle(path: str | Path) -> PosteriorBundle:
    """Load a safe FS archive after format, checksum, and class validation."""
    source = Path(path)
    with zipfile.ZipFile(source, "r") as archive:
        names = set(archive.namelist())
        if names != {"metadata.json", "arrays.npz"}:
            raise ValueError("Invalid bucex fit archive members.")
        metadata = json.loads(archive.read("metadata.json"))
        array_bytes = archive.read("arrays.npz")
    if metadata.get("format") != FORMAT:
        raise ValueError("This is not a Fruehwirth--Schnatter bucex fit archive.")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported FS archive schema {metadata.get('schema_version')!r}."
        )
    if sha256(array_bytes).hexdigest() != metadata.get("arrays_sha256"):
        raise ValueError("Fit archive integrity check failed.")
    with np.load(BytesIO(array_bytes), allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    payload = _decode(metadata["payload"], arrays)
    fit = PosteriorBundle(
        draws_static=payload["draws_static"],
        draws_states=payload["draws_states"],
        logpost=payload["logpost"],
        acceptance=payload["acceptance"],
        meta=payload["meta"],
        y=payload["y"],
        dates=payload["dates"],
        model=payload["model"],
        state_names=payload["state_names"],
        series_name=payload["series_name"],
        transform_sign=float(payload["transform_sign"]),
    )
    fit.priors = payload.get("priors")
    fit.config = payload.get("config", {})
    fit.initial_values = payload.get("initial_values", {})
    fit.exog = payload.get("exog")
    fit.plan = payload.get("plan")
    return fit
