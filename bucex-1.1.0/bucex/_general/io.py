"""Versioned, non-pickle fit serialization."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .__about__ import __version__
from .compiler import compile_model
from .model import Model
from .plan import InferencePlan
from .priors import Priors
from .results import FitResult


FORMAT = "bucex-fit"
SCHEMA_VERSION = "1.0"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_fit(fit: FitResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "y": fit.y,
        "state_draws": fit.state_draws,
        "log_posterior": fit.log_posterior,
    }
    if fit.exog is not None:
        arrays["exog"] = np.asarray(fit.exog)
    if fit.dates is not None:
        dates = np.asarray(fit.dates)
        arrays["dates"] = dates.astype(str) if dates.dtype.kind == "O" else dates
    for name, values in fit.parameter_draws.items():
        arrays[f"parameter::{name}"] = np.asarray(values)
    for name, values in fit.auxiliary_draws.items():
        arrays[f"auxiliary::{name}"] = np.asarray(values)
    diagnostics = fit.sampler_diagnostics
    for group in ("acceptance", "final_proposal_steps", "draw_metrics"):
        for name, values in diagnostics.get(group, {}).items():
            arrays[f"diagnostic::{group}::{name}"] = np.asarray(values)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    array_bytes = buffer.getvalue()
    metadata = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "created_with": __version__,
        "array_sha256": hashlib.sha256(array_bytes).hexdigest(),
        "model": fit.model.to_dict(),
        "priors": fit.priors.to_dict(),
        "plan": fit.plan.to_dict(),
        "series_name": fit.series_name,
        "transform_sign": float(fit.transform_sign),
        "initial_values": _jsonable(fit.initial_values),
        "diagnostics_metadata": {
            key: _jsonable(value)
            for key, value in diagnostics.items()
            if key not in {"acceptance", "final_proposal_steps", "draw_metrics"}
        },
    }
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True))
        archive.writestr("arrays.npz", array_bytes)


def load_fit(path: str | Path) -> FitResult:
    with zipfile.ZipFile(Path(path), "r") as archive:
        metadata = json.loads(archive.read("metadata.json"))
        array_bytes = archive.read("arrays.npz")
    if metadata.get("format") != FORMAT or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported bucex fit format or schema version.")
    if hashlib.sha256(array_bytes).hexdigest() != metadata.get("array_sha256"):
        raise ValueError("Fit archive integrity check failed.")
    with np.load(io.BytesIO(array_bytes), allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    model = Model.from_dict(metadata["model"])
    exog = arrays.get("exog")
    compiled = compile_model(model, arrays["y"], exog=exog)
    parameter_draws = {
        name.split("::", 1)[1]: values
        for name, values in arrays.items()
        if name.startswith("parameter::")
    }
    auxiliary_draws = {
        name.split("::", 1)[1]: values
        for name, values in arrays.items()
        if name.startswith("auxiliary::")
    }
    diagnostics = dict(metadata.get("diagnostics_metadata", {}))
    for group in ("acceptance", "final_proposal_steps", "draw_metrics"):
        prefix = f"diagnostic::{group}::"
        diagnostics[group] = {
            name[len(prefix) :]: values
            for name, values in arrays.items()
            if name.startswith(prefix)
        }
    return FitResult(
        model=model,
        compiled=compiled,
        priors=Priors.from_dict(metadata["priors"]),
        y=arrays["y"],
        exog=exog,
        dates=arrays.get("dates"),
        series_name=metadata.get("series_name"),
        transform_sign=float(metadata.get("transform_sign", 1.0)),
        state_draws=arrays["state_draws"],
        parameter_draws=parameter_draws,
        log_posterior=arrays["log_posterior"],
        plan=InferencePlan.from_dict(metadata["plan"]),
        sampler_diagnostics=diagnostics,
        initial_values=metadata.get("initial_values", {}),
        auxiliary_draws=auxiliary_draws,
    )
