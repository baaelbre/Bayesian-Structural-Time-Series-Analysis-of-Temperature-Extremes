"""Safe persistence for fitted models."""
from .posterior import (
    load_fit,
    load_posterior_bundle,
    save_fit,
    save_posterior_bundle,
)

__all__ = [
    "save_fit",
    "load_fit",
    "save_posterior_bundle",
    "load_posterior_bundle",
]
