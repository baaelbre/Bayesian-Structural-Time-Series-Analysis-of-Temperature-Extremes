"""Core result objects and numerical utilities."""
from .fit import BulkTailFit, FitResult, PosteriorBundle, combine_fits

__all__ = ["FitResult", "PosteriorBundle", "BulkTailFit", "combine_fits"]
