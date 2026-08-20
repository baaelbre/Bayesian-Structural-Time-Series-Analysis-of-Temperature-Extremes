"""bucex: Bayesian unobserved components for environmental extremes.

The top-level namespace exposes the stable modelling, fitting, diagnostic,
simulation, persistence, plotting, data, and presentation-workflow APIs.
"""
from __future__ import annotations

from .__about__ import __version__
from . import api as _api
from . import components as _components
from . import core as _core
from . import datasets as _datasets
from . import diagnostics as _diagnostics
from . import inference as _inference
from . import io as _io
from . import models as _models
from . import observation as _observation
from . import plotting as _plotting
from . import priors as _priors
from . import simulate as _simulate
from . import workflows as _workflows
from .api import *
from .components import *
from .core import *
from .datasets import *
from .diagnostics import *
from .inference import *
from .io import *
from .models import *
from .models.compiler import CompiledModel, compile_model
from .observation import *
from .plotting import *
from .priors import *
from .simulate import *
from .workflows import *


__all__ = sorted(
    {
        "__version__",
        "CompiledModel",
        "compile_model",
        *(_api.__all__),
        *(_components.__all__),
        *(_core.__all__),
        *(_datasets.__all__),
        *(_diagnostics.__all__),
        *(_inference.__all__),
        *(_io.__all__),
        *(_models.__all__),
        *(_observation.__all__),
        *(_plotting.__all__),
        *(_priors.__all__),
        *(_simulate.__all__),
        *(_workflows.__all__),
    }
)
