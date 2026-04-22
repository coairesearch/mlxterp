"""
Causal interpretability tools for mlxterp.

Provides activation patching, causal trace, attribution patching,
path patching, and circuit discovery utilities.
"""

from .patching import activation_patching
from .trace import CausalTrace
from .residual import ResidualStreamAccessor
from .dla import direct_logit_attribution
from .attribution import attribution_patching
from .path_patching import path_patching
from .acdc import acdc

__all__ = [
    "activation_patching",
    "CausalTrace",
    "ResidualStreamAccessor",
    "direct_logit_attribution",
    "attribution_patching",
    "path_patching",
    "acdc",
]
