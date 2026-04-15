"""
Causal interpretability tools for mlxterp.

Provides activation patching, causal trace, attribution patching,
path patching, and circuit discovery utilities.
"""

from .patching import activation_patching
from .trace import CausalTrace

__all__ = [
    "activation_patching",
    "CausalTrace",
]
