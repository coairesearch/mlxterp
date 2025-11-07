"""Core components for mlxterp."""

from .proxy import ModuleProxy, OutputProxy, LayerListProxy, TraceContext
from .trace import Trace
from .intervention import (
    zero_out,
    scale,
    add_vector,
    replace_with,
    clamp,
    noise,
    InterventionComposer,
)
from .cache import ActivationCache, collect_activations

__all__ = [
    # Proxy
    "ModuleProxy",
    "OutputProxy",
    "LayerListProxy",
    "TraceContext",
    # Trace
    "Trace",
    # Intervention
    "zero_out",
    "scale",
    "add_vector",
    "replace_with",
    "clamp",
    "noise",
    "InterventionComposer",
    # Cache
    "ActivationCache",
    "collect_activations",
]
