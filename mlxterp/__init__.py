"""
mlxterp: Mechanistic Interpretability for MLX

A clean, intuitive library for mechanistic interpretability on Apple Silicon.

Example:
    >>> from mlxterp import InterpretableModel
    >>> import mlx.core as mx
    >>>
    >>> # Load any MLX model
    >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
    >>>
    >>> # Trace execution and capture activations
    >>> with model.trace("The capital of France is"):
    >>>     attn_3 = model.layers[3].attn.output.save()
    >>>     mlp_8 = model.layers[8].mlp.output.save()
    >>>     logits = model.output.save()
    >>>
    >>> # Use intervention functions
    >>> from mlxterp import interventions as iv
    >>> with model.trace("Hello", interventions={"layers.4": iv.scale(0.5)}):
    >>>     modified_output = model.output.save()
"""

__version__ = "0.1.0"

from .model import InterpretableModel
from .core import (
    # Intervention functions
    zero_out,
    scale,
    add_vector,
    replace_with,
    clamp,
    noise,
    InterventionComposer,
    # Cache
    ActivationCache,
    collect_activations,
)
from .utils import get_activations, batch_get_activations

# Create interventions namespace for cleaner imports
class _Interventions:
    """Namespace for intervention functions"""
    zero_out = staticmethod(zero_out)
    scale = staticmethod(scale)
    add_vector = staticmethod(add_vector)
    replace_with = staticmethod(replace_with)
    clamp = staticmethod(clamp)
    noise = staticmethod(noise)
    compose = InterventionComposer

interventions = _Interventions()

__all__ = [
    # Main class
    "InterpretableModel",
    # Interventions
    "interventions",
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
    # Utils
    "get_activations",
    "batch_get_activations",
]
