"""
Intervention utilities for modifying activations during forward passes.

Provides helper functions for common intervention patterns.
"""

import mlx.core as mx
from typing import Union, Callable


def zero_out(x: mx.array) -> mx.array:
    """
    Zero out an activation.

    Example:
        with model.trace(input, interventions={"layers.4": zero_out}):
            ...
    """
    return mx.zeros_like(x)


def scale(factor: float) -> Callable[[mx.array], mx.array]:
    """
    Scale an activation by a constant factor.

    Args:
        factor: Scaling factor

    Returns:
        Intervention function

    Example:
        with model.trace(input, interventions={"layers.4": scale(0.5)}):
            ...
    """
    def _scale(x: mx.array) -> mx.array:
        return x * factor
    return _scale


def add_vector(vector: mx.array) -> Callable[[mx.array], mx.array]:
    """
    Add a vector to an activation (steering vector).

    Args:
        vector: Vector to add (must be broadcastable)

    Returns:
        Intervention function

    Example:
        steering_vec = mx.random.normal((hidden_dim,))
        with model.trace(input, interventions={"layers.4": add_vector(steering_vec)}):
            ...
    """
    def _add(x: mx.array) -> mx.array:
        return x + vector
    return _add


def replace_with(value: Union[mx.array, float], align: str = "end") -> Callable[[mx.array], mx.array]:
    """
    Replace activation with a fixed value.

    Args:
        value: Replacement value (array or scalar)
        align: How to align when sequence lengths differ:
            - "end": Align at end (last tokens match) - default, best for activation patching
            - "start": Align at start (first tokens match)
            - "strict": Raise error if shapes don't match

    Returns:
        Intervention function

    Example:
        with model.trace(input, interventions={"layers.4": replace_with(0.0)}):
            ...

        # Activation patching with different sequence lengths:
        with model.trace(clean_text) as trace:
            clean_act = trace.activations["model.model.layers.10.mlp"]
        with model.trace(corrupted_text,
                        interventions={"layers.10.mlp": replace_with(clean_act)}):
            patched = model.output.save()
    """
    def _replace(x: mx.array) -> mx.array:
        if isinstance(value, (int, float)):
            return mx.full(x.shape, value)

        # If shapes match exactly, use the value directly
        if value.shape == x.shape:
            return value

        # Handle sequence length mismatch (common in activation patching)
        # Assuming shape is (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
        if value.ndim >= 2 and x.ndim >= 2:
            # Get sequence dimension (usually axis 1 for 3D, axis 0 for 2D)
            seq_axis = 1 if value.ndim == 3 else 0
            value_seq_len = value.shape[seq_axis]
            x_seq_len = x.shape[seq_axis]

            if value_seq_len != x_seq_len:
                if align == "strict":
                    raise ValueError(
                        f"Shape mismatch: replacement has shape {value.shape} but target has shape {x.shape}. "
                        f"Use align='end' or align='start' to handle different sequence lengths."
                    )

                # Create output with same shape as x
                result = x.copy() if hasattr(x, 'copy') else mx.array(x)

                if align == "end":
                    # Align at end: patch last min(value_seq_len, x_seq_len) tokens
                    min_len = min(value_seq_len, x_seq_len)
                    if value.ndim == 3:
                        result[:, -min_len:, :] = value[:, -min_len:, :]
                    else:
                        result[-min_len:, :] = value[-min_len:, :]
                else:  # align == "start"
                    # Align at start: patch first min(value_seq_len, x_seq_len) tokens
                    min_len = min(value_seq_len, x_seq_len)
                    if value.ndim == 3:
                        result[:, :min_len, :] = value[:, :min_len, :]
                    else:
                        result[:min_len, :] = value[:min_len, :]

                return result

        # Fallback: try to broadcast (will fail if truly incompatible)
        return mx.broadcast_to(value, x.shape)
    return _replace


def clamp(min_val: float = None, max_val: float = None) -> Callable[[mx.array], mx.array]:
    """
    Clamp activation values to a range.

    Args:
        min_val: Minimum value (None for no minimum)
        max_val: Maximum value (None for no maximum)

    Returns:
        Intervention function

    Example:
        with model.trace(input, interventions={"layers.4": clamp(min_val=-1, max_val=1)}):
            ...
    """
    def _clamp(x: mx.array) -> mx.array:
        result = x
        if min_val is not None:
            result = mx.maximum(result, min_val)
        if max_val is not None:
            result = mx.minimum(result, max_val)
        return result
    return _clamp


def noise(std: float = 0.1) -> Callable[[mx.array], mx.array]:
    """
    Add Gaussian noise to an activation.

    Args:
        std: Standard deviation of noise

    Returns:
        Intervention function

    Example:
        with model.trace(input, interventions={"layers.4": noise(std=0.1)}):
            ...
    """
    def _noise(x: mx.array) -> mx.array:
        return x + mx.random.normal(x.shape) * std
    return _noise


class InterventionComposer:
    """
    Compose multiple interventions into a single function.

    Example:
        combined = InterventionComposer() \\
            .add(scale(0.5)) \\
            .add(add_vector(steering_vec)) \\
            .build()

        with model.trace(input, interventions={"layers.4": combined}):
            ...
    """

    def __init__(self):
        self.interventions = []

    def add(self, fn: Callable[[mx.array], mx.array]) -> 'InterventionComposer':
        """Add an intervention to the composition"""
        self.interventions.append(fn)
        return self

    def build(self) -> Callable[[mx.array], mx.array]:
        """Build the composed intervention function"""
        def _composed(x: mx.array) -> mx.array:
            result = x
            for fn in self.interventions:
                result = fn(result)
            return result
        return _composed
