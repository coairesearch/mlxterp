"""
Activation caching utilities.

Provides efficient caching of activations during forward passes.
"""

import mlx.core as mx
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ActivationCache:
    """
    Storage for cached activations from a model forward pass.

    Attributes:
        activations: Dict mapping module names to activation arrays
        metadata: Additional information about the forward pass
    """

    activations: Dict[str, mx.array]
    metadata: Optional[Dict] = None

    def get(self, name: str) -> Optional[mx.array]:
        """
        Get activation by module name.

        Args:
            name: Module name (e.g., "layers.3.attn")

        Returns:
            Activation array or None if not found
        """
        return self.activations.get(name)

    def keys(self) -> List[str]:
        """Get all cached module names"""
        return list(self.activations.keys())

    def __contains__(self, name: str) -> bool:
        """Check if activation exists in cache"""
        return name in self.activations

    def __len__(self) -> int:
        """Number of cached activations"""
        return len(self.activations)

    def __repr__(self):
        return f"ActivationCache(cached={len(self.activations)} modules)"


def collect_activations(
    model,
    inputs,
    layers: Optional[List[str]] = None
) -> ActivationCache:
    """
    Collect activations for specified layers.

    Args:
        model: InterpretableModel instance
        inputs: Input data (text, tokens, or arrays)
        layers: List of layer names to cache (None = all layers)

    Returns:
        ActivationCache with the requested activations

    Example:
        cache = collect_activations(
            model,
            "The capital of France is",
            layers=["layers.3", "layers.8"]
        )
        attn_3 = cache.get("layers.3")
    """
    # Use the model's trace functionality
    with model.trace(inputs) as trace:
        # If specific layers requested, save them
        if layers:
            for layer_name in layers:
                # Access the layer to trigger caching
                parts = layer_name.split('.')
                obj = model
                for part in parts:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)

                # The output property triggers saving
                if hasattr(obj, 'output'):
                    obj.output.save()

    # Return cache with all activations
    return ActivationCache(
        activations=trace.activations,
        metadata={"input_shape": trace.output.shape if trace.output is not None else None}
    )
