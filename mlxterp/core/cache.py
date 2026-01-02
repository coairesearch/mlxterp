"""
Activation caching utilities.

Provides efficient caching of activations during forward passes.
"""

import mlx.core as mx
from typing import Dict, List, Optional
from dataclasses import dataclass, field


def _normalize_key(key: str) -> str:
    """
    Normalize an activation key to a short form.

    Removes common prefixes like "model." and "model.model." to provide
    consistent, user-friendly keys.

    Examples:
        "model.model.layers.0.self_attn" -> "layers.0.self_attn"
        "model.layers.0" -> "layers.0"
        "layers.0" -> "layers.0"
    """
    # Remove "model.model." prefix (mlx-lm models)
    if key.startswith("model.model."):
        return key[12:]
    # Remove "model." prefix
    if key.startswith("model."):
        return key[6:]
    return key


def _matches_layer_filter(key: str, layer_filter: str) -> bool:
    """
    Check if an activation key exactly matches a layer filter.

    Handles both short and long key formats.

    Args:
        key: The activation key (e.g., "model.model.layers.0.self_attn")
        layer_filter: The filter (e.g., "layers.0")

    Returns:
        True if the key exactly matches the filter (after normalization)
    """
    normalized_key = _normalize_key(key)
    normalized_filter = _normalize_key(layer_filter)

    # Exact match after normalization
    return normalized_key == normalized_filter


@dataclass
class ActivationCache:
    """
    Storage for cached activations from a model forward pass.

    Attributes:
        activations: Dict mapping module names to activation arrays
        metadata: Additional information about the forward pass
    """

    activations: Dict[str, mx.array] = field(default_factory=dict)
    metadata: Optional[Dict] = None

    def get(self, name: str) -> Optional[mx.array]:
        """
        Get activation by module name.

        Supports both short names (e.g., "layers.0") and full names
        (e.g., "model.model.layers.0").

        Args:
            name: Module name (e.g., "layers.3.self_attn")

        Returns:
            Activation array or None if not found
        """
        # Try exact match first
        if name in self.activations:
            return self.activations[name]

        # Try with common prefixes for backwards compatibility
        prefixes = ["model.model.", "model."]
        for prefix in prefixes:
            prefixed_name = prefix + name
            if prefixed_name in self.activations:
                return self.activations[prefixed_name]

        # Try normalized lookup (search all keys)
        normalized_name = _normalize_key(name)
        for key, value in self.activations.items():
            if _normalize_key(key) == normalized_name:
                return value

        return None

    def keys(self) -> List[str]:
        """Get all cached module names"""
        return list(self.activations.keys())

    def __contains__(self, name: str) -> bool:
        """Check if activation exists in cache"""
        return self.get(name) is not None

    def __len__(self) -> int:
        """Number of cached activations"""
        return len(self.activations)

    def __repr__(self):
        return f"ActivationCache(cached={len(self.activations)} modules)"


def collect_activations(
    model,
    inputs,
    layers: Optional[List[str]] = None,
    normalize_keys: bool = True
) -> ActivationCache:
    """
    Collect activations for specified layers.

    Args:
        model: InterpretableModel instance
        inputs: Input data (text, tokens, or arrays)
        layers: List of layer names to cache (None = all layers).
                Supports short names like "layers.0" or "layers.0.self_attn".
        normalize_keys: If True, normalize keys to short form (default True)

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

    # Filter activations if layers specified
    if layers:
        filtered_activations = {}
        for key, value in trace.activations.items():
            # Skip the special __model_output__ key
            if key == "__model_output__":
                continue
            # Check if this activation matches any of the requested layers
            for layer_filter in layers:
                if _matches_layer_filter(key, layer_filter):
                    # Use normalized key if requested
                    final_key = _normalize_key(key) if normalize_keys else key
                    filtered_activations[final_key] = value
                    break
    else:
        # Return all activations (except __model_output__)
        if normalize_keys:
            filtered_activations = {
                _normalize_key(k): v
                for k, v in trace.activations.items()
                if k != "__model_output__"
            }
        else:
            filtered_activations = {
                k: v for k, v in trace.activations.items()
                if k != "__model_output__"
            }

    return ActivationCache(
        activations=filtered_activations,
        metadata={"input_shape": trace.output.shape if trace.output is not None else None}
    )
