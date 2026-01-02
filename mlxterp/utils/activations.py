"""
Utility functions for collecting and analyzing activations.
"""

import mlx.core as mx
from typing import List, Dict, Union, Optional


def get_activations(
    model,
    prompts: Union[str, List[str]],
    layers: Optional[List[int]] = None,
    positions: Union[int, List[int]] = -1,
) -> Dict[str, mx.array]:
    """
    Collect activations for specified layers and token positions.

    Args:
        model: InterpretableModel instance
        prompts: Single prompt or list of prompts
        layers: List of layer indices (None = all layers)
        positions: Token position(s) to extract. -1 = last token

    Returns:
        Dict mapping "layer_{i}" to activation arrays
        Shape: (batch_size, hidden_dim) for single position
               (batch_size, num_positions, hidden_dim) for multiple positions

    Example:
        >>> acts = get_activations(
        ...     model,
        ...     ["Hello", "World"],
        ...     layers=[3, 8, 12],
        ...     positions=-1
        ... )
        >>> acts["layer_3"].shape  # (2, hidden_dim)
    """
    # Normalize inputs
    if isinstance(prompts, str):
        prompts = [prompts]

    if layers is None:
        layers = list(range(len(model.layers)))

    if isinstance(positions, int):
        positions = [positions]

    # Collect activations
    result = {}

    with model.trace(prompts) as trace:
        for layer_idx in layers:
            # Save the layer output
            act = model.layers[layer_idx].output.save()

    # Extract saved activations
    # Look for activation keys that match the layer pattern
    for layer_idx in layers:
        # Find the activation key for this layer
        # Keys can be like "model.model.layers.{i}" or "layers.{i}" depending on model
        activation_key = None
        for key in trace.activations.keys():
            if key.endswith(f".{model._layer_attr}.{layer_idx}") or key == f"{model._layer_attr}.{layer_idx}":
                activation_key = key
                break

        if activation_key is not None:
            act = trace.activations[activation_key]

            # Extract positions
            # act shape: (batch, seq_len, hidden_dim)
            if len(positions) == 1:
                pos = positions[0]
                result[f"layer_{layer_idx}"] = act[:, pos, :]
            else:
                # Multiple positions
                extracted = mx.stack([act[:, pos, :] for pos in positions], axis=1)
                result[f"layer_{layer_idx}"] = extracted

    return result


def batch_get_activations(
    model,
    prompts: List[str],
    layers: Optional[List[int]] = None,
    positions: Union[int, List[int]] = -1,
    batch_size: int = 8,
) -> Dict[str, mx.array]:
    """
    Collect activations in batches for memory efficiency.

    Args:
        model: InterpretableModel instance
        prompts: List of prompts
        layers: List of layer indices (None = all layers)
        positions: Token position(s) to extract
        batch_size: Number of prompts per batch

    Returns:
        Dict mapping "layer_{i}" to concatenated activation arrays

    Example:
        >>> acts = batch_get_activations(
        ...     model,
        ...     large_prompt_list,
        ...     layers=[3, 8],
        ...     batch_size=16
        ... )
    """
    if layers is None:
        layers = list(range(len(model.layers)))

    all_results = {f"layer_{i}": [] for i in layers}

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_acts = get_activations(model, batch, layers, positions)

        for key, value in batch_acts.items():
            all_results[key].append(value)

    # Concatenate results
    final_results = {}
    for key, values in all_results.items():
        if values:
            final_results[key] = mx.concatenate(values, axis=0)

    return final_results
