"""
Activation patching API for causal interpretability.

Implements component-level, position-level, and head-level activation patching
with support for multiple metrics and result visualization.
"""

import mlx.core as mx
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.intervention import replace_with, replace_at_positions
from ..core.module_resolver import resolve_component, resolve_intervention_key
from ..metrics import get_metric
from ..results import PatchingResult


def activation_patching(
    model,
    clean: Union[str, mx.array],
    corrupted: Union[str, mx.array],
    layers: Optional[List[int]] = None,
    component: str = "resid_post",
    metric: Union[str, Callable] = "l2",
    positions: Optional[List[int]] = None,
    metric_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> PatchingResult:
    """
    Run activation patching to identify causally important components.

    Patches clean activations into corrupted forward passes at each layer
    and measures how much this recovers the clean output.

    Args:
        model: InterpretableModel instance
        clean: Clean/correct input (text or token array)
        corrupted: Corrupted/counterfactual input (text or token array)
        layers: Layer indices to test. None = all layers.
        component: Component to patch. Options:
            - "resid_post": Layer output (residual stream after layer)
            - "attn" or "self_attn": Attention component
            - "mlp": MLP/feed-forward component
            - "attn_head": Per-head patching (returns n_layers x n_heads matrix)
        metric: Metric function or name string. Available:
            "logit_diff", "kl", "l2", "cosine", "ce_diff"
        positions: Token positions to patch. None = all positions.
        metric_kwargs: Extra kwargs passed to metric (e.g., correct_token, incorrect_token)
        verbose: Print progress during patching

    Returns:
        PatchingResult with effect_matrix and metadata
    """
    metric_fn = get_metric(metric)
    metric_name = metric if isinstance(metric, str) else getattr(metric, "__name__", "custom")
    metric_kwargs = metric_kwargs or {}

    # Get baseline outputs
    if verbose:
        print("Running clean forward pass...")
    with model.trace(clean):
        clean_output = model.output.save()

    if verbose:
        print("Running corrupted forward pass...")
    with model.trace(corrupted):
        corrupted_output = model.output.save()

    mx.eval(clean_output, corrupted_output)

    # Use last-token logits for metrics
    clean_logits = clean_output[0:1, -1:, :].reshape(1, -1) if clean_output.ndim == 3 else clean_output
    corrupted_logits = corrupted_output[0:1, -1:, :].reshape(1, -1) if corrupted_output.ndim == 3 else corrupted_output

    # Determine layers
    if layers is None:
        layers = list(range(len(model.layers)))

    if component == "attn_head":
        return _patching_per_head(
            model, clean, corrupted, layers,
            clean_logits, corrupted_logits,
            metric_fn, metric_name, metric_kwargs,
            verbose,
        )

    effects = []
    for layer_idx in layers:
        if verbose:
            print(f"  Patching layer {layer_idx}...", end="\r")

        # Get clean activation for this component
        with model.trace(clean) as clean_trace:
            pass  # just run to get activations

        act_key = resolve_component(component, layer_idx, clean_trace.activations)
        if act_key is None:
            if verbose:
                print(f"  Warning: No activation found for {component} at layer {layer_idx}")
            effects.append(0.0)
            continue

        clean_act = clean_trace.activations[act_key]
        mx.eval(clean_act)

        # Build intervention
        intervention_key = resolve_intervention_key(act_key)
        if positions is not None:
            intervention = replace_at_positions(clean_act, positions)
        else:
            intervention = replace_with(clean_act)

        # Run corrupted with patch
        with model.trace(corrupted, interventions={intervention_key: intervention}):
            patched_output = model.output.save()

        mx.eval(patched_output)

        patched_logits = patched_output[0:1, -1:, :].reshape(1, -1) if patched_output.ndim == 3 else patched_output

        effect = metric_fn(
            patched_logits, clean_logits, corrupted_logits,
            **metric_kwargs,
        )
        effects.append(float(effect))

    if verbose:
        print()

    effect_matrix = mx.array(effects)

    return PatchingResult(
        data={
            "effects": {layer_idx: eff for layer_idx, eff in zip(layers, effects)},
            "component": component,
            "metric": metric_name,
        },
        metadata={
            "clean": clean if isinstance(clean, str) else f"array({clean.shape})",
            "corrupted": corrupted if isinstance(corrupted, str) else f"array({corrupted.shape})",
            "n_layers": len(layers),
            "positions": positions,
        },
        effect_matrix=effect_matrix,
        layers=layers,
        component=component,
        metric_name=metric_name,
    )


def _patching_per_head(
    model,
    clean,
    corrupted,
    layers,
    clean_logits,
    corrupted_logits,
    metric_fn,
    metric_name,
    metric_kwargs,
    verbose,
) -> PatchingResult:
    """Perform per-head activation patching.

    For each attention head, patches only that head's contribution
    from clean into corrupted and measures the effect.
    """
    # First, determine number of heads from a trace
    with model.trace(clean) as clean_trace:
        pass

    # Find an attention activation to determine head count
    n_heads = None
    sample_key = None
    for layer_idx in layers:
        key = resolve_component("attn", layer_idx, clean_trace.activations)
        if key is not None:
            sample_key = key
            act = clean_trace.activations[key]
            # attn output shape: (batch, seq, n_heads * head_dim)
            # We need to find n_heads from model config or attention weights
            attn_weight_key = f"{key}.attention_weights"
            if attn_weight_key in clean_trace.activations:
                # attention_weights shape: (batch, n_heads, seq, seq)
                n_heads = clean_trace.activations[attn_weight_key].shape[1]
            break

    if n_heads is None:
        # Fallback: try to get from model config
        try:
            if hasattr(model.model, 'args'):
                n_heads = model.model.args.num_attention_heads
            elif hasattr(model.model, 'model') and hasattr(model.model.model, 'args'):
                n_heads = model.model.model.args.num_attention_heads
            elif hasattr(model.model, 'config'):
                n_heads = model.model.config.num_attention_heads
        except Exception:
            pass

    if n_heads is None:
        raise ValueError(
            "Could not determine number of attention heads. "
            "Ensure the model exposes attention weights during tracing."
        )

    head_dim = None
    effects = []

    for layer_idx in layers:
        layer_effects = []

        # Get clean attention output
        with model.trace(clean) as ct:
            pass

        attn_key = resolve_component("attn", layer_idx, ct.activations)
        if attn_key is None:
            layer_effects = [0.0] * n_heads
            effects.append(layer_effects)
            continue

        clean_attn = ct.activations[attn_key]
        mx.eval(clean_attn)

        if head_dim is None:
            total_dim = clean_attn.shape[-1]
            head_dim = total_dim // n_heads

        intervention_key = resolve_intervention_key(attn_key)

        for head_idx in range(n_heads):
            if verbose:
                print(f"  Layer {layer_idx}, Head {head_idx}...", end="\r")

            # Create per-head intervention: only replace this head's slice
            start = head_idx * head_dim
            end = start + head_dim

            def _head_patch(x, clean_act=clean_attn, s=start, e=end):
                result = mx.array(x)
                # Handle sequence length mismatch
                min_seq = min(x.shape[1], clean_act.shape[1])
                if x.ndim == 3:
                    result[:, -min_seq:, s:e] = clean_act[:, -min_seq:, s:e]
                return result

            with model.trace(corrupted, interventions={intervention_key: _head_patch}):
                patched_output = model.output.save()

            mx.eval(patched_output)
            patched_logits = patched_output[0:1, -1:, :].reshape(1, -1) if patched_output.ndim == 3 else patched_output

            effect = metric_fn(
                patched_logits, clean_logits, corrupted_logits,
                **metric_kwargs,
            )
            layer_effects.append(float(effect))

        effects.append(layer_effects)

    if verbose:
        print()

    effect_matrix = mx.array(effects)

    return PatchingResult(
        data={
            "effects": {
                layer_idx: {h: effects[i][h] for h in range(n_heads)}
                for i, layer_idx in enumerate(layers)
            },
            "component": "attn_head",
            "metric": metric_name,
            "n_heads": n_heads,
        },
        metadata={
            "clean": clean if isinstance(clean, str) else f"array({clean.shape})",
            "corrupted": corrupted if isinstance(corrupted, str) else f"array({corrupted.shape})",
            "n_layers": len(layers),
            "n_heads": n_heads,
        },
        effect_matrix=effect_matrix,
        layers=layers,
        component="attn_head",
        metric_name=metric_name,
    )
