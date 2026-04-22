"""
Path patching for edge-level circuit discovery.

Determines which connections between components matter by freezing
all components except the sender and measuring the effect on the receiver.
"""

import mlx.core as mx
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.intervention import replace_with
from ..core.module_resolver import resolve_component, resolve_intervention_key
from ..metrics import get_metric
from ..results import PatchingResult


def path_patching(
    model,
    clean: Union[str, mx.array],
    corrupted: Union[str, mx.array],
    sender: str,
    receiver: str,
    sender_layer: Optional[int] = None,
    receiver_layer: Optional[int] = None,
    metric: Union[str, Callable] = "l2",
    metric_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> PatchingResult:
    """
    Path patching: measure the causal effect along a specific path.

    Freezes all components to their clean values EXCEPT the sender,
    allowing only the sender's corrupted signal through. Measures
    the resulting effect on the receiver/output.

    Args:
        model: InterpretableModel instance
        clean: Clean input
        corrupted: Corrupted input
        sender: Sender component (e.g., "layers.7.self_attn")
        receiver: Receiver component (e.g., "layers.9.self_attn")
        sender_layer: Layer index if sender is a canonical name
        receiver_layer: Layer index if receiver is a canonical name
        metric: Metric function or name
        metric_kwargs: Extra metric args
        verbose: Print progress

    Returns:
        PatchingResult with path effect
    """
    metric_fn = get_metric(metric)
    metric_name = metric if isinstance(metric, str) else "custom"
    metric_kwargs = metric_kwargs or {}

    # Run clean trace
    if verbose:
        print("Running clean trace...")
    with model.trace(clean) as clean_trace:
        clean_output = model.output.save()

    mx.eval(clean_output)

    # Evaluate all clean activations
    for key in clean_trace.activations:
        mx.eval(clean_trace.activations[key])

    # Run corrupted trace
    if verbose:
        print("Running corrupted trace...")
    with model.trace(corrupted) as corrupted_trace:
        corrupted_output = model.output.save()

    mx.eval(corrupted_output)

    def _last_logits(output):
        if output.ndim == 3:
            return output[0:1, -1:, :].reshape(1, -1)
        return output

    clean_logits = _last_logits(clean_output)
    corrupted_logits = _last_logits(corrupted_output)

    # Resolve sender key
    import re
    sender_key = None
    m = re.match(r"layers\.(\d+)\.(.+)", sender)
    if m:
        sender_key = resolve_component(m.group(2), int(m.group(1)), clean_trace.activations)
    elif sender_layer is not None:
        sender_key = resolve_component(sender, sender_layer, clean_trace.activations)

    if sender_key is None:
        # Try direct match
        for prefix in ["model.model.", "model.", ""]:
            k = f"{prefix}{sender}"
            if k in clean_trace.activations:
                sender_key = k
                break

    # Build freeze-all-except-sender interventions
    interventions = {}
    for key, value in clean_trace.activations.items():
        if key == "__model_output__":
            continue
        if key == sender_key:
            continue
        # Skip sub-keys of the sender (e.g., sender.resid_pre)
        if sender_key and key.startswith(sender_key + "."):
            continue
        iv_key = resolve_intervention_key(key)
        interventions[iv_key] = replace_with(value)

    if verbose:
        print(f"Freezing {len(interventions)} components, sender={sender_key}")

    # Run with all frozen except sender
    with model.trace(corrupted, interventions=interventions):
        patched_output = model.output.save()

    mx.eval(patched_output)
    patched_logits = _last_logits(patched_output)

    effect = metric_fn(
        patched_logits, clean_logits, corrupted_logits,
        **metric_kwargs,
    )

    return PatchingResult(
        data={
            "sender": sender,
            "receiver": receiver,
            "effect": float(effect),
            "metric": metric_name,
            "n_frozen": len(interventions),
        },
        metadata={
            "clean": clean if isinstance(clean, str) else f"array({clean.shape})",
            "corrupted": corrupted if isinstance(corrupted, str) else f"array({corrupted.shape})",
        },
        effect_matrix=mx.array([float(effect)]),
        layers=[],
        component=f"path:{sender}->{receiver}",
        metric_name=metric_name,
    )
