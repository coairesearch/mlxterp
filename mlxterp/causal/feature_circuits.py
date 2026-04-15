"""
SAE Feature Circuits: causal analysis at the feature level.

Computes indirect effects of individual SAE features on model output,
enabling feature-level circuit discovery and pruning.
"""

import mlx.core as mx
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.intervention import replace_with
from ..core.module_resolver import resolve_component, resolve_intervention_key
from ..metrics import get_metric
from ..results import CircuitResult


def feature_patching(
    model,
    sae,
    text: Union[str, mx.array],
    layer: int,
    component: str = "mlp",
    feature_ids: Optional[List[int]] = None,
    top_k: int = 20,
    metric: Union[str, Callable] = "l2",
    metric_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[int, float]:
    """
    Compute causal effect of individual SAE features via ablation.

    For each active feature, zero it out and measure the effect on output.

    Args:
        model: InterpretableModel instance
        sae: Trained SAE (must have encode/decode methods)
        text: Input text or tokens
        layer: Layer to analyze
        component: Component the SAE was trained on
        feature_ids: Specific features to test. None = top_k most active.
        top_k: Number of top features to test if feature_ids not specified
        metric: Metric function or name
        metric_kwargs: Extra metric args
        verbose: Print progress

    Returns:
        Dict mapping feature_id -> causal effect score
    """
    metric_fn = get_metric(metric)
    metric_kwargs = metric_kwargs or {}

    # Get baseline activations
    with model.trace(text) as trace:
        baseline_output = model.output.save()

    mx.eval(baseline_output)

    # Find the activation for this layer/component
    act_key = resolve_component(component, layer, trace.activations)
    if act_key is None:
        raise ValueError(f"No activation found for {component} at layer {layer}")

    activation = trace.activations[act_key]
    mx.eval(activation)

    # Encode through SAE to get feature activations
    # Handle shape: activation may be (batch, seq, hidden)
    act_flat = activation.reshape(-1, activation.shape[-1])
    features = sae.encode(act_flat)  # (batch*seq, n_features)
    mx.eval(features)

    # Determine which features to test
    if feature_ids is None:
        # Get top-k most active features
        max_activations = mx.max(mx.abs(features), axis=0)  # (n_features,)
        mx.eval(max_activations)
        top_indices = mx.argsort(max_activations)[::-1][:top_k]
        feature_ids = [int(i) for i in top_indices.tolist()]

    def _last_logits(output):
        if output.ndim == 3:
            return output[0:1, -1:, :].reshape(1, -1)
        return output

    baseline_logits = _last_logits(baseline_output)

    # Test each feature by zeroing it out
    effects = {}
    intervention_key = resolve_intervention_key(act_key)

    for feat_id in feature_ids:
        if verbose:
            print(f"  Testing feature {feat_id}...", end="\r")

        # Zero out this feature and decode back
        modified_features = mx.array(features)
        modified_features[:, feat_id] = 0.0
        modified_activation = sae.decode(modified_features)
        modified_activation = modified_activation.reshape(activation.shape)

        # Run with modified activation
        with model.trace(text, interventions={
            intervention_key: replace_with(modified_activation)
        }):
            patched_output = model.output.save()

        mx.eval(patched_output)
        patched_logits = _last_logits(patched_output)

        effect = metric_fn(
            patched_logits, baseline_logits, baseline_logits,
            **metric_kwargs,
        )
        effects[feat_id] = float(effect)

    if verbose:
        print()

    return effects


def feature_circuit(
    model,
    sae,
    text: Union[str, mx.array],
    layer: int,
    component: str = "mlp",
    threshold: float = 0.01,
    top_k: int = 50,
    verbose: bool = False,
) -> CircuitResult:
    """
    Discover a feature-level circuit by ablation.

    Tests top-k SAE features, keeps those with causal effect above threshold.

    Args:
        model: InterpretableModel
        sae: Trained SAE
        text: Input
        layer: Layer
        component: Component
        threshold: Minimum effect to include
        top_k: Features to test
        verbose: Print progress

    Returns:
        CircuitResult with feature nodes
    """
    effects = feature_patching(
        model, sae, text, layer, component,
        top_k=top_k, verbose=verbose,
    )

    # Filter by threshold
    circuit_features = {
        fid: eff for fid, eff in effects.items()
        if abs(eff) >= threshold
    }

    nodes = [f"L{layer}.{component}.f{fid}" for fid in circuit_features]
    edges = []  # Single-layer: no edges between features

    return CircuitResult(
        data={
            "feature_effects": circuit_features,
            "layer": layer,
            "component": component,
            "all_tested": effects,
        },
        metadata={
            "threshold": threshold,
            "top_k": top_k,
        },
        nodes=nodes,
        edges=edges,
        threshold=threshold,
    )
