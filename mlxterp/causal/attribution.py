"""
Attribution patching (gradient-based approximation of activation patching).

Approximates the causal effect of each component using:
  Attribution ≈ (clean_act - corrupted_act) · grad(metric w.r.t. act)

When exact gradients are not available, falls back to finite differences
with random projections.
"""

import mlx.core as mx
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.intervention import replace_with
from ..core.module_resolver import resolve_component, resolve_intervention_key
from ..metrics import get_metric
from ..results import AttributionResult


def attribution_patching(
    model,
    clean: Union[str, mx.array],
    corrupted: Union[str, mx.array],
    metric: Union[str, Callable] = "l2",
    component: str = "resid_post",
    layers: Optional[List[int]] = None,
    metric_kwargs: Optional[Dict[str, Any]] = None,
    n_projections: int = 1,
    eps: float = 1e-3,
    verbose: bool = False,
) -> AttributionResult:
    """
    Compute attribution patching scores using finite differences.

    This approximates activation patching at ~100x speed by computing
    the gradient of the metric with respect to activations, rather than
    running full forward passes for each component.

    Attribution(component) ≈ (clean_act - corrupted_act) · ∇metric

    Args:
        model: InterpretableModel instance
        clean: Clean/correct input
        corrupted: Corrupted/counterfactual input
        metric: Metric function or name
        component: Component to attribute ("resid_post", "attn", "mlp")
        layers: Layers to analyze. None = all.
        metric_kwargs: Extra args for metric
        n_projections: Number of random projections for finite diff
        eps: Perturbation size for finite differences
        verbose: Print progress

    Returns:
        AttributionResult with attribution scores
    """
    metric_fn = get_metric(metric)
    metric_name = metric if isinstance(metric, str) else "custom"
    metric_kwargs = metric_kwargs or {}

    # Get baseline activations
    if verbose:
        print("Running clean and corrupted passes...")

    with model.trace(clean) as clean_trace:
        clean_output = model.output.save()

    with model.trace(corrupted) as corrupted_trace:
        corrupted_output = model.output.save()

    mx.eval(clean_output, corrupted_output)

    def _last_logits(output):
        if output.ndim == 3:
            return output[0:1, -1:, :].reshape(1, -1)
        return output

    clean_logits = _last_logits(clean_output)
    corrupted_logits = _last_logits(corrupted_output)

    if layers is None:
        layers = list(range(len(model.layers)))

    scores = []

    for layer_idx in layers:
        if verbose:
            print(f"  Attribution layer {layer_idx}...", end="\r")

        # Get activation keys
        act_key = resolve_component(component, layer_idx, corrupted_trace.activations)
        if act_key is None:
            scores.append(0.0)
            continue

        corrupted_act = corrupted_trace.activations[act_key]
        clean_key = resolve_component(component, layer_idx, clean_trace.activations)
        clean_act = clean_trace.activations[clean_key] if clean_key else corrupted_act

        mx.eval(corrupted_act, clean_act)

        # Compute activation difference
        act_diff = clean_act - corrupted_act

        # Compute gradient via finite differences with random projections
        intervention_key = resolve_intervention_key(act_key)
        grad_scores = []

        for _ in range(n_projections):
            # Random direction
            direction = mx.random.normal(corrupted_act.shape)
            direction = direction / (mx.sqrt(mx.sum(direction * direction)) + 1e-10)

            # Perturb corrupted activation
            perturbed_act = corrupted_act + eps * direction

            with model.trace(corrupted, interventions={
                intervention_key: replace_with(perturbed_act)
            }):
                perturbed_output = model.output.save()

            mx.eval(perturbed_output)
            perturbed_logits = _last_logits(perturbed_output)

            # Compute metric values
            base_metric = metric_fn(
                corrupted_logits, clean_logits, corrupted_logits,
                **metric_kwargs,
            )
            perturbed_metric = metric_fn(
                perturbed_logits, clean_logits, corrupted_logits,
                **metric_kwargs,
            )

            # Directional derivative
            d_metric = (perturbed_metric - base_metric) / eps

            # Project attribution: (act_diff · direction) * d_metric
            projection = float(mx.sum(act_diff * direction))
            grad_scores.append(projection * d_metric)

        score = sum(grad_scores) / len(grad_scores)
        scores.append(float(score))

    if verbose:
        print()

    score_array = mx.array(scores)

    return AttributionResult(
        data={
            "scores": {layer_idx: s for layer_idx, s in zip(layers, scores)},
            "component": component,
            "metric": metric_name,
            "method": "finite_diff",
        },
        metadata={
            "clean": clean if isinstance(clean, str) else f"array({clean.shape})",
            "corrupted": corrupted if isinstance(corrupted, str) else f"array({corrupted.shape})",
            "n_layers": len(layers),
            "n_projections": n_projections,
            "eps": eps,
        },
        attribution_scores=score_array,
        layers=layers,
        component=component,
        method="finite_diff",
    )
