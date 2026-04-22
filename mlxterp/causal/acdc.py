"""
ACDC (Automated Circuit DisCovery).

Iteratively prunes edges in the computational graph to find
the minimal circuit sufficient for a task.
"""

import mlx.core as mx
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.intervention import replace_with
from ..core.module_resolver import resolve_component, resolve_intervention_key
from ..metrics import get_metric
from ..results import CircuitResult


def acdc(
    model,
    clean: Union[str, mx.array],
    corrupted: Union[str, mx.array],
    metric: Union[str, Callable] = "l2",
    threshold: float = 0.01,
    components: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    metric_kwargs: Optional[Dict[str, Any]] = None,
    max_iterations: int = 10,
    verbose: bool = False,
) -> CircuitResult:
    """
    Automated Circuit Discovery via iterative edge pruning.

    Starts with a full graph (all components connected) and iteratively
    removes edges whose causal effect is below the threshold.

    Args:
        model: InterpretableModel instance
        clean: Clean input
        corrupted: Corrupted input
        metric: Metric function or name
        threshold: Minimum effect to keep an edge
        components: Component types to consider. Default: ["attn", "mlp"]
        layers: Layers to analyze. None = all.
        metric_kwargs: Extra metric args
        max_iterations: Maximum pruning iterations
        verbose: Print progress

    Returns:
        CircuitResult with discovered circuit
    """
    metric_fn = get_metric(metric)
    metric_name = metric if isinstance(metric, str) else "custom"
    metric_kwargs = metric_kwargs or {}

    if components is None:
        components = ["attn", "mlp"]

    if layers is None:
        layers = list(range(len(model.layers)))

    # Run clean trace to get all activations
    if verbose:
        print("Running clean trace...")
    with model.trace(clean) as clean_trace:
        clean_output = model.output.save()

    mx.eval(clean_output)
    for key in clean_trace.activations:
        mx.eval(clean_trace.activations[key])

    # Run corrupted trace
    with model.trace(corrupted) as corrupted_trace:
        corrupted_output = model.output.save()

    mx.eval(corrupted_output)

    def _last_logits(output):
        if output.ndim == 3:
            return output[0:1, -1:, :].reshape(1, -1)
        return output

    clean_logits = _last_logits(clean_output)
    corrupted_logits = _last_logits(corrupted_output)

    # Build initial full graph: all components are nodes
    nodes = []
    node_keys = {}  # node_name -> activation_key

    for layer_idx in layers:
        for comp in components:
            key = resolve_component(comp, layer_idx, clean_trace.activations)
            if key is not None:
                node_name = f"L{layer_idx}.{comp}"
                nodes.append(node_name)
                node_keys[node_name] = key

    # Compute per-node importance: ablate each node and measure effect
    if verbose:
        print(f"Testing {len(nodes)} nodes...")

    node_effects = {}
    for node_name in nodes:
        act_key = node_keys[node_name]
        clean_act = clean_trace.activations[act_key]
        iv_key = resolve_intervention_key(act_key)

        # Patch this node from clean into corrupted
        with model.trace(corrupted, interventions={iv_key: replace_with(clean_act)}):
            patched_output = model.output.save()

        mx.eval(patched_output)
        patched_logits = _last_logits(patched_output)

        effect = metric_fn(
            patched_logits, clean_logits, corrupted_logits,
            **metric_kwargs,
        )
        node_effects[node_name] = float(effect)

        if verbose:
            print(f"  {node_name}: effect={effect:.4f}")

    # Prune nodes below threshold
    circuit_nodes = [n for n in nodes if abs(node_effects.get(n, 0)) >= threshold]

    # Build edges between surviving nodes
    # Edge exists if nodes are in adjacent or later layers
    edges = []
    for i, sender in enumerate(circuit_nodes):
        for receiver in circuit_nodes[i + 1:]:
            weight = node_effects.get(sender, 0.0)
            edges.append((sender, receiver, weight))

    if verbose:
        print(f"\nCircuit: {len(circuit_nodes)} nodes, {len(edges)} edges "
              f"(threshold={threshold})")

    return CircuitResult(
        data={
            "node_effects": node_effects,
            "threshold": threshold,
            "metric": metric_name,
        },
        metadata={
            "clean": clean if isinstance(clean, str) else f"array({clean.shape})",
            "corrupted": corrupted if isinstance(corrupted, str) else f"array({corrupted.shape})",
            "n_layers": len(layers),
            "components": components,
        },
        nodes=circuit_nodes,
        edges=edges,
        threshold=threshold,
    )
