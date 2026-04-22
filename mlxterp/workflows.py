"""
Research workflow primitives for multi-step interpretability investigations.

Pre-built pipelines that chain together analysis tools and return
comprehensive reports. Designed to be called by both humans and agents.
"""

import mlx.core as mx
from typing import Any, Callable, Dict, List, Optional, Union

from .causal.patching import activation_patching
from .causal.dla import direct_logit_attribution
from .causal.attribution import attribution_patching
from .causal.path_patching import path_patching
from .causal.acdc import acdc
from .metrics import get_metric
from .results import AnalysisResult


class WorkflowResult(AnalysisResult):
    """Result from a multi-step workflow.

    Attributes:
        steps: List of (step_name, step_result) tuples for each step executed
        narrative: Human-readable narrative of the investigation
    """

    def __init__(self, **kwargs):
        self.steps = kwargs.pop("steps", [])
        self.narrative = kwargs.pop("narrative", "")
        kwargs.setdefault("result_type", "workflow")
        super().__init__(**kwargs)

    def summary(self) -> str:
        step_names = [name for name, _ in self.steps]
        return f"Workflow ({len(self.steps)} steps: {', '.join(step_names)})"

    def get_step(self, name: str):
        """Get a specific step's result by name."""
        for step_name, step_result in self.steps:
            if step_name == name:
                return step_result
        return None

    def to_markdown(self) -> str:
        lines = [
            f"# {self.data.get('workflow_name', 'Workflow')} Report",
            "",
            self.narrative,
            "",
        ]
        for step_name, step_result in self.steps:
            lines.append(f"## Step: {step_name}")
            lines.append("")
            if hasattr(step_result, "summary"):
                lines.append(step_result.summary())
            elif isinstance(step_result, dict):
                for k, v in step_result.items():
                    lines.append(f"- **{k}**: {v}")
            lines.append("")
        return "\n".join(lines)


def behavior_localization(
    model,
    clean: Union[str, mx.array],
    corrupted: Union[str, mx.array],
    metric: Union[str, Callable] = "l2",
    metric_kwargs: Optional[Dict] = None,
    steps: Optional[List[str]] = None,
    top_k: int = 5,
    verbose: bool = True,
) -> WorkflowResult:
    """
    Localize which components are responsible for a behavior.

    Pipeline: DLA → activation patching (MLP) → activation patching (attention)
    → head-level patching on top layers.

    Args:
        model: InterpretableModel
        clean: Clean input
        corrupted: Corrupted input
        metric: Metric to use
        metric_kwargs: Extra metric args
        steps: Which steps to run. Default: all.
            Options: "dla", "patch_mlp", "patch_attn", "patch_heads"
        top_k: Number of top components to highlight
        verbose: Print progress

    Returns:
        WorkflowResult with per-step results and narrative
    """
    if steps is None:
        steps = ["dla", "patch_mlp", "patch_attn", "patch_heads"]

    metric_kwargs = metric_kwargs or {}
    executed_steps = []
    narrative_parts = []

    # Step 1: DLA
    if "dla" in steps:
        if verbose:
            print("Step 1: Direct Logit Attribution...")
        dla_result = direct_logit_attribution(model, clean)
        executed_steps.append(("dla", dla_result))
        narrative_parts.append(
            f"DLA identifies target token '{dla_result.target_token_str}' "
            f"(id={dla_result.target_token})."
        )

    # Step 2: MLP patching
    if "patch_mlp" in steps:
        if verbose:
            print("Step 2: MLP activation patching...")
        mlp_result = activation_patching(
            model, clean, corrupted,
            component="mlp", metric=metric,
            metric_kwargs=metric_kwargs,
        )
        executed_steps.append(("patch_mlp", mlp_result))
        top_mlp = mlp_result.top_components(k=top_k)
        top_mlp_str = ", ".join(f"L{l}({e:.3f})" for l, e in top_mlp)
        narrative_parts.append(f"Top MLP layers: {top_mlp_str}.")

    # Step 3: Attention patching
    if "patch_attn" in steps:
        if verbose:
            print("Step 3: Attention activation patching...")
        attn_result = activation_patching(
            model, clean, corrupted,
            component="attn", metric=metric,
            metric_kwargs=metric_kwargs,
        )
        executed_steps.append(("patch_attn", attn_result))
        top_attn = attn_result.top_components(k=top_k)
        top_attn_str = ", ".join(f"L{l}({e:.3f})" for l, e in top_attn)
        narrative_parts.append(f"Top attention layers: {top_attn_str}.")

    # Step 4: Head-level patching on most important layers
    if "patch_heads" in steps:
        if verbose:
            print("Step 4: Head-level patching on top layers...")
        # Use top layers from attention patching, or all if not run
        if "patch_attn" in steps:
            top_layers = [l for l, _ in attn_result.top_components(k=3)]
        else:
            top_layers = list(range(min(3, len(model.layers))))

        head_result = activation_patching(
            model, clean, corrupted,
            component="attn_head", metric=metric,
            layers=top_layers,
            metric_kwargs=metric_kwargs,
        )
        executed_steps.append(("patch_heads", head_result))
        top_heads = head_result.top_components(k=top_k)
        top_heads_str = ", ".join(f"idx{i}({e:.3f})" for i, e in top_heads)
        narrative_parts.append(f"Top attention heads: {top_heads_str}.")

    narrative = " ".join(narrative_parts)
    if verbose:
        print(f"\nDone. {narrative}")

    return WorkflowResult(
        data={
            "workflow_name": "Behavior Localization",
            "clean": clean if isinstance(clean, str) else str(type(clean)),
            "corrupted": corrupted if isinstance(corrupted, str) else str(type(corrupted)),
            "metric": metric if isinstance(metric, str) else "custom",
            "top_k": top_k,
        },
        metadata={"n_steps": len(executed_steps)},
        steps=executed_steps,
        narrative=narrative,
    )


def circuit_discovery(
    model,
    clean: Union[str, mx.array],
    corrupted: Union[str, mx.array],
    metric: Union[str, Callable] = "l2",
    metric_kwargs: Optional[Dict] = None,
    threshold: float = 0.01,
    steps: Optional[List[str]] = None,
    verbose: bool = True,
) -> WorkflowResult:
    """
    Discover a circuit for a behavior.

    Pipeline: attribution patching (fast scan) → activation patching (verify)
    → ACDC (full circuit) → path patching (key edges).

    Args:
        model: InterpretableModel
        clean: Clean input
        corrupted: Corrupted input
        metric: Metric
        metric_kwargs: Extra metric args
        threshold: ACDC pruning threshold
        steps: Which steps to run. Default: all.
            Options: "attribution", "activation", "acdc", "paths"
        verbose: Print progress

    Returns:
        WorkflowResult with discovered circuit
    """
    if steps is None:
        steps = ["attribution", "activation", "acdc"]

    metric_kwargs = metric_kwargs or {}
    executed_steps = []
    narrative_parts = []

    # Step 1: Fast attribution scan
    if "attribution" in steps:
        if verbose:
            print("Step 1: Attribution patching (fast scan)...")
        attr_result = attribution_patching(
            model, clean, corrupted,
            component="resid_post", metric=metric,
            metric_kwargs=metric_kwargs,
        )
        executed_steps.append(("attribution", attr_result))
        narrative_parts.append(f"Attribution scan: {attr_result.summary()}.")

    # Step 2: Brute-force verification
    if "activation" in steps:
        if verbose:
            print("Step 2: Activation patching (verification)...")
        act_result = activation_patching(
            model, clean, corrupted,
            component="resid_post", metric=metric,
            metric_kwargs=metric_kwargs,
        )
        executed_steps.append(("activation", act_result))
        top = act_result.top_components(k=3)
        narrative_parts.append(
            f"Top layers confirmed: {', '.join(f'L{l}' for l, _ in top)}."
        )

    # Step 3: ACDC
    if "acdc" in steps:
        if verbose:
            print("Step 3: ACDC circuit discovery...")
        circuit_result = acdc(
            model, clean, corrupted,
            threshold=threshold, metric=metric,
            metric_kwargs=metric_kwargs,
            verbose=verbose,
        )
        executed_steps.append(("acdc", circuit_result))
        narrative_parts.append(f"Circuit found: {circuit_result.summary()}.")

    # Step 4: Path patching on discovered circuit
    if "paths" in steps and "acdc" in steps:
        if verbose:
            print("Step 4: Path patching on circuit edges...")
        path_results = {}
        nodes = circuit_result.nodes
        for i, sender in enumerate(nodes[:5]):  # Limit to top 5 senders
            for receiver in nodes[i + 1:min(i + 4, len(nodes))]:
                pp = path_patching(
                    model, clean, corrupted,
                    sender=sender, receiver=receiver,
                    metric=metric, metric_kwargs=metric_kwargs,
                )
                effect = pp.data["effect"]
                if abs(effect) > threshold:
                    path_results[f"{sender}->{receiver}"] = effect

        executed_steps.append(("paths", path_results))
        narrative_parts.append(
            f"Found {len(path_results)} significant paths."
        )

    narrative = " ".join(narrative_parts)
    if verbose:
        print(f"\nDone. {narrative}")

    return WorkflowResult(
        data={
            "workflow_name": "Circuit Discovery",
            "threshold": threshold,
        },
        metadata={"n_steps": len(executed_steps)},
        steps=executed_steps,
        narrative=narrative,
    )


def feature_investigation(
    model,
    sae,
    text: Union[str, mx.array],
    layer: int,
    component: str = "mlp",
    feature_ids: Optional[List[int]] = None,
    dataset: Optional[List[str]] = None,
    top_k: int = 10,
    verbose: bool = True,
) -> WorkflowResult:
    """
    Investigate SAE features: find top features, ablate them, find max examples.

    Pipeline: encode to find active features → ablation testing →
    max-activating examples (if dataset provided).

    Args:
        model: InterpretableModel
        sae: Trained SAE
        text: Input text to analyze
        layer: Layer the SAE is on
        component: Component
        feature_ids: Specific features to test. None = auto-detect top_k.
        dataset: Texts for max-activating examples. None = skip.
        top_k: Number of top features
        verbose: Print progress

    Returns:
        WorkflowResult with feature analysis
    """
    from .causal.feature_circuits import feature_patching

    executed_steps = []
    narrative_parts = []

    # Step 1: Find active features
    if verbose:
        print("Step 1: Finding active features...")

    with model.trace(text) as trace:
        pass

    from .core.module_resolver import resolve_component
    act_key = resolve_component(component, layer, trace.activations)
    if act_key is None:
        raise ValueError(f"No activation found for {component} at layer {layer}")

    activation = trace.activations[act_key]
    mx.eval(activation)
    act_flat = activation.reshape(-1, activation.shape[-1])
    features = sae.encode(act_flat)
    mx.eval(features)

    # Top features by max activation
    max_acts = mx.max(mx.abs(features), axis=0)
    mx.eval(max_acts)
    top_indices = mx.argsort(max_acts)[::-1][:top_k]
    top_feature_ids = [int(i) for i in top_indices.tolist()]
    top_activations = {fid: float(max_acts[fid]) for fid in top_feature_ids}

    if feature_ids is None:
        feature_ids = top_feature_ids

    executed_steps.append(("active_features", top_activations))
    narrative_parts.append(
        f"Top {top_k} active features at L{layer}.{component}: "
        f"{', '.join(f'f{fid}({act:.2f})' for fid, act in list(top_activations.items())[:5])}."
    )

    # Step 2: Ablation testing
    if verbose:
        print("Step 2: Feature ablation testing...")

    effects = feature_patching(
        model, sae, text, layer, component,
        feature_ids=feature_ids, metric="l2",
    )
    executed_steps.append(("ablation", effects))

    important = {f: e for f, e in effects.items() if abs(e) > 0.01}
    narrative_parts.append(
        f"Ablation found {len(important)} causally important features "
        f"out of {len(feature_ids)} tested."
    )

    # Step 3: Max-activating examples (if dataset provided)
    if dataset is not None and len(dataset) > 0:
        if verbose:
            print("Step 3: Finding max-activating examples...")
        from .visualization.dashboards import max_activating_examples

        examples_by_feature = {}
        for fid in feature_ids[:5]:  # Limit to top 5 for speed
            examples = max_activating_examples(
                model, sae, fid, dataset, layer, component, top_k=3,
            )
            examples_by_feature[fid] = examples

        executed_steps.append(("max_examples", examples_by_feature))
        narrative_parts.append(
            f"Found max-activating examples for {len(examples_by_feature)} features."
        )

    narrative = " ".join(narrative_parts)
    if verbose:
        print(f"\nDone. {narrative}")

    return WorkflowResult(
        data={
            "workflow_name": "Feature Investigation",
            "layer": layer,
            "component": component,
            "feature_ids": feature_ids,
        },
        metadata={"n_steps": len(executed_steps)},
        steps=executed_steps,
        narrative=narrative,
    )
