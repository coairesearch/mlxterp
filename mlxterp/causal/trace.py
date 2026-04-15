"""
CausalTrace: Declarative context manager for causal experiments.

Provides a clean API for clean/corrupted paired analysis with
declarative patching and metric computation.
"""

import mlx.core as mx
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.intervention import replace_with, replace_at_positions
from ..core.module_resolver import resolve_component, resolve_intervention_key
from ..metrics import get_metric


class CausalTrace:
    """
    Context manager for causal interpretability experiments.

    Runs a clean forward pass on entry, caches all activations,
    then provides declarative patching for corrupted runs.

    Example:
        with model.causal_trace("The Eiffel Tower is in", "The Colosseum is in") as ct:
            ct.patch("layers.5.mlp")
            ct.patch("layers.7.self_attn", positions=[3, 4])
            effect = ct.metric(logit_diff, correct_token=123, incorrect_token=456)

    Attributes:
        clean_activations: Dict of all activations from clean forward pass
        clean_output: Model output from clean forward pass
        corrupted_input: The corrupted input (for running patched passes)
    """

    def __init__(self, model, clean_input, corrupted_input):
        self._model = model
        self._clean_input = clean_input
        self._corrupted_input = corrupted_input
        self._patches = []
        self.clean_activations = {}
        self.clean_output = None
        self.corrupted_output = None

    def __enter__(self):
        # Run clean forward pass and cache all activations
        with self._model.trace(self._clean_input) as clean_trace:
            self.clean_output = self._model.output.save()

        self.clean_activations = dict(clean_trace.activations)
        mx.eval(self.clean_output)
        # Evaluate all activations
        for key in self.clean_activations:
            mx.eval(self.clean_activations[key])

        # Run corrupted forward pass for baseline
        with self._model.trace(self._corrupted_input) as corrupted_trace:
            self.corrupted_output = self._model.output.save()

        mx.eval(self.corrupted_output)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up references
        self._patches.clear()
        return False

    def patch(
        self,
        component: str,
        positions: Optional[List[int]] = None,
        layer: Optional[int] = None,
    ):
        """
        Register a component to be patched from clean into corrupted.

        Args:
            component: Component path or canonical name. Examples:
                - "layers.5.mlp" — specific layer's MLP
                - "layers.5.self_attn" — specific layer's attention
                - Or use layer param with canonical: "mlp", "attn"
            positions: Optional token positions to patch. None = all.
            layer: Optional layer index (used with canonical component names)
        """
        self._patches.append({
            "component": component,
            "positions": positions,
            "layer": layer,
        })

    def metric(
        self,
        metric_fn: Union[str, Callable],
        **kwargs,
    ) -> float:
        """
        Build interventions from registered patches, run corrupted forward
        pass with patches applied, and compute the metric.

        Args:
            metric_fn: Metric function or name string
            **kwargs: Additional args passed to metric (e.g., correct_token)

        Returns:
            Metric value (higher = more recovery of clean behavior)
        """
        metric_fn = get_metric(metric_fn)

        # Build intervention dict from patches
        interventions = self._build_interventions()

        # Run corrupted pass with patches
        with self._model.trace(self._corrupted_input, interventions=interventions):
            patched_output = self._model.output.save()

        mx.eval(patched_output)

        # Extract last-token logits
        def _last_logits(output):
            if output.ndim == 3:
                return output[0:1, -1:, :].reshape(1, -1)
            return output

        return metric_fn(
            _last_logits(patched_output),
            _last_logits(self.clean_output),
            _last_logits(self.corrupted_output),
            **kwargs,
        )

    def _build_interventions(self) -> Dict[str, Callable]:
        """Convert registered patches to an intervention dict."""
        interventions = {}

        for patch in self._patches:
            component = patch["component"]
            positions = patch["positions"]
            layer = patch["layer"]

            # Resolve the activation key
            act_key = self._resolve_patch_key(component, layer)
            if act_key is None:
                raise ValueError(
                    f"Could not find activation for '{component}' "
                    f"(layer={layer}) in clean activations. "
                    f"Available keys: {list(self.clean_activations.keys())[:10]}..."
                )

            clean_act = self.clean_activations[act_key]
            intervention_key = resolve_intervention_key(act_key)

            if positions is not None:
                interventions[intervention_key] = replace_at_positions(clean_act, positions)
            else:
                interventions[intervention_key] = replace_with(clean_act)

        return interventions

    def _resolve_patch_key(
        self, component: str, layer: Optional[int]
    ) -> Optional[str]:
        """Resolve a patch specification to an activation key."""
        # If layer is provided with a canonical name
        if layer is not None:
            return resolve_component(component, layer, self.clean_activations)

        # If component is a full path like "layers.5.mlp"
        # Try to find it directly or with prefixes
        for prefix in ["model.model.", "model.", ""]:
            full_key = f"{prefix}{component}"
            if full_key in self.clean_activations:
                return full_key

        # Try parsing "layers.X.component" format
        import re
        match = re.match(r"layers\.(\d+)\.(.+)", component)
        if match:
            layer_idx = int(match.group(1))
            comp = match.group(2)
            return resolve_component(comp, layer_idx, self.clean_activations)

        return None

    def get_clean_activation(self, component: str, layer: Optional[int] = None) -> mx.array:
        """
        Get a specific clean activation.

        Args:
            component: Component path or canonical name
            layer: Optional layer index

        Returns:
            The clean activation tensor
        """
        key = self._resolve_patch_key(component, layer)
        if key is None:
            raise KeyError(
                f"No clean activation found for '{component}' (layer={layer})"
            )
        return self.clean_activations[key]
