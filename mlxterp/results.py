"""
Structured analysis result types for mlxterp.

Every analysis method returns an AnalysisResult subclass, providing both
machine-readable data (for agent consumption) and human-readable summaries.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx


def _array_to_list(obj):
    """Convert mx.array and nested structures to JSON-serializable form."""
    if isinstance(obj, mx.array):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _array_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_array_to_list(v) for v in obj]
    return obj


@dataclass
class AnalysisResult:
    """Base class for all analysis results.

    Attributes:
        data: Structured result data (JSON-serializable after conversion)
        metadata: Analysis parameters, timing, model info
        result_type: String tag identifying the result type
    """

    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    result_type: str = "analysis"

    def summary(self) -> str:
        """Human-readable one-line summary of the result."""
        return f"{self.result_type}: {len(self.data)} entries"

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        payload = {
            "result_type": self.result_type,
            "data": _array_to_list(self.data),
            "metadata": _array_to_list(self.metadata),
            "summary": self.summary(),
        }
        return json.dumps(payload, indent=indent, default=str)

    def to_markdown(self) -> str:
        """Human-readable markdown report."""
        lines = [
            f"# {self.result_type.replace('_', ' ').title()}",
            "",
            f"**Summary**: {self.summary()}",
            "",
        ]
        if self.metadata:
            lines.append("## Metadata")
            for k, v in self.metadata.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")
        return "\n".join(lines)

    def plot(self, **kwargs):
        """Generate visualization. Override in subclasses."""
        raise NotImplementedError(
            f"{self.result_type} does not support plotting. "
            "Override plot() in the subclass."
        )


@dataclass
class PatchingResult(AnalysisResult):
    """Result from activation patching experiments.

    Attributes:
        effect_matrix: Array of causal effects, shape depends on patching type:
            - Layer-level: (n_layers,)
            - Position-level: (n_layers, n_positions)
            - Head-level: (n_layers, n_heads)
        layers: List of layer indices tested
        component: Component that was patched
        metric_name: Name of the metric used
    """

    effect_matrix: Optional[Any] = None  # mx.array or list
    layers: List[int] = field(default_factory=list)
    component: str = ""
    metric_name: str = ""
    result_type: str = "patching"

    def summary(self) -> str:
        if self.effect_matrix is None:
            return "Patching: no results"
        effects = self.effect_matrix
        if isinstance(effects, mx.array):
            effects_list = effects.tolist()
        else:
            effects_list = effects

        if isinstance(effects_list, list) and len(effects_list) > 0:
            if isinstance(effects_list[0], list):
                # 2D: find max across all positions/heads
                flat = [v for row in effects_list for v in row]
            else:
                flat = effects_list
            if flat:
                max_val = max(flat)
                max_idx = flat.index(max_val)
                return (
                    f"Patching {self.component} ({self.metric_name}): "
                    f"max effect {max_val:.4f} at index {max_idx}, "
                    f"{len(self.layers)} layers tested"
                )
        return f"Patching {self.component}: {len(self.layers)} layers tested"

    def top_components(self, k: int = 5) -> List[Tuple[int, float]]:
        """Return top-k layers/components by effect size."""
        if self.effect_matrix is None:
            return []
        effects = self.effect_matrix
        if isinstance(effects, mx.array):
            if effects.ndim == 1:
                values = effects.tolist()
            else:
                # For 2D, use max across second dim
                values = mx.max(effects, axis=1).tolist()
        else:
            values = effects if isinstance(effects, list) else list(effects)

        indexed = [(self.layers[i] if i < len(self.layers) else i, v)
                   for i, v in enumerate(values)]
        indexed.sort(key=lambda x: abs(x[1]), reverse=True)
        return indexed[:k]

    def plot(self, **kwargs):
        """Plot patching heatmap."""
        from .visualization.patching import plot_patching_result
        return plot_patching_result(self, **kwargs)


@dataclass
class AttributionResult(AnalysisResult):
    """Result from attribution patching (gradient-based approximation).

    Attributes:
        attribution_scores: Array of attribution values
        layers: Layers analyzed
        component: Component analyzed
        method: Attribution method used (gradient, finite_diff)
    """

    attribution_scores: Optional[Any] = None
    layers: List[int] = field(default_factory=list)
    component: str = ""
    method: str = ""
    result_type: str = "attribution"

    def summary(self) -> str:
        if self.attribution_scores is None:
            return "Attribution: no results"
        scores = self.attribution_scores
        if isinstance(scores, mx.array):
            max_val = float(mx.max(mx.abs(scores)))
        else:
            flat = scores if not isinstance(scores[0], list) else [v for r in scores for v in r]
            max_val = max(abs(v) for v in flat) if flat else 0.0
        return (
            f"Attribution ({self.method}, {self.component}): "
            f"max |score| = {max_val:.4f}, {len(self.layers)} layers"
        )


@dataclass
class DLAResult(AnalysisResult):
    """Result from Direct Logit Attribution.

    Attributes:
        head_contributions: Per-head contribution to target logit (n_layers, n_heads)
        mlp_contributions: Per-MLP contribution to target logit (n_layers,)
        target_token: Token ID being attributed
        target_token_str: String representation of target token
    """

    head_contributions: Optional[Any] = None
    mlp_contributions: Optional[Any] = None
    target_token: Optional[int] = None
    target_token_str: str = ""
    result_type: str = "dla"

    def summary(self) -> str:
        parts = []
        if self.head_contributions is not None:
            hc = self.head_contributions
            if isinstance(hc, mx.array):
                max_val = float(mx.max(mx.abs(hc)))
                max_idx = int(mx.argmax(mx.abs(hc.reshape(-1))))
            else:
                flat = [v for r in hc for v in r] if isinstance(hc[0], list) else hc
                max_val = max(abs(v) for v in flat)
                max_idx = flat.index(max(flat, key=abs))
            parts.append(f"max head contribution {max_val:.4f} at index {max_idx}")
        if self.target_token_str:
            parts.append(f"target='{self.target_token_str}'")
        return f"DLA: {', '.join(parts)}" if parts else "DLA: no results"


@dataclass
class GenerationResult(AnalysisResult):
    """Result from text generation.

    Attributes:
        text: Generated text string
        tokens: List of generated token IDs
        token_logits: Per-token logit distributions (optional)
        prompt: Original prompt
    """

    text: str = ""
    tokens: List[int] = field(default_factory=list)
    token_logits: Optional[Any] = None
    prompt: str = ""
    result_type: str = "generation"

    def summary(self) -> str:
        return f"Generated {len(self.tokens)} tokens: '{self.text[:80]}{'...' if len(self.text) > 80 else ''}'"


@dataclass
class ConversationResult(AnalysisResult):
    """Result from conversation-level analysis.

    Attributes:
        turns: List of turn metadata dicts
        cross_turn_attention: Turn x turn attention matrix (optional)
    """

    turns: List[Dict[str, Any]] = field(default_factory=list)
    cross_turn_attention: Optional[Any] = None
    result_type: str = "conversation"

    def summary(self) -> str:
        roles = [t.get("role", "?") for t in self.turns]
        return f"Conversation: {len(self.turns)} turns ({', '.join(roles)})"


@dataclass
class CircuitResult(AnalysisResult):
    """Result from circuit discovery (ACDC, path patching).

    Attributes:
        nodes: List of component names in the circuit
        edges: List of (sender, receiver, weight) tuples
        threshold: Pruning threshold used
    """

    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str, float]] = field(default_factory=list)
    threshold: float = 0.0
    result_type: str = "circuit"

    def summary(self) -> str:
        return f"Circuit: {len(self.nodes)} nodes, {len(self.edges)} edges (threshold={self.threshold:.4f})"

    def to_graph(self) -> Dict[str, Any]:
        """Return graph representation suitable for visualization."""
        return {
            "nodes": [{"id": n} for n in self.nodes],
            "edges": [{"source": s, "target": t, "weight": w} for s, t, w in self.edges],
        }
