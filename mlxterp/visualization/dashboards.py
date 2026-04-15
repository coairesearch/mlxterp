"""
Feature dashboards for SAE analysis.

Generates standalone HTML dashboards with max-activating examples,
feature activation distributions, and logit weight analysis.
"""

import mlx.core as mx
import json
from typing import Any, Dict, List, Optional, Tuple, Union


def max_activating_examples(
    model,
    sae,
    feature_id: int,
    texts: List[str],
    layer: int,
    component: str = "mlp",
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Find texts that maximally activate a specific SAE feature.

    Args:
        model: InterpretableModel
        sae: Trained SAE
        feature_id: Feature index to analyze
        texts: Dataset of texts to search
        layer: Layer the SAE was trained on
        component: Component (mlp, attn, etc.)
        top_k: Number of top examples to return

    Returns:
        List of dicts with text, activation_value, token_position
    """
    from ..core.module_resolver import resolve_component

    results = []

    for text in texts:
        with model.trace(text) as trace:
            pass

        act_key = resolve_component(component, layer, trace.activations)
        if act_key is None:
            continue

        activation = trace.activations[act_key]
        mx.eval(activation)

        # Encode through SAE
        act_flat = activation.reshape(-1, activation.shape[-1])
        features = sae.encode(act_flat)
        mx.eval(features)

        # Get activation for this feature
        feat_activations = features[:, feature_id]  # (seq_len,)
        mx.eval(feat_activations)

        max_val = float(mx.max(feat_activations))
        max_pos = int(mx.argmax(feat_activations))

        results.append({
            "text": text if isinstance(text, str) else str(text),
            "activation_value": max_val,
            "token_position": max_pos,
        })

    # Sort by activation value
    results.sort(key=lambda x: x["activation_value"], reverse=True)
    return results[:top_k]


def feature_activation_histogram(
    model,
    sae,
    feature_id: int,
    texts: List[str],
    layer: int,
    component: str = "mlp",
    n_bins: int = 50,
) -> Dict[str, Any]:
    """
    Compute activation distribution for a feature across a dataset.

    Args:
        model: InterpretableModel
        sae: Trained SAE
        feature_id: Feature to analyze
        texts: Dataset texts
        layer: Layer
        component: Component
        n_bins: Histogram bins

    Returns:
        Dict with bin_edges, counts, mean, std, sparsity
    """
    from ..core.module_resolver import resolve_component

    all_activations = []

    for text in texts:
        with model.trace(text) as trace:
            pass

        act_key = resolve_component(component, layer, trace.activations)
        if act_key is None:
            continue

        activation = trace.activations[act_key]
        mx.eval(activation)

        act_flat = activation.reshape(-1, activation.shape[-1])
        features = sae.encode(act_flat)
        mx.eval(features)

        feat_vals = features[:, feature_id].tolist()
        all_activations.extend(feat_vals)

    if not all_activations:
        return {"bin_edges": [], "counts": [], "mean": 0.0, "std": 0.0, "sparsity": 1.0}

    values = mx.array(all_activations)
    mx.eval(values)

    mean = float(mx.mean(values))
    std = float(mx.sqrt(mx.mean((values - mean) ** 2)))
    sparsity = float(mx.mean((mx.abs(values) < 1e-6).astype(mx.float32)))

    # Compute histogram
    min_val = float(mx.min(values))
    max_val = float(mx.max(values))

    if max_val - min_val < 1e-10:
        return {
            "bin_edges": [min_val, max_val],
            "counts": [len(all_activations)],
            "mean": mean,
            "std": std,
            "sparsity": sparsity,
        }

    bin_width = (max_val - min_val) / n_bins
    bin_edges = [min_val + i * bin_width for i in range(n_bins + 1)]
    counts = [0] * n_bins

    for v in all_activations:
        bin_idx = min(int((v - min_val) / bin_width), n_bins - 1)
        counts[bin_idx] += 1

    return {
        "bin_edges": bin_edges,
        "counts": counts,
        "mean": mean,
        "std": std,
        "sparsity": sparsity,
    }


def generate_feature_dashboard_html(
    feature_id: int,
    examples: List[Dict[str, Any]],
    histogram: Dict[str, Any],
    title: Optional[str] = None,
) -> str:
    """
    Generate a standalone HTML dashboard for a single SAE feature.

    Args:
        feature_id: Feature index
        examples: From max_activating_examples()
        histogram: From feature_activation_histogram()
        title: Optional title override

    Returns:
        HTML string
    """
    title = title or f"Feature {feature_id} Dashboard"

    examples_html = ""
    for i, ex in enumerate(examples):
        examples_html += f"""
        <div class="example">
            <div class="rank">#{i+1}</div>
            <div class="text">{_html_escape(ex['text'][:200])}</div>
            <div class="score">Activation: {ex['activation_value']:.4f} (pos {ex['token_position']})</div>
        </div>
        """

    stats_html = f"""
    <div class="stats">
        <div class="stat"><span class="label">Mean:</span> {histogram.get('mean', 0):.4f}</div>
        <div class="stat"><span class="label">Std:</span> {histogram.get('std', 0):.4f}</div>
        <div class="stat"><span class="label">Sparsity:</span> {histogram.get('sparsity', 0):.1%}</div>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{_html_escape(title)}</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #333; }}
.stats {{ display: flex; gap: 20px; margin: 20px 0; }}
.stat {{ background: #f5f5f5; padding: 10px 16px; border-radius: 8px; }}
.stat .label {{ font-weight: 600; }}
.example {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin: 8px 0; }}
.example .rank {{ font-weight: 700; color: #666; font-size: 0.9em; }}
.example .text {{ margin: 4px 0; font-family: monospace; white-space: pre-wrap; }}
.example .score {{ color: #0066cc; font-size: 0.9em; }}
h2 {{ margin-top: 30px; }}
</style>
</head>
<body>
<h1>{_html_escape(title)}</h1>
<h2>Statistics</h2>
{stats_html}
<h2>Max-Activating Examples</h2>
{examples_html}
<p style="color:#999; font-size:0.8em;">Generated by mlxterp</p>
</body>
</html>"""

    return html


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
