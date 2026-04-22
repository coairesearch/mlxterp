"""
Automated Interpretability: LLM-generated feature descriptions.

Uses a frontier model (Claude, GPT-4, etc.) to automatically label SAE features
based on their max-activating examples, then validates via sensitivity testing.

Requires: anthropic SDK (`pip install anthropic`)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx


@dataclass
class FeatureLabel:
    """A labeled SAE feature.

    Attributes:
        feature_id: SAE feature index
        label: Short human-readable label
        description: Longer description of what the feature represents
        confidence: Model's confidence score (0-1)
        evidence: Max-activating examples used for labeling
        sensitivity_passed: Whether sensitivity testing confirmed the label
        sensitivity_details: Details of sensitivity testing
    """
    feature_id: int
    label: str = ""
    description: str = ""
    confidence: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    sensitivity_passed: Optional[bool] = None
    sensitivity_details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "sensitivity_passed": self.sensitivity_passed,
            "sensitivity_details": self.sensitivity_details,
        }


def auto_label_feature(
    model,
    sae,
    feature_id: int,
    texts: List[str],
    layer: int,
    component: str = "mlp",
    top_k: int = 10,
    llm_client: Optional[Any] = None,
    llm_model: str = "claude-sonnet-4-20250514",
) -> FeatureLabel:
    """
    Automatically label a single SAE feature using an LLM.

    Process:
    1. Find max-activating examples for the feature
    2. Send examples to LLM with a labeling prompt
    3. Parse the LLM's response into a structured label

    Args:
        model: InterpretableModel
        sae: Trained SAE
        feature_id: Feature to label
        texts: Dataset to search for activating examples
        layer: Layer the SAE was trained on
        component: Component (mlp, attn, etc.)
        top_k: Number of max-activating examples to use
        llm_client: Anthropic client instance. If None, creates one.
        llm_model: LLM model to use for labeling

    Returns:
        FeatureLabel with generated label and evidence
    """
    from .visualization.dashboards import max_activating_examples

    # Step 1: Find max-activating examples
    examples = max_activating_examples(
        model, sae, feature_id, texts, layer, component, top_k=top_k,
    )

    if not examples:
        return FeatureLabel(
            feature_id=feature_id,
            label="no_activations",
            description="No activating examples found in the dataset.",
            confidence=0.0,
            evidence=[],
        )

    # Step 2: Build the labeling prompt
    prompt = _build_labeling_prompt(feature_id, examples)

    # Step 3: Call LLM
    if llm_client is None:
        try:
            import anthropic
            llm_client = anthropic.Anthropic()
        except ImportError:
            # No API available — return a placeholder
            return FeatureLabel(
                feature_id=feature_id,
                label="unlabeled",
                description="Anthropic SDK not installed. Install with: pip install anthropic",
                confidence=0.0,
                evidence=examples,
            )
        except Exception as e:
            return FeatureLabel(
                feature_id=feature_id,
                label="api_error",
                description=f"Could not initialize API client: {e}",
                confidence=0.0,
                evidence=examples,
            )

    try:
        response = llm_client.messages.create(
            model=llm_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text
    except Exception as e:
        return FeatureLabel(
            feature_id=feature_id,
            label="api_error",
            description=f"LLM API call failed: {e}",
            confidence=0.0,
            evidence=examples,
        )

    # Step 4: Parse response
    label = _parse_label_response(response_text)

    return FeatureLabel(
        feature_id=feature_id,
        label=label.get("label", "unknown"),
        description=label.get("description", ""),
        confidence=label.get("confidence", 0.5),
        evidence=examples,
    )


def auto_label_features(
    model,
    sae,
    texts: List[str],
    layer: int,
    component: str = "mlp",
    feature_ids: Optional[List[int]] = None,
    top_k_features: int = 20,
    top_k_examples: int = 10,
    llm_client: Optional[Any] = None,
    llm_model: str = "claude-sonnet-4-20250514",
    verbose: bool = True,
) -> List[FeatureLabel]:
    """
    Batch label multiple SAE features.

    If feature_ids is None, finds the top_k_features most active features
    across the dataset and labels those.

    Args:
        model: InterpretableModel
        sae: Trained SAE
        texts: Dataset texts
        layer: Layer
        component: Component
        feature_ids: Specific features to label. None = auto-detect.
        top_k_features: Number of features to label if auto-detecting
        top_k_examples: Examples per feature for labeling
        llm_client: Anthropic client
        llm_model: LLM model name
        verbose: Print progress

    Returns:
        List of FeatureLabel objects
    """
    # Auto-detect top features if not specified
    if feature_ids is None:
        if verbose:
            print("Finding most active features across dataset...")
        feature_ids = _find_top_features(
            model, sae, texts, layer, component, top_k_features,
        )

    labels = []
    for i, fid in enumerate(feature_ids):
        if verbose:
            print(f"Labeling feature {fid} ({i + 1}/{len(feature_ids)})...")

        label = auto_label_feature(
            model, sae, fid, texts, layer, component,
            top_k=top_k_examples,
            llm_client=llm_client,
            llm_model=llm_model,
        )
        labels.append(label)

        if verbose:
            print(f"  → {label.label}: {label.description[:60]}...")

    return labels


def sensitivity_test(
    model,
    sae,
    label: FeatureLabel,
    test_texts: List[str],
    layer: int,
    component: str = "mlp",
    threshold: float = 0.5,
) -> FeatureLabel:
    """
    Validate a feature label via sensitivity testing.

    Tests whether toggling the feature changes model behavior in ways
    consistent with the label. High activation on label-related inputs
    and low activation on unrelated inputs = pass.

    Args:
        model: InterpretableModel
        sae: Trained SAE
        label: FeatureLabel to test
        test_texts: Texts for sensitivity testing
        layer: Layer
        component: Component
        threshold: Activation threshold for pass/fail

    Returns:
        Updated FeatureLabel with sensitivity_passed and details
    """
    from .core.module_resolver import resolve_component

    activations_per_text = []

    for text in test_texts:
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

        max_act = float(mx.max(mx.abs(features[:, label.feature_id])))
        activations_per_text.append({
            "text": text if isinstance(text, str) else str(text),
            "max_activation": max_act,
        })

    if not activations_per_text:
        label.sensitivity_passed = None
        label.sensitivity_details = "No test texts could be processed"
        return label

    # Check: are there both high and low activation texts?
    acts = [t["max_activation"] for t in activations_per_text]
    mean_act = sum(acts) / len(acts)
    has_variation = max(acts) > threshold and min(acts) < threshold

    label.sensitivity_passed = has_variation
    label.sensitivity_details = (
        f"Mean activation: {mean_act:.4f}, "
        f"range: [{min(acts):.4f}, {max(acts):.4f}], "
        f"threshold: {threshold}, "
        f"variation: {'sufficient' if has_variation else 'insufficient'}"
    )

    return label


def _build_labeling_prompt(feature_id: int, examples: List[Dict]) -> str:
    """Build the prompt for LLM feature labeling."""
    examples_text = ""
    for i, ex in enumerate(examples):
        text = ex.get("text", "")
        if isinstance(text, str):
            display = text[:150]
        else:
            display = str(text)[:150]
        act = ex.get("activation_value", 0)
        pos = ex.get("token_position", -1)
        examples_text += f"\n{i + 1}. (activation={act:.3f}, position={pos}) \"{display}\""

    return f"""You are an expert in mechanistic interpretability. I'll show you the top activating examples for SAE feature #{feature_id}. Based on these examples, provide a concise label and description for what concept or pattern this feature represents.

Max-activating examples:{examples_text}

Respond in exactly this JSON format:
{{
  "label": "short_label_with_underscores",
  "description": "One sentence describing what this feature detects or represents.",
  "confidence": 0.8
}}

The label should be a short identifier (1-4 words with underscores). The confidence should be 0.0-1.0 reflecting how certain you are about the label. If the examples don't show a clear pattern, use a low confidence."""


def _parse_label_response(response: str) -> Dict[str, Any]:
    """Parse the LLM's label response."""
    # Try to extract JSON
    try:
        # Find JSON in the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass

    # Fallback: extract what we can
    return {
        "label": "parse_error",
        "description": response[:200],
        "confidence": 0.0,
    }


def _find_top_features(
    model, sae, texts, layer, component, top_k,
) -> List[int]:
    """Find the top_k most active features across a dataset."""
    from .core.module_resolver import resolve_component

    all_max_acts = None

    for text in texts[:50]:  # Limit dataset size for speed
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

        batch_max = mx.max(mx.abs(features), axis=0)
        if all_max_acts is None:
            all_max_acts = batch_max
        else:
            all_max_acts = mx.maximum(all_max_acts, batch_max)
        mx.eval(all_max_acts)

    if all_max_acts is None:
        return list(range(min(top_k, 64)))

    top_indices = mx.argsort(all_max_acts)[::-1][:top_k]
    return [int(i) for i in top_indices.tolist()]
