"""
SAE Feature Analysis Example

This demonstrates Phase 2 capabilities:
1. Finding top-activating features for a text
2. Finding top-activating texts for a feature
3. Feature steering (ablation, amplification)
4. Feature-based interventions during tracing

This example shows how to use SAEs for interpretability research.
"""

import mlx.core as mx
from mlxterp import InterpretableModel
from mlxterp.sae import SAE, BatchTopKSAE
from typing import List, Tuple, Dict
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

SAE_PATH = "examples/models/sae_layer23_mlp_10000samples.mlx"
MODEL_NAME = "arogister/Qwen3-8B-ShiningValiant3-mlx-4Bit"
LAYER = 23
COMPONENT = "mlp"


# =============================================================================
# Feature Analysis Functions
# =============================================================================

def get_top_activating_features(
    sae,  # SAE or BatchTopKSAE
    model: InterpretableModel,
    text: str,
    layer: int,
    component: str,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find which features activate most strongly for a given text.

    Args:
        sae: Trained SAE
        model: InterpretableModel instance
        text: Input text to analyze
        layer: Layer number
        component: Component name (e.g., "mlp", "attn")
        top_k: Number of top features to return

    Returns:
        List of (feature_id, max_activation) tuples
    """
    # Trace model
    with model.trace(text) as trace:
        pass

    # Get activations
    activation_key = None
    for key in trace.activations.keys():
        if f"layers.{layer}" in key and key.endswith(f".{component}"):
            activation_key = key
            break

    if not activation_key:
        raise ValueError(f"Could not find activations for layer {layer}, component {component}")

    acts = trace.activations[activation_key]  # (batch, seq_len, d_model)

    # Add seq dimension if needed
    if len(acts.shape) == 2:
        acts = acts[:, None, :]

    # Run through SAE
    _, features = sae(acts)  # (batch, seq_len, d_hidden)

    # Get max activation per feature across all tokens
    max_activations = mx.max(mx.abs(features), axis=(0, 1))  # (d_hidden,)

    # Get top-k
    top_indices = mx.argsort(max_activations)[-top_k:][::-1]

    results = []
    for idx in top_indices.tolist():
        activation_value = float(max_activations[idx])
        if activation_value > 0:  # Only include if feature activated
            results.append((idx, activation_value))

    return results


def get_top_activating_texts(
    sae,  # SAE or BatchTopKSAE
    model: InterpretableModel,
    feature_id: int,
    texts: List[str],
    layer: int,
    component: str,
    top_k: int = 10
) -> List[Tuple[str, float, int]]:
    """
    Find texts where a specific feature activates most strongly.

    Args:
        sae: Trained SAE
        model: InterpretableModel instance
        feature_id: Feature to analyze
        texts: List of texts to search
        layer: Layer number
        component: Component name
        top_k: Number of top examples to return

    Returns:
        List of (text, max_activation, token_position) tuples
    """
    activations = []

    # Find activation key once
    with model.trace(texts[0]) as trace:
        pass

    activation_key = None
    for key in trace.activations.keys():
        if f"layers.{layer}" in key and key.endswith(f".{component}"):
            activation_key = key
            break

    if not activation_key:
        raise ValueError(f"Could not find activations for layer {layer}, component {component}")

    # Collect activations for this feature across all texts
    print(f"Analyzing {len(texts)} texts for feature {feature_id}...")

    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(texts)}...")

        with model.trace(text) as trace:
            pass

        acts = trace.activations[activation_key]

        # Add seq dimension if needed
        if len(acts.shape) == 2:
            acts = acts[:, None, :]

        # Run through SAE
        _, features = sae(acts)  # (batch, seq_len, d_hidden)

        # Get this feature's activations
        feature_acts = features[:, :, feature_id]  # (batch, seq_len)

        # Get max activation and position
        max_val = float(mx.max(feature_acts))
        max_pos = int(mx.argmax(feature_acts.reshape(-1)))

        activations.append((text, max_val, max_pos))

    # Sort by activation strength
    activations.sort(key=lambda x: x[1], reverse=True)

    return activations[:top_k]


def ablate_sae_feature_in_trace(
    sae,  # SAE or BatchTopKSAE
    model: InterpretableModel,
    text: str,
    layer: int,
    component: str,
    feature_id: int
) -> str:
    """
    Ablate (zero out) a specific SAE feature and see how output changes.

    Args:
        sae: Trained SAE
        model: InterpretableModel instance
        text: Input text
        layer: Layer number
        component: Component name
        feature_id: Feature to ablate

    Returns:
        Generated text with feature ablated
    """
    # First get clean output
    with model.trace(text) as clean_trace:
        clean_output = model.output.save()

    # Find activation key
    activation_key = None
    for key in clean_trace.activations.keys():
        if f"layers.{layer}" in key and key.endswith(f".{component}"):
            activation_key = key
            break

    # Now ablate the feature
    with model.trace(text) as trace:
        # Get original activations
        acts = trace.activations[activation_key]

        # Add seq dimension if needed
        if len(acts.shape) == 2:
            acts = acts[:, None, :]

        # Run through SAE
        reconstructed, features = sae(acts)

        # Ablate feature
        features_ablated = features.copy()
        features_ablated[:, :, feature_id] = 0.0

        # Reconstruct with ablated features
        acts_ablated = sae.decode(features_ablated)

        # Replace activations
        # This is tricky - we need to integrate this with the intervention system
        # For now, we'll just return the features for analysis

    print(f"Feature {feature_id} ablated")
    print(f"Original max activation: {float(mx.max(features[:, :, feature_id])):.4f}")
    print(f"Ablated activation: {float(mx.max(features_ablated[:, :, feature_id])):.4f}")

    return features, features_ablated


# =============================================================================
# Example Usage
# =============================================================================

def main():
    print("=" * 80)
    print("SAE FEATURE ANALYSIS")
    print("=" * 80)

    # Load SAE
    print(f"\n[1/4] Loading SAE from {SAE_PATH}...")
    try:
        sae = BatchTopKSAE.load(SAE_PATH)
    except ValueError:
        sae = SAE.load(SAE_PATH)
    print(f"   ✓ Loaded {sae.__class__.__name__}")
    print(f"   Architecture: {sae.d_model} → {sae.d_hidden} ({sae.d_hidden/sae.d_model:.1f}x)")

    # Load model
    print(f"\n[2/4] Loading model...")
    model = InterpretableModel(MODEL_NAME)
    print(f"   ✓ Model loaded")

    # Example 1: Find top features for a text
    print(f"\n[3/4] Finding top-activating features...")
    print("-" * 80)

    example_texts = [
        "The capital of France is Paris, which is known for the Eiffel Tower.",
        "Machine learning models can learn patterns from data automatically.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for text in example_texts:
        print(f"\nText: \"{text}\"")
        top_features = get_top_activating_features(
            sae, model, text, LAYER, COMPONENT, top_k=5
        )

        print(f"Top 5 activating features:")
        for feat_id, activation in top_features:
            print(f"  Feature {feat_id:6d}: {activation:.4f}")

    # Example 2: Find top texts for a feature
    print(f"\n[4/4] Finding top-activating texts for specific features...")
    print("-" * 80)

    # Load some test texts
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=False)
    test_texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50][:200]

    # Pick a few interesting features from the first example
    if len(example_texts) > 0:
        first_text_features = get_top_activating_features(
            sae, model, example_texts[0], LAYER, COMPONENT, top_k=3
        )

        for feat_id, _ in first_text_features[:2]:  # Analyze top 2 features
            print(f"\n\nFeature {feat_id}:")
            print(f"Finding top activating examples from {len(test_texts)} texts...")

            top_texts = get_top_activating_texts(
                sae, model, feat_id, test_texts, LAYER, COMPONENT, top_k=5
            )

            print(f"\nTop 5 activating texts:")
            for i, (text, activation, pos) in enumerate(top_texts, 1):
                text_preview = text[:100].replace("\n", " ")
                print(f"  {i}. [{activation:.4f}] {text_preview}...")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nThis demonstrates Phase 2 capabilities:")
    print("  ✅ Find top-activating features for any text")
    print("  ✅ Find top-activating texts for any feature")
    print("  ⬜ Feature steering (ablation/amplification) - TODO: Full integration")
    print("\nNext steps:")
    print("  - Integrate feature steering with intervention system")
    print("  - Add visualization dashboard")
    print("  - Add feature clustering and similarity analysis")


if __name__ == "__main__":
    main()
