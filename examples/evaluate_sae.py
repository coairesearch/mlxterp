"""
Comprehensive SAE Quality Evaluation Script

This script evaluates a trained SAE on multiple quality metrics:
1. Reconstruction quality (MSE, cosine similarity)
2. Sparsity metrics (L0, dead features)
3. Feature interpretability (top activating samples)
4. Comparison to baseline (identity function)
"""

import mlx.core as mx
from mlxterp import InterpretableModel
from mlxterp.sae import SAE, BatchTopKSAE, BaseSAE
from mlx_lm import load as mlx_load
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple
import json
import os

# =============================================================================
# Configuration
# =============================================================================

SAE_PATH = "examples/models/sae_layer23_mlp_10000samples.mlx"
MODEL_NAME = "arogister/Qwen3-8B-ShiningValiant3-mlx-4Bit"
LAYER = 23
COMPONENT = "mlp"
NUM_TEST_SAMPLES = 500  # Number of texts to evaluate on
NUM_FEATURE_EXAMPLES = 10  # Number of top examples to show per feature


# =============================================================================
# Helper Functions
# =============================================================================

def load_test_dataset(num_samples: int = 500) -> List[str]:
    """Load test dataset (using validation set)."""
    print(f"📦 Loading test dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation", trust_remote_code=False)

    # Filter and prepare
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]
    texts = texts[:num_samples]

    print(f"   ✓ Loaded {len(texts)} test samples")
    return texts


def collect_activations(model, texts: List[str], layer: int, component: str) -> Tuple[mx.array, List[int]]:
    """Collect activations from the model.

    Returns:
        activations: Combined activations (n_tokens, d_model)
        text_indices: Mapping from activation index to text index
    """
    print(f"\n📥 Collecting activations from Layer {layer} {component}...")

    all_activations = []
    text_indices = []

    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"   Processing {i+1}/{len(texts)}...")

        with model.trace(text) as trace:
            pass

        # Find activation key
        for key in trace.activations.keys():
            if f"layers.{layer}" in key and key.endswith(f".{component}"):
                acts = trace.activations[key]
                # Flatten sequence dimension
                flat = acts.reshape(-1, acts.shape[-1])
                all_activations.append(flat)
                # Track which text each activation came from
                text_indices.extend([i] * flat.shape[0])
                break

    combined = mx.concatenate(all_activations, axis=0)
    print(f"   ✓ Collected {combined.shape[0]:,} activation samples (d_model={combined.shape[1]})")
    return combined, text_indices


def compute_reconstruction_metrics(original: mx.array, reconstructed: mx.array) -> Dict[str, float]:
    """Compute reconstruction quality metrics."""
    # MSE
    mse = float(mx.mean((original - reconstructed) ** 2))

    # Cosine similarity
    orig_norm = mx.sqrt(mx.sum(original ** 2, axis=-1, keepdims=True))
    recon_norm = mx.sqrt(mx.sum(reconstructed ** 2, axis=-1, keepdims=True))
    cosine_sim = mx.sum(original * reconstructed, axis=-1) / (orig_norm.squeeze() * recon_norm.squeeze() + 1e-8)
    mean_cosine_sim = float(mx.mean(cosine_sim))

    # Explained variance
    variance_original = float(mx.var(original))
    variance_error = float(mx.var(original - reconstructed))
    explained_variance = 1 - (variance_error / (variance_original + 1e-8))

    return {
        "mse": mse,
        "cosine_similarity": mean_cosine_sim,
        "explained_variance": explained_variance
    }


def compute_sparsity_metrics(features: mx.array, sae) -> Dict[str, float]:
    """Compute sparsity metrics."""
    # L0: Average number of active features
    l0 = float(mx.mean(mx.sum(features != 0, axis=-1)))
    l0_fraction = l0 / sae.d_hidden

    # Feature activation distribution
    feature_counts = mx.sum(features != 0, axis=(0, 1))
    n_samples = features.shape[0]

    # Dead features: Features that never activate
    # NOTE: With large eval sets, this metric is misleading (rare features will eventually activate)
    # Better metric: features that activate very rarely (< 1% of samples)
    dead_features_strict = int(mx.sum(feature_counts == 0))
    rare_threshold = max(10, int(n_samples * 0.01))  # < 1% of samples
    dead_features_practical = int(mx.sum(feature_counts < rare_threshold))
    dead_fraction = dead_features_practical / sae.d_hidden

    return {
        "l0_mean": l0,
        "l0_fraction": l0_fraction,
        "dead_features": dead_features_strict,
        "dead_features_practical": dead_features_practical,
        "dead_fraction": dead_fraction,
        "rare_threshold": rare_threshold,
        "max_feature_activations": int(mx.max(feature_counts)),
        "min_feature_activations": int(mx.min(feature_counts)),
    }


def find_top_activating_examples(
    features: mx.array,
    texts: List[str],
    text_indices: List[int],
    feature_idx: int,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Find texts where a feature activates most strongly."""
    # Get activation values for this feature across all tokens
    feature_acts = features[:, :, feature_idx]  # (n_tokens, seq_len)

    # Get max activation per token
    max_per_token = mx.max(feature_acts, axis=1)  # (n_tokens,)

    # Get top-k token indices
    top_token_indices = mx.argsort(max_per_token)[-top_k:][::-1]

    results = []
    seen_texts = set()
    for token_idx in top_token_indices.tolist():
        activation_value = float(max_per_token[token_idx])
        if activation_value > 0:  # Only include if feature actually activated
            # Map token index back to text index
            text_idx = text_indices[token_idx]
            # Only show each text once
            if text_idx not in seen_texts:
                text_snippet = texts[text_idx][:200]  # First 200 chars
                results.append((text_snippet, activation_value))
                seen_texts.add(text_idx)

    return results


def analyze_feature_interpretability(
    sae,
    features: mx.array,
    texts: List[str],
    text_indices: List[int],
    num_features_to_analyze: int = 10
) -> Dict[int, List[Tuple[str, float]]]:
    """Analyze what concepts different features represent."""
    print(f"\n🔍 Analyzing feature interpretability...")

    # Find most active features (those that activate frequently)
    feature_counts = mx.sum(features != 0, axis=(0, 1))
    top_features = mx.argsort(feature_counts)[-num_features_to_analyze:][::-1]

    feature_examples = {}
    for feat_idx in top_features.tolist():
        examples = find_top_activating_examples(features, texts, text_indices, feat_idx, top_k=5)
        if examples:
            feature_examples[feat_idx] = examples

    print(f"   ✓ Analyzed {len(feature_examples)} features")
    return feature_examples


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_sae(sae_path: str, model_name: str, layer: int, component: str):
    """Run comprehensive SAE evaluation."""

    print("=" * 80)
    print("SAE QUALITY EVALUATION")
    print("=" * 80)
    print(f"\nSAE Path: {sae_path}")
    print(f"Model: {model_name}")
    print(f"Layer: {layer}, Component: {component}")

    # Step 1: Load SAE
    print(f"\n[1/6] Loading SAE...")
    # Load - try BatchTopKSAE first, then SAE
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE file not found: {sae_path}")

    try:
        sae = BatchTopKSAE.load(sae_path)
    except ValueError:
        # Fall back to regular SAE
        sae = SAE.load(sae_path)
    print(f"   ✓ SAE loaded")
    print(f"   Architecture: {sae.__class__.__name__}")
    print(f"   d_model: {sae.d_model}, d_hidden: {sae.d_hidden}")
    print(f"   Expansion: {sae.d_hidden / sae.d_model:.1f}x")
    if hasattr(sae, 'k'):
        print(f"   k (sparsity): {sae.k}")

    # Step 2: Load model
    print(f"\n[2/6] Loading model...")
    mlx_model, tokenizer = mlx_load(model_name)
    model = InterpretableModel(mlx_model, tokenizer=tokenizer)
    print(f"   ✓ Model loaded")

    # Step 3: Load test data
    print(f"\n[3/6] Loading test dataset...")
    test_texts = load_test_dataset(NUM_TEST_SAMPLES)

    # Step 4: Collect activations
    print(f"\n[4/6] Collecting activations...")
    activations, text_indices = collect_activations(model, test_texts, layer, component)

    # Add sequence dimension for SAE (expects 3D)
    activations_3d = activations[:, None, :]  # (batch, 1, d_model)

    # Step 5: Run SAE
    print(f"\n[5/6] Running SAE inference...")
    reconstructed, features = sae(activations_3d)

    # Remove sequence dimension
    reconstructed_2d = reconstructed[:, 0, :]  # (batch, d_model)

    print(f"   ✓ SAE inference complete")
    print(f"   Features shape: {features.shape}")
    print(f"   Reconstructed shape: {reconstructed_2d.shape}")

    # Step 6: Compute metrics
    print(f"\n[6/6] Computing quality metrics...")

    # Reconstruction metrics
    recon_metrics = compute_reconstruction_metrics(activations, reconstructed_2d)

    # Sparsity metrics
    sparsity_metrics = compute_sparsity_metrics(features, sae)

    # Feature interpretability
    feature_examples = analyze_feature_interpretability(
        sae, features, test_texts, text_indices, num_features_to_analyze=NUM_FEATURE_EXAMPLES
    )

    # =============================================================================
    # Print Results
    # =============================================================================

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\n📊 RECONSTRUCTION QUALITY")
    print("-" * 40)
    print(f"  MSE:                 {recon_metrics['mse']:.6f}")
    print(f"  Cosine Similarity:   {recon_metrics['cosine_similarity']:.4f}")
    print(f"  Explained Variance:  {recon_metrics['explained_variance']:.4f}")

    print("\n🎯 SPARSITY METRICS")
    print("-" * 40)
    print(f"  Average L0:          {sparsity_metrics['l0_mean']:.1f} / {sae.d_hidden}")
    print(f"  L0 Fraction:         {sparsity_metrics['l0_fraction']:.4f}")
    print(f"  Dead Features (strict):  {sparsity_metrics['dead_features']:,} / {sae.d_hidden:,}")
    print(f"  Dead Features (< {sparsity_metrics['rare_threshold']} activations): {sparsity_metrics['dead_features_practical']:,} / {sae.d_hidden:,} ({sparsity_metrics['dead_fraction']:.2%})")
    print(f"  Most Active Feature: {sparsity_metrics['max_feature_activations']:,} activations")
    print(f"  Least Active:        {sparsity_metrics['min_feature_activations']:,} activations")

    print("\n🔍 FEATURE INTERPRETABILITY (Sample Features)")
    print("-" * 40)
    for feat_idx, examples in list(feature_examples.items())[:5]:  # Show top 5 features
        print(f"\n  Feature #{feat_idx}:")
        for i, (text, activation) in enumerate(examples[:3], 1):  # Show top 3 examples
            print(f"    {i}. [{activation:.3f}] {text[:100]}...")

    # Quality assessment
    print("\n" + "=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)

    # Scoring
    score = 0
    max_score = 5

    # Reconstruction (0-2 points)
    if recon_metrics['cosine_similarity'] > 0.95:
        score += 2
        print("  ✅ Excellent reconstruction (cosine sim > 0.95)")
    elif recon_metrics['cosine_similarity'] > 0.90:
        score += 1.5
        print("  ✅ Good reconstruction (cosine sim > 0.90)")
    elif recon_metrics['cosine_similarity'] > 0.85:
        score += 1
        print("  ⚠️  Fair reconstruction (cosine sim > 0.85)")
    else:
        print("  ❌ Poor reconstruction (cosine sim < 0.85)")

    # Sparsity (0-2 points)
    if sparsity_metrics['dead_fraction'] < 0.30:
        score += 2
        print("  ✅ Excellent sparsity (< 30% dead features)")
    elif sparsity_metrics['dead_fraction'] < 0.50:
        score += 1.5
        print("  ✅ Good sparsity (< 50% dead features)")
    elif sparsity_metrics['dead_fraction'] < 0.70:
        score += 1
        print("  ⚠️  Fair sparsity (< 70% dead features)")
    else:
        print("  ❌ Poor sparsity (> 70% dead features)")

    # Interpretability (0-1 point)
    if len(feature_examples) >= 8:
        score += 1
        print("  ✅ Good feature diversity")
    elif len(feature_examples) >= 5:
        score += 0.5
        print("  ⚠️  Fair feature diversity")
    else:
        print("  ❌ Poor feature diversity")

    print(f"\n  Overall Score: {score}/{max_score} ({score/max_score*100:.0f}%)")

    if score >= 4.5:
        print("  🎉 EXCELLENT SAE - Ready for production use!")
    elif score >= 3.5:
        print("  ✅ GOOD SAE - Suitable for interpretability research")
    elif score >= 2.5:
        print("  ⚠️  FAIR SAE - May need retraining with different hyperparameters")
    else:
        print("  ❌ POOR SAE - Retraining recommended")

    # Save detailed results
    results = {
        "sae_path": sae_path,
        "model": model_name,
        "layer": layer,
        "component": component,
        "architecture": {
            "type": sae.__class__.__name__,
            "d_model": int(sae.d_model),
            "d_hidden": int(sae.d_hidden),
            "expansion": float(sae.d_hidden / sae.d_model),
            "k": int(sae.k) if hasattr(sae, 'k') else None,
        },
        "reconstruction": recon_metrics,
        "sparsity": sparsity_metrics,
        "quality_score": float(score),
        "max_score": max_score,
    }

    results_path = sae_path.replace(".mlx", "_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {results_path}")

    return results


if __name__ == "__main__":
    evaluate_sae(SAE_PATH, MODEL_NAME, LAYER, COMPONENT)
