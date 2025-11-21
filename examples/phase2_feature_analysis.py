"""
Phase 2: SAE Feature Analysis Example

This example demonstrates how to use the integrated feature analysis tools
to understand what your trained SAE has learned.

Uses the trained SAE: examples/models/sae_layer23_mlp_10000samples.mlx
"""

from mlxterp import InterpretableModel
from datasets import load_dataset

# =============================================================================
# Setup
# =============================================================================

print("="*80)
print("SAE FEATURE ANALYSIS - Phase 2 Demo")
print("="*80)

# Load model and SAE
print("\n[1/3] Loading model and SAE...")
from mlx_lm import load

mlx_model, tokenizer = load("arogister/Qwen3-8B-ShiningValiant3-mlx-4Bit")
model = InterpretableModel(mlx_model, tokenizer=tokenizer)

# Load your trained SAE
sae_path = "examples/models/sae_layer23_mlp_10000samples.mlx"
print(f"   Loading SAE from: {sae_path}")
sae = model.load_sae(sae_path)
print(f"   ✓ Loaded {sae}")

# Configuration
LAYER = 23
COMPONENT = "mlp"

# =============================================================================
# Example 1: What features activate for specific texts?
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 1: Top Features for Specific Texts")
print("="*80)

test_texts = [
    "The Eiffel Tower is located in Paris, France",
    "Machine learning models learn patterns from data",
    "The mitochondria is the powerhouse of the cell",
]

for text in test_texts:
    print(f"\nText: '{text}'")
    print("-" * 80)

    # Get top 5 features
    top_features = model.get_top_features_for_text(
        text=text,
        sae=sae,
        layer=LAYER,
        component=COMPONENT,
        top_k=5
    )

    print("Top 5 features:")
    for feature_id, activation in top_features:
        print(f"  Feature {feature_id:6d}: {activation:.3f}")

# =============================================================================
# Example 2: What texts activate a specific feature?
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: Top Texts for Specific Features")
print("="*80)

# Load dataset for searching
print("\n[2/3] Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=False)
texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50][:200]  # Use 200 for speed
print(f"   ✓ Loaded {len(texts)} texts")

# Pick a few features to analyze
# (These are features that showed up in the evaluation)
features_to_analyze = [9826, 32318, 24477]

print("\n[3/3] Analyzing features...")
for feature_id in features_to_analyze:
    print(f"\n{'='*80}")
    print(f"Feature #{feature_id}")
    print('='*80)

    # Find top texts
    examples = model.get_top_texts_for_feature(
        feature_id=feature_id,
        sae=sae,
        texts=texts,
        layer=LAYER,
        component=COMPONENT,
        top_k=5
    )

    if not examples:
        print("  (Feature did not activate on any texts)")
        continue

    print(f"\nTop {len(examples)} activating texts:")
    for i, (text, activation, pos) in enumerate(examples, 1):
        # Clean up text for display
        clean_text = ' '.join(text.split())
        print(f"\n  {i}. Activation: {activation:.3f} (token position {pos})")
        print(f"     Text: {clean_text[:150]}...")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print("""
Next Steps:
-----------
1. Try different texts to see which features activate
2. Analyze more features to understand what they represent
3. Look for patterns in the top-activating texts
4. Compare features across different layers

For more details, see:
- docs/guides/sae_evaluation.md
- docs/guides/dictionary_learning.md
""")
