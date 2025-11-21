"""
Neuronpedia-Style Feature Visualization

This example demonstrates how to visualize SAE feature activations
in text, similar to Neuronpedia's interface.

The visualization shows tokens colored by their activation strength:
- Blue tokens: Positive activation
- Red tokens: Negative activation
- Darker colors: Stronger activation
"""

from mlxterp import InterpretableModel
from mlxterp.sae import visualize_feature_activations
from mlx_lm import load

# =============================================================================
# Setup
# =============================================================================

print("="*80)
print("NEURONPEDIA-STYLE VISUALIZATION")
print("="*80)

print("\n[1/2] Loading model and SAE...")
mlx_model, tokenizer = load("arogister/Qwen3-8B-ShiningValiant3-mlx-4Bit")
model = InterpretableModel(mlx_model, tokenizer=tokenizer)

sae_path = "examples/models/sae_layer23_mlp_10000samples.mlx"
sae = model.load_sae(sae_path)
print(f"   ✓ Loaded {sae}")

# =============================================================================
# Example 1: Visualize Top Features
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 1: Top 5 Features for Text")
print("="*80)

test_texts = [
    "The Eiffel Tower is located in Paris, France",
    "Machine learning models learn patterns from data",
]

for text in test_texts:
    print(f"\nAnalyzing: '{text}'")
    print("-" * 80)

    # This will show tokens colored by activation strength
    visualize_feature_activations(
        model,
        text,
        sae,
        layer=23,
        component="mlp",
        top_k_features=3  # Show top 3 features
    )

# =============================================================================
# Example 2: Visualize Specific Features
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: Specific Features")
print("="*80)

# Pick features that were active in evaluation
feature_ids = [9826, 32318, 24477]

text = "The city of Paris is known for its landmarks like the Eiffel Tower"
print(f"\nAnalyzing specific features for: '{text}'")
print("-" * 80)

visualize_feature_activations(
    model,
    text,
    sae,
    layer=23,
    component="mlp",
    feature_ids=feature_ids  # Visualize these specific features
)

# =============================================================================
# Example 3: Show Activation Values
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: With Activation Values")
print("="*80)

text = "Python is a programming language used for machine learning"
print(f"\nAnalyzing: '{text}'")
print("-" * 80)

visualize_feature_activations(
    model,
    text,
    sae,
    layer=23,
    component="mlp",
    top_k_features=2,
    show_values=True  # Also show numerical values
)

# =============================================================================
# Example 4: Find Top Activating Tokens for a Feature
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 4: Top Tokens for Feature")
print("="*80)

from mlxterp.sae import get_top_activating_tokens

feature_id = 21780  # A feature that showed up in examples
text = "The Eiffel Tower is a famous landmark in Paris, France, visited by millions"

top_tokens = get_top_activating_tokens(
    model,
    text,
    sae,
    layer=23,
    feature_id=feature_id,
    component="mlp",
    top_k=10
)

print(f"\nFeature {feature_id} - Top activating tokens:")
for token, activation, position in top_tokens:
    print(f"  Position {position:2d}: '{token:15s}' = {activation:.3f}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)

print("""
Color Coding:
- BLUE tokens: Positive feature activation
- RED tokens: Negative feature activation
- Darker/Bold: Stronger activation

Usage Tips:
1. Use top_k_features to explore what features activate
2. Use feature_ids to focus on specific features
3. Use show_values to see exact activation strengths
4. Compare activations across different texts

For more details, see:
- docs/guides/sae_feature_analysis.md
""")
