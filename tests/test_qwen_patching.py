#!/usr/bin/env python3
"""
Test activation patching with Qwen model using different metrics.
"""

from mlxterp import InterpretableModel
from mlx_lm import load

print("Loading Qwen model...")
base_model, tokenizer = load('mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print(f"Model loaded with {len(model.layers)} layers")
print(f"Vocabulary size: {model.vocab_size}")

print("\n" + "="*60)
print("Test 1: MSE metric (most stable for large vocabularies)")
print("="*60)

results_mse = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    layers=[0, 10, 20, 30, 40, 47],  # Test subset first
    metric="mse",
    plot=False
)

print("\nMSE results:")
sorted_mse = sorted(results_mse.items(), key=lambda x: x[1], reverse=True)
for layer_idx, recovery in sorted_mse:
    print(f"  Layer {layer_idx:2d}: {recovery:6.1f}% recovery")

print("\n" + "="*60)
print("Test 2: Cosine metric")
print("="*60)

results_cosine = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    layers=[0, 10, 20, 30, 40, 47],
    metric="cosine",
    plot=False
)

print("\nCosine results:")
sorted_cosine = sorted(results_cosine.items(), key=lambda x: x[1], reverse=True)
for layer_idx, recovery in sorted_cosine:
    print(f"  Layer {layer_idx:2d}: {recovery:6.1f}% recovery")

print("\n" + "="*60)
print("Test 3: L2 metric (with overflow protection)")
print("="*60)

results_l2 = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    layers=[0, 10, 20, 30, 40, 47],
    metric="l2",
    plot=False
)

print("\nL2 results:")
sorted_l2 = sorted(results_l2.items(), key=lambda x: x[1], reverse=True)
for layer_idx, recovery in sorted_l2:
    print(f"  Layer {layer_idx:2d}: {recovery:6.1f}% recovery")

print("\n" + "="*60)
print("✅ All metrics completed successfully!")
print("="*60)

print("\nRecommendation: Use metric='mse' or metric='cosine' for models with large vocabularies (>100k tokens)")
