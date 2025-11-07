#!/usr/bin/env python3
"""
Test activation patching with L2 distance to find important layers.

Simpler and more robust than KL divergence - measures Euclidean distance
between output logits when patching clean activations into corrupted input.
"""

import mlx.core as mx
from mlxterp import InterpretableModel, interventions as iv
from mlx_lm import load

print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print(f"\n{'='*60}")
print("Testing Activation Patching with L2 Distance")
print(f"{'='*60}\n")


def compute_l2_distance(a, b):
    """L2 (Euclidean) distance between two vectors"""
    diff = a - b
    return mx.sqrt(mx.sum(diff * diff))


# Test texts
clean_text = "Paris is the capital of France"
corrupted_text = "London is the capital of France"

print(f"Clean text: '{clean_text}'")
print(f"Corrupted text: '{corrupted_text}'\n")

# Step 1: Get clean output (baseline)
print("Step 1: Getting clean output baseline...")
with model.trace(clean_text) as clean_trace:
    clean_output = model.output.save()

mx.eval(clean_output)
print(f"  Clean output shape: {clean_output.shape}")

# Step 2: Get corrupted output (to compare)
print("\nStep 2: Getting corrupted output (unpatched baseline)...")
with model.trace(corrupted_text) as corrupted_trace:
    corrupted_output = model.output.save()

mx.eval(corrupted_output)
print(f"  Corrupted output shape: {corrupted_output.shape}")

# Calculate baseline distance (corrupted vs clean)
baseline_dist = compute_l2_distance(corrupted_output[0, -1], clean_output[0, -1])
mx.eval(baseline_dist)
print(f"  Baseline L2 distance (corrupted vs clean): {float(baseline_dist):.4f}")

# Step 3: Run activation patching across layers
print(f"\n{'='*60}")
print("Step 3: Patching activations at each layer")
print(f"{'='*60}\n")

results = {}
num_layers = len(model.layers)

for layer_idx in range(num_layers):
    print(f"Testing layer {layer_idx}/{num_layers-1}...", end="\r")

    # Get clean activation at this layer
    with model.trace(clean_text):
        clean_act = model.layers[layer_idx].output.save()

    mx.eval(clean_act)

    # Patch into corrupted
    with model.trace(corrupted_text,
                    interventions={f"layers.{layer_idx}": iv.replace_with(clean_act)}):
        patched_out = model.output.save()

    mx.eval(patched_out)

    # Measure L2 distance to clean output (lower = patching helps more)
    l2_dist = compute_l2_distance(patched_out[0, -1], clean_output[0, -1])
    mx.eval(l2_dist)

    results[layer_idx] = float(l2_dist)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Sort by L2 distance (lowest = most important)
sorted_layers = sorted(results.items(), key=lambda x: x[1])

print(f"\nBaseline: corrupted text has L2 distance = {float(baseline_dist):.4f} from clean")
print("\nMost important layers (lowest L2 = patching makes output closest to clean):")
for layer, dist in sorted_layers[:10]:
    improvement = float(baseline_dist) - dist
    percent_improvement = (improvement / float(baseline_dist)) * 100
    print(f"  Layer {layer:2d}: L2 = {dist:8.4f} (improvement: {improvement:7.4f}, {percent_improvement:5.1f}%)")

print("\nLeast important layers (highest L2 = patching has little/negative effect):")
for layer, dist in sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]:
    improvement = float(baseline_dist) - dist
    percent_improvement = (improvement / float(baseline_dist)) * 100
    print(f"  Layer {layer:2d}: L2 = {dist:8.4f} (improvement: {improvement:7.4f}, {percent_improvement:5.1f}%)")

print("\n" + "="*60)
print("✅ Activation patching test complete!")
print("="*60)

# Interpretation guide
print("\nHow to interpret these results:")
print("  - Lower L2 distance = patching that layer makes output closer to clean")
print("  - Positive improvement = layer is important for the task")
print("  - Negative improvement = patching makes things worse (layer encodes the corruption)")
print("  - Layers with highest positive improvement are most critical for correct output")
