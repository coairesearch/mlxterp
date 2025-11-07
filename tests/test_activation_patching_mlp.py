#!/usr/bin/env python3
"""
Test activation patching on MLP components to find important layers.

This patches the MLP output (not the entire residual stream) to isolate
which layers' MLPs are most important for the task.
"""

import mlx.core as mx
from mlxterp import InterpretableModel, interventions as iv
from mlx_lm import load

print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print(f"\n{'='*60}")
print("Testing MLP Activation Patching")
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

# Step 2: Get corrupted output (baseline)
print("\nStep 2: Getting corrupted output (unpatched)...")
with model.trace(corrupted_text) as corrupted_trace:
    corrupted_output = model.output.save()

mx.eval(corrupted_output)

baseline_dist = compute_l2_distance(corrupted_output[0, -1], clean_output[0, -1])
mx.eval(baseline_dist)
print(f"  Baseline L2 distance: {float(baseline_dist):.4f}")

# Step 3: Check available MLP keys
print("\nStep 3: Checking MLP activation keys...")
with model.trace(clean_text) as trace:
    pass

mlp_keys = sorted([k for k in trace.activations.keys() if 'mlp' in k.lower() and 'layers' in k])
print(f"  Found {len(mlp_keys)} MLP-related keys")
print(f"  Sample MLP keys:")
for key in mlp_keys[:5]:
    print(f"    {key}: {trace.activations[key].shape}")

# Step 4: Run MLP activation patching
print(f"\n{'='*60}")
print("Step 4: Patching MLP outputs at each layer")
print(f"{'='*60}\n")

results = {}
num_layers = len(model.layers)

for layer_idx in range(num_layers):
    print(f"Testing layer {layer_idx}/{num_layers-1}...", end="\r")

    # Try to get MLP output for this layer
    mlp_key = f"model.model.layers.{layer_idx}.mlp"

    # Get clean MLP activation
    with model.trace(clean_text) as trace:
        if mlp_key in trace.activations:
            clean_mlp = trace.activations[mlp_key]
        else:
            print(f"\n  Warning: {mlp_key} not found, skipping")
            continue

    mx.eval(clean_mlp)

    # Patch MLP into corrupted
    with model.trace(corrupted_text,
                    interventions={f"layers.{layer_idx}.mlp": iv.replace_with(clean_mlp)}):
        patched_out = model.output.save()

    mx.eval(patched_out)

    # Measure distance
    l2_dist = compute_l2_distance(patched_out[0, -1], clean_output[0, -1])
    mx.eval(l2_dist)

    results[layer_idx] = float(l2_dist)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

if not results:
    print("\n✗ No results - MLP keys might have different naming")
    print("\nAll MLP-related keys found:")
    for key in mlp_keys:
        print(f"  {key}")
else:
    # Sort by L2 distance
    sorted_layers = sorted(results.items(), key=lambda x: x[1])

    print(f"\nBaseline: corrupted has L2 = {float(baseline_dist):.4f} from clean")
    print(f"Perfect recovery: L2 = 0.0000")

    print("\nMost important MLP layers (lowest L2 = patching helps most):")
    for layer, dist in sorted_layers[:10]:
        improvement = float(baseline_dist) - dist
        percent = (improvement / float(baseline_dist)) * 100
        print(f"  Layer {layer:2d} MLP: L2 = {dist:8.4f} ({percent:5.1f}% recovery)")

    print("\nLeast important MLP layers (highest L2 = patching helps least):")
    for layer, dist in sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]:
        improvement = float(baseline_dist) - dist
        percent = (improvement / float(baseline_dist)) * 100
        print(f"  Layer {layer:2d} MLP: L2 = {dist:8.4f} ({percent:5.1f}% recovery)")

    print("\n" + "="*60)
    print("✅ MLP activation patching complete!")
    print("="*60)

    print("\nInterpretation:")
    print("  - MLP layers with high % recovery are crucial for this task")
    print("  - These layers likely encode the factual knowledge (Paris vs London)")
    print("  - Lower layers might process syntax, higher layers process semantics")
