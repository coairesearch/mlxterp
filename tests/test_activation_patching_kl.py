#!/usr/bin/env python3
"""
Test activation patching with KL divergence to find important layers.

This test implements a classic interpretability technique:
- Patch clean activations into corrupted input at each layer
- Measure how close the output gets to clean output (using KL divergence)
- Layers with lowest KL are most important for the task
"""

import mlx.core as mx
from mlxterp import InterpretableModel, interventions as iv
from mlx_lm import load

print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print(f"\n{'='*60}")
print("Testing Activation Patching with KL Divergence")
print(f"{'='*60}\n")


def compute_kl_divergence(logits_p, logits_q, epsilon=1e-8):
    """KL divergence with numerical stability

    Args:
        logits_p: Raw logits (not probabilities)
        logits_q: Raw logits (not probabilities)
    """
    # Normalize logits to prevent overflow in softmax
    logits_p = logits_p - mx.max(logits_p)
    logits_q = logits_q - mx.max(logits_q)

    # Convert logits to probabilities
    p = mx.softmax(logits_p, axis=-1)
    q = mx.softmax(logits_q, axis=-1)

    # Evaluate to check for issues
    mx.eval(p, q)

    # Check for NaN or inf
    if mx.any(mx.isnan(p)) or mx.any(mx.isnan(q)) or mx.any(mx.isinf(p)) or mx.any(mx.isinf(q)):
        return mx.array(float('nan'))

    # Add epsilon and renormalize
    p = p + epsilon
    q = q + epsilon
    p = p / mx.sum(p)
    q = q / mx.sum(q)

    # Compute KL divergence: KL(P || Q) = sum(P * log(P/Q))
    log_ratio = mx.log(p) - mx.log(q)
    kl = mx.sum(p * log_ratio)

    mx.eval(kl)

    return kl


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
print("\nStep 2: Getting corrupted output...")
with model.trace(corrupted_text) as corrupted_trace:
    corrupted_output = model.output.save()

mx.eval(corrupted_output)
print(f"  Corrupted output shape: {corrupted_output.shape}")

# Calculate baseline KL (corrupted vs clean)
print("\n  Checking if outputs are actually different...")
diff = mx.sum(mx.abs(corrupted_output[0, -1] - clean_output[0, -1]))
mx.eval(diff)
print(f"  L1 difference between outputs: {float(diff):.4f}")

if float(diff) < 0.01:
    print("  WARNING: Outputs are nearly identical! The texts might produce the same result.")
    print("  This is expected for short prompts where the difference is late in the sequence.")
else:
    print("  ✓ Outputs are different")

baseline_kl = compute_kl_divergence(corrupted_output[0, -1], clean_output[0, -1])
mx.eval(baseline_kl)
print(f"  Baseline KL (corrupted vs clean): {float(baseline_kl):.4f}")

# Step 3: Test intervention naming
print("\nStep 3: Testing intervention key format...")
with model.trace(clean_text):
    test_act = model.layers[0].output.save()

mx.eval(test_act)

# Try different naming patterns
test_names = [
    "layers.0",
    "model.layers.0",
    "model.model.layers.0"
]

correct_prefix = None
for name in test_names:
    try:
        with model.trace(corrupted_text, interventions={name: iv.replace_with(test_act)}):
            test_out = model.output.save()
        mx.eval(test_out)
        print(f"  ✓ '{name}' works!")
        correct_prefix = name.rsplit('.', 1)[0]
        break
    except Exception as e:
        print(f"  ✗ '{name}' failed")

if correct_prefix is None:
    # Fall back to checking activation keys
    print("\n  Checking activation keys to find correct pattern...")
    with model.trace(clean_text) as trace:
        pass

    layer_keys = [k for k in trace.activations.keys() if 'layers.0' in k and not 'self_attn' in k and not 'mlp' in k]
    print(f"  Found layer 0 keys: {layer_keys[:3]}")

    if layer_keys:
        correct_prefix = layer_keys[0].rsplit('.', 1)[0]
        print(f"  Using prefix: '{correct_prefix}'")
    else:
        raise ValueError("Cannot determine layer naming pattern")

# Step 4: Run activation patching across all layers
print(f"\n{'='*60}")
print("Step 4: Patching activations at each layer")
print(f"{'='*60}\n")

results = {}
num_layers = min(len(model.layers), 16)  # Test first 16 layers for speed

for layer_idx in range(num_layers):
    print(f"Testing layer {layer_idx}/{num_layers-1}...", end="\r")

    # Get clean activation at this layer
    with model.trace(clean_text):
        clean_act = model.layers[layer_idx].output.save()

    mx.eval(clean_act)

    # Patch into corrupted
    intervention_key = f"{correct_prefix}.{layer_idx}"

    with model.trace(corrupted_text,
                    interventions={intervention_key: iv.replace_with(clean_act)}):
        patched_out = model.output.save()

    mx.eval(patched_out)

    # Measure effect - lower KL means patching this layer makes output closer to clean
    kl_div = compute_kl_divergence(patched_out[0, -1], clean_output[0, -1])
    mx.eval(kl_div)

    kl_value = float(kl_div)
    results[layer_idx] = kl_value

    # Debug first few iterations
    if layer_idx < 3:
        print(f"\nLayer {layer_idx}: KL = {kl_value:.4f}")
        if mx.isnan(kl_div):
            print(f"  WARNING: NaN detected!")
            print(f"  Patched output range: [{float(mx.min(patched_out)):.4f}, {float(mx.max(patched_out)):.4f}]")
            print(f"  Clean output range: [{float(mx.min(clean_output)):.4f}, {float(mx.max(clean_output)):.4f}]")

print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Filter out NaN values
valid_results = {k: v for k, v in results.items() if not (isinstance(v, float) and v != v)}

if not valid_results:
    print("\n✗ All results are NaN - there's a numerical issue")
    print("\nDebugging information:")
    print(f"  Baseline KL: {float(baseline_kl)}")
    print(f"  Clean output stats: min={float(mx.min(clean_output)):.4f}, max={float(mx.max(clean_output)):.4f}")
    print(f"  Corrupted output stats: min={float(mx.min(corrupted_output)):.4f}, max={float(mx.max(corrupted_output)):.4f}")
else:
    print(f"\n✓ Got {len(valid_results)} valid results")

    # Sort by KL divergence (lowest = most important)
    sorted_layers = sorted(valid_results.items(), key=lambda x: x[1])

    print(f"\nBaseline: corrupted text has KL = {float(baseline_kl):.4f} from clean")
    print("\nMost important layers (lowest KL = patching makes output closest to clean):")
    for layer, kl in sorted_layers[:5]:
        improvement = float(baseline_kl) - kl
        print(f"  Layer {layer:2d}: KL = {kl:.4f} (improvement: {improvement:.4f})")

    print("\nLeast important layers (highest KL = patching has little effect):")
    for layer, kl in sorted(valid_results.items(), key=lambda x: x[1], reverse=True)[:5]:
        improvement = float(baseline_kl) - kl
        print(f"  Layer {layer:2d}: KL = {kl:.4f} (improvement: {improvement:.4f})")

    print("\n" + "="*60)
    print("✅ Activation patching test complete!")
    print("="*60)
