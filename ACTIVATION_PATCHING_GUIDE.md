# Activation Patching Guide

This guide explains how to use activation patching correctly with mlxterp.

## What Went Wrong

Your original code had two issues:

### Issue 1: Lambda Closure Bug

```python
# WRONG - all interventions use the same activation
for layer_idx in range(num_layers):
    with model.trace(clean_text):
        clean_act = model.layers[layer_idx].output.save()

    # BUG: lambda captures clean_act by reference!
    with model.trace(corrupted_text,
                    interventions={f"layers.{layer_idx}": lambda x: clean_act}):
        pass
```

**Solution**: Use `iv.replace_with()` instead of lambda:

```python
with model.trace(corrupted_text,
                interventions={f"layers.{layer_idx}": iv.replace_with(clean_act)}):
    pass
```

### Issue 2: Patching the Residual Stream

When you patch `model.layers[i].output`, you're replacing the **entire residual stream** at that position. This affects ALL downstream layers, which is why you got 0.0 L2 distance for every layer.

```python
# WRONG - patches entire residual stream
with model.trace(corrupted_text,
                interventions={f"layers.{layer_idx}": iv.replace_with(clean_act)}):
    pass
```

**Solution**: Patch specific components like MLP or attention:

```python
# CORRECT - patches only the MLP component
with model.trace(corrupted_text,
                interventions={f"layers.{layer_idx}.mlp": iv.replace_with(clean_mlp)}):
    pass
```

## Working Example

```python
import mlx.core as mx
from mlxterp import InterpretableModel, interventions as iv
from mlx_lm import load

# Load model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Define inputs
clean_text = "Paris is the capital of France"
corrupted_text = "London is the capital of France"

# Get baselines
with model.trace(clean_text):
    clean_output = model.output.save()

with model.trace(corrupted_text):
    corrupted_output = model.output.save()

mx.eval(clean_output, corrupted_output)

# Helper
def l2_distance(a, b):
    return float(mx.sqrt(mx.sum((a - b) ** 2)))

baseline = l2_distance(corrupted_output[0, -1], clean_output[0, -1])

# Patch each layer's MLP
for layer_idx in range(len(model.layers)):
    # Get clean MLP activation
    with model.trace(clean_text) as trace:
        clean_mlp = trace.activations[f"model.model.layers.{layer_idx}.mlp"]

    mx.eval(clean_mlp)

    # Patch into corrupted
    with model.trace(corrupted_text,
                    interventions={f"layers.{layer_idx}.mlp": iv.replace_with(clean_mlp)}):
        patched_output = model.output.save()

    mx.eval(patched_output)

    # Measure effect
    dist = l2_distance(patched_output[0, -1], clean_output[0, -1])
    recovery = (baseline - dist) / baseline * 100

    print(f"Layer {layer_idx:2d}: {recovery:6.1f}% recovery")
```

## Understanding the Results

### Layer 0 MLP: 43.1% recovery
- Most important layer!
- Early layers often process fundamental features

### Layers 7-10 MLP: Negative recovery
- Patching makes output WORSE
- These layers likely encode the corruption ("London")
- This is expected and informative!

### Layer 15 MLP: 24.2% recovery
- Final layer is important for output formation

## Available Components to Patch

You can patch different components to test their importance:

```python
# MLP components
f"layers.{i}.mlp"                  # Full MLP output
f"layers.{i}.mlp.gate_proj"       # Gate projection
f"layers.{i}.mlp.up_proj"         # Up projection
f"layers.{i}.mlp.down_proj"       # Down projection

# Attention components
f"layers.{i}.self_attn"           # Full attention output
f"layers.{i}.self_attn.q_proj"    # Query projection
f"layers.{i}.self_attn.k_proj"    # Key projection
f"layers.{i}.self_attn.v_proj"    # Value projection
f"layers.{i}.self_attn.o_proj"    # Output projection

# Layer norms
f"layers.{i}.input_layernorm"     # Pre-attention norm
f"layers.{i}.post_attention_layernorm"  # Pre-MLP norm
```

## Why Not KL Divergence?

Your original code used KL divergence, which gave NaN values. This happened because:

1. KL divergence is sensitive to numerical issues (log(0), etc.)
2. Softmax over 128K vocab can cause overflow/underflow
3. L2 distance is simpler and more robust

If you really need KL divergence, make sure to:
- Normalize logits before softmax
- Add epsilon for numerical stability
- Check for NaN/Inf values

## Complete Working Scripts

Check these files in the tests/ directory:

- `test_activation_patching_mlp.py` - Full MLP patching example
- `test_activation_patching_simple.py` - Simple L2 distance version
- `test_what_we_capture.py` - Debugging tool

And in examples/:

- `activation_patching_example.py` - Clean, simple example for your notebook

## Key Takeaways

✅ **DO**: Use `iv.replace_with()` for interventions
✅ **DO**: Patch specific components (MLP, attention) not full layers
✅ **DO**: Use L2 distance for robustness
✅ **DO**: Expect some layers to have negative recovery

❌ **DON'T**: Use lambda closures in loops
❌ **DON'T**: Patch the entire residual stream (`layers.i.output`)
❌ **DON'T**: Use KL divergence unless you handle numerical issues carefully
