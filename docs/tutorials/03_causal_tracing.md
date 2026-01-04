# Tutorial 3: Causal Tracing (ROME)

**Paper**: [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) by Meng et al. (NeurIPS 2022)

**Project Page**: [rome.baulab.info](https://rome.baulab.info/)

**Difficulty**: Intermediate | **Time**: 3-4 hours

---

## Learning Objectives

By the end of this tutorial, you will:

1. Understand the causal tracing methodology
2. Know how to corrupt inputs and measure recovery
3. Use mlxterp's `activation_patching()` method
4. Identify which layers store factual knowledge
5. Compare MLP vs. attention contributions

---

## Introduction

### The Big Question: Where is Knowledge Stored?

Language models know many facts: "The Eiffel Tower is in Paris", "Einstein developed relativity", etc. But where is this knowledge stored in the model's weights?

The ROME paper introduces **causal tracing** - a technique to localize where factual associations are stored in transformers.

### The Causal Tracing Method

The key insight is to use **corruption and restoration**:

1. **Clean Run**: Run the model on a factual prompt, observe the correct output
2. **Corrupted Run**: Add noise to the subject embedding, observe incorrect output
3. **Patched Run**: Patch in clean activations at specific layers, see if correct output is restored

!!! info "Core Finding"
    The paper discovered that factual knowledge is primarily stored in **MLP modules at middle layers**, specifically when processing the **subject's last token**.

### Why This Matters

Understanding where knowledge is stored enables:
- **Model editing**: Surgically updating specific facts without retraining
- **Interpretability**: Understanding how transformers store and retrieve information
- **Safety**: Identifying where harmful knowledge might be located

---

## Prerequisites

```python
# Install mlxterp if you haven't already
# pip install mlxterp

from mlxterp import InterpretableModel
import mlx.core as mx
```

---

## Part 1: Manual Causal Tracing

Let's first understand causal tracing by implementing it manually.

### Step 1: Understanding the Setup

```python
from mlxterp import InterpretableModel
from mlxterp import interventions as iv

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Factual prompt: The model should complete with "Paris"
clean_text = "The Eiffel Tower is located in the city of"

# Corrupted prompt: Different subject, should change prediction
corrupted_text = "The Louvre Museum is located in the city of"
```

### Step 2: Get Baseline Outputs

```python
# Helper to get predictions from logits (model.output is logits, not hidden states)
def get_top_from_logits(model, logits, top_k=3):
    """Get top-k predictions from logits."""
    if len(logits.shape) == 3:
        logits = logits[0, -1, :]
    top_indices = mx.argsort(logits)[-top_k:][::-1]
    top_scores = logits[top_indices]
    mx.eval(top_indices, top_scores)
    return [(int(i), float(s)) for i, s in zip(top_indices.tolist(), top_scores.tolist())]

# Clean run - should predict "Paris"
with model.trace(clean_text) as trace:
    clean_output = model.output.save()

# Check prediction (note: model.output is logits, not hidden states)
clean_pred = get_top_from_logits(model, clean_output, top_k=3)
print("Clean prediction:", [(model.token_to_str(t), f"{s:.2f}") for t, s in clean_pred])
```

### Step 3: Corrupted Run

```python
# Corrupted run - may predict differently
with model.trace(corrupted_text) as trace:
    corrupted_output = model.output.save()

corrupted_pred = get_top_from_logits(model, corrupted_output, top_k=3)
print("Corrupted prediction:", [(model.token_to_str(t), f"{s:.2f}") for t, s in corrupted_pred])

# Sanity check: verify corruption changes prediction
if clean_pred[0][0] == corrupted_pred[0][0]:
    print("WARNING: Corruption did not change prediction!")
```

### Step 4: Patch and Measure Recovery

```python
# Get clean activations from middle layer MLP
layer_idx = 8  # Middle layer

with model.trace(clean_text) as trace:
    pass

# Find the MLP module output key (not subcomponents like gate_proj)
mlp_key = None
for key in sorted(trace.activations.keys()):
    if key.endswith(f"layers.{layer_idx}.mlp"):
        mlp_key = key
        break

if mlp_key:
    clean_mlp = trace.activations[mlp_key]

    # Build intervention key (remove model prefixes)
    if mlp_key.startswith("model.model."):
        intervention_key = mlp_key[12:]
    elif mlp_key.startswith("model."):
        intervention_key = mlp_key[6:]
    else:
        intervention_key = mlp_key

    with model.trace(corrupted_text,
                     interventions={intervention_key: iv.replace_with(clean_mlp)}):
        patched_output = model.output.save()

    # Use our helper since model.output is logits
    patched_pred = get_top_from_logits(model, patched_output, top_k=3)
    print(f"Patched (layer {layer_idx} MLP):", [(model.token_to_str(t), f"{s:.2f}") for t, s in patched_pred])
```

---

## Part 2: Using the Built-in Method

mlxterp provides `activation_patching()` which automates causal tracing:

```python
# Automated activation patching
results = model.activation_patching(
    clean_text="The Eiffel Tower is located in the city of",
    corrupted_text="The Louvre Museum is located in the city of",
    component="mlp",  # Patch MLP components
    plot=True         # Visualize results
)

# Get most important layers
sorted_layers = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\nMost important MLP layers for factual recall:")
for layer, recovery in sorted_layers[:5]:
    print(f"  Layer {layer}: {recovery:.1f}% recovery")
```

### Understanding the Results

- **Positive recovery**: Patching this layer helps restore the correct output
- **Negative recovery**: This layer encodes the corruption
- **Near-zero recovery**: This layer isn't relevant for this task

---

## Part 3: Reproducing Paper Findings

### Finding 1: MLP vs. Attention

The paper found that MLPs are more important than attention for factual recall:

```python
# Compare MLP vs. attention contributions
print("MLP contributions:")
mlp_results = model.activation_patching(
    clean_text="The Eiffel Tower is located in the city of",
    corrupted_text="The Louvre Museum is located in the city of",
    component="mlp",
    plot=False
)

print("\nAttention contributions:")
attn_results = model.activation_patching(
    clean_text="The Eiffel Tower is located in the city of",
    corrupted_text="The Louvre Museum is located in the city of",
    component="self_attn",
    plot=False
)

# Compare
print("\nComparison (top 3 layers):")
print("-" * 45)
print("Layer | MLP Recovery | Attn Recovery")
print("-" * 45)

all_layers = sorted(set(mlp_results.keys()) | set(attn_results.keys()))
for layer in all_layers[:10]:
    mlp_rec = mlp_results.get(layer, 0)
    attn_rec = attn_results.get(layer, 0)
    marker = " *" if mlp_rec > attn_rec else ""
    print(f"{layer:5d} | {mlp_rec:10.1f}% | {attn_rec:10.1f}%{marker}")
```

**Expected Finding**: MLPs generally show higher recovery than attention for factual prompts.

!!! note "Model-Specific Results"
    The exact layer numbers will vary by model. The paper used GPT-2 and GPT-J. With Llama models, the pattern should be similar: MLPs at middle layers are most important.

### Finding 2: Middle Layer Concentration

The paper found factual knowledge concentrates in middle layers:

```python
# Analyze layer distribution
import matplotlib.pyplot as plt

results = model.activation_patching(
    clean_text="Albert Einstein developed the theory of",
    corrupted_text="Isaac Newton developed the theory of",
    component="mlp",
    plot=False
)

layers = sorted(results.keys())
recoveries = [results[l] for l in layers]

# Find peak
peak_layer = max(results, key=results.get)
print(f"\nPeak recovery at layer {peak_layer} ({results[peak_layer]:.1f}%)")

# Categorize layers
n_layers = len(layers)
early = layers[:n_layers//3]
middle = layers[n_layers//3:2*n_layers//3]
late = layers[2*n_layers//3:]

early_avg = sum(results[l] for l in early) / len(early)
middle_avg = sum(results[l] for l in middle) / len(middle)
late_avg = sum(results[l] for l in late) / len(late)

print(f"\nAverage recovery by region:")
print(f"  Early layers (0-{early[-1]}):   {early_avg:.1f}%")
print(f"  Middle layers ({middle[0]}-{middle[-1]}): {middle_avg:.1f}%")
print(f"  Late layers ({late[0]}-{late[-1]}):  {late_avg:.1f}%")
```

### Finding 3: Subject Token Importance

The paper emphasizes that patching at the **subject's last token** is most effective.

!!! warning "Simplified Implementation"
    mlxterp's current `activation_patching()` patches the entire sequence. For position-specific patching (as in the full paper), you would need to:

    1. Identify the subject tokens
    2. Create position-masked interventions
    3. Patch only at those positions

    This is an advanced extension covered in the exercises.

---

## Part 4: Paper-Accurate Methodology

The script tutorial includes paper-accurate experiments (6-8) that implement the full ROME methodology.

### Gaussian Noise Corruption

The paper corrupts inputs by adding Gaussian noise to subject token embeddings:

```python
def add_gaussian_noise_to_embedding(model, tokens, noise_std=0.3, subject_range=None):
    """Add Gaussian noise to token embeddings (paper's corruption method)."""
    embed_layer = model._module_resolver.get_embedding_layer()

    # Get embeddings (handle quantized models)
    if hasattr(embed_layer, 'scales'):
        embeddings = mx.dequantize(...)
    else:
        embeddings = embed_layer(tokens)

    # Add noise only to subject positions
    noise = mx.random.normal(embeddings.shape) * noise_std
    if subject_range:
        # Mask noise to only affect subject tokens
        ...
    return embeddings + noise
```

### Position-Specific Patching

The paper patches at the subject's **last token position**, not the entire sequence:

```python
def position_patch(x, clean_activation, subject_last_pos):
    """Patch only at the subject's last position."""
    result = x.copy()
    result = result.at[0, subject_last_pos, :].set(
        clean_activation[0, subject_last_pos, :]
    )
    return result

# Use as intervention
with model.trace(corrupted_text,
                 interventions={"layers.8.mlp": position_patch}):
    output = model.output.save()
```

### Target-Token Probability Recovery

The paper measures recovery using target token probability:

```python
def get_target_token_probability(logits, target_token_id):
    """Get P(target_token) from logits."""
    probs = mx.softmax(logits[0, -1, :])
    return float(probs[target_token_id])

# Recovery metric
recovery = (patched_prob - corrupted_prob) / (clean_prob - corrupted_prob) * 100
```

### Statistical Averaging

Average results over multiple factual prompts for robustness:

```python
factual_prompts = [
    ("The Eiffel Tower is in", "The Statue of Liberty is in"),
    ("The capital of France is", "The capital of Germany is"),
    # ... more prompts
]

layer_recoveries = {i: [] for i in range(n_layers)}
for clean, corrupted in factual_prompts:
    results = model.activation_patching(clean, corrupted, "mlp")
    for layer, recovery in results.items():
        layer_recoveries[layer].append(recovery)

# Compute mean and std for each layer
for layer in layer_recoveries:
    mean = sum(layer_recoveries[layer]) / len(layer_recoveries[layer])
    print(f"Layer {layer}: {mean:.1f}% mean recovery")
```

---

## Part 5: Multiple Factual Prompts

Test across different types of factual knowledge:

```python
test_cases = [
    # (clean, corrupted, expected)
    ("The capital of France is", "The capital of Germany is", "Paris/Berlin"),
    ("The CEO of Tesla is", "The CEO of Apple is", "Musk/Cook"),
    ("Water freezes at", "Water boils at", "0/100"),
    ("Shakespeare wrote", "Dickens wrote", "varies"),
]

print("Factual Knowledge Localization Across Different Facts")
print("=" * 60)

for clean, corrupted, expected in test_cases:
    print(f"\nFact: {clean}...")
    results = model.activation_patching(
        clean_text=clean,
        corrupted_text=corrupted,
        component="mlp",
        plot=False
    )

    # Find peak layer
    if results:
        peak = max(results, key=results.get)
        print(f"  Peak MLP layer: {peak} ({results[peak]:.1f}% recovery)")
```

---

## Part 6: Exercises

### Exercise 1: Layer Output Patching

Compare patching full layer outputs vs. specific components:

```python
# Patch entire layer output (not just MLP or attention)
output_results = model.activation_patching(
    clean_text="The Eiffel Tower is located in the city of",
    corrupted_text="The Louvre Museum is located in the city of",
    component="output",  # Full layer output
    plot=False
)

# Compare with MLP-only
# ... your comparison code ...
```

### Exercise 2: Different Corruption Types

The paper uses embedding noise as corruption. Our simplified approach uses different subjects. Try exploring:

```python
# Different corruption approaches
corruptions = [
    # Subject substitution
    ("The Eiffel Tower is located in", "The Statue of Liberty is located in"),
    # Attribute substitution
    ("The capital of France is", "The capital of France was"),
    # Partial corruption
    ("Paris is the capital of France", "Paris is the capital of Germany"),
]

# Compare how corruption type affects localization
# ... your analysis code ...
```

### Exercise 3: Create a Heatmap

Visualize layer x component importance:

```python
import matplotlib.pyplot as plt
import numpy as np

# Collect results for multiple components
components = ["mlp", "self_attn"]
all_results = {}

for comp in components:
    all_results[comp] = model.activation_patching(
        clean_text="The Eiffel Tower is located in the city of",
        corrupted_text="The Louvre Museum is located in the city of",
        component=comp,
        plot=False
    )

# Create heatmap
# ... your visualization code ...
```

---

## Summary

In this tutorial, you learned:

1. **Causal tracing methodology**: Corrupt inputs, patch clean activations, measure recovery
2. **Key finding**: Factual knowledge is stored in MLPs at middle layers
3. **Using mlxterp**: The `activation_patching()` method automates causal tracing
4. **Model comparison**: MLPs generally contribute more than attention for factual recall
5. **Paper-accurate methodology**: Gaussian noise corruption, position-specific patching, probability recovery

---

## Implementation Approaches

The tutorial provides two corruption methods with important differences:

### Subject Substitution (Simplified)

Replace the subject entity with a different one:

| Type | Prompt | Prediction |
|------|--------|------------|
| Clean | "The **Eiffel Tower** is located in" | Paris |
| Corrupted | "The **Louvre Museum** is located in" | Paris/Lyon |

**Characteristics:**
- Creates a semantically meaningful corrupted input
- The model processes a valid sentence about a different entity
- **Middle-layer peaks correctly observed** (layers 14-17 for GPT-2 XL)
- Good for learning the concepts

**Results with GPT-2 XL (48 layers):**
```
Most important MLP layers:
  Layer 16:   14.4%
  Layer 15:   12.5%
  Layer 14:   11.4%
  Layer 17:   10.5%
```

### Gaussian Noise (Paper Method)

Add noise to subject token embeddings:

```python
noisy_embeddings = embeddings + noise * 0.13  # ~3x embedding std
```

**Characteristics:**
- Creates nonsense embeddings at subject positions
- Very aggressive corruption (often 99%+ probability drop)
- Early-layer peaks observed (layer 0 dominates)
- More faithful to paper but harder to reproduce middle-layer peaks

**Why the difference?**

| Aspect | Subject Substitution | Gaussian Noise |
|--------|---------------------|----------------|
| Corruption strength | Moderate | Very aggressive |
| Semantic validity | Valid sentence | Nonsense embeddings |
| Layer distribution | Middle-layer peak | Early-layer peak |
| Best for | Learning & demos | Paper reproduction |

!!! info "Key Insight"
    Subject substitution shows the paper's expected middle-layer peaks because:

    1. The corruption is **semantically meaningful** - the model understands "Louvre Museum"
    2. The model's factual recall mechanism is **cleanly disrupted**
    3. Patching middle-layer MLPs **restores the correct association**

    Gaussian noise is so aggressive that only early-layer patching helps before the signal is completely destroyed.

### What Can Be Patched?

mlxterp can patch any module in the model:

| Key Pattern | What It Is | Use Case |
|-------------|------------|----------|
| `h.N` or `layers.N` | **Residual stream** (full layer output) | Paper's method |
| `h.N.mlp` | MLP output only | Component analysis |
| `h.N.attn` | Attention output only | Component analysis |
| `h.N.ln_1` | LayerNorm output | Fine-grained analysis |

```python
# Check all available activation keys
with model.trace("Hello world") as trace:
    pass

for key in sorted(trace.activations.keys())[:10]:
    print(f"  {key}: {trace.activations[key].shape}")
```

### Position-Specific Patching

The library supports patching at specific token positions using custom interventions:

```python
def replace_at_position(clean_activation, position):
    """Patch only at a specific token position."""
    def _replace(x):
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)
        mask = (positions == position).reshape(1, seq_len, 1).astype(x.dtype)
        return x * (1 - mask) + clean_activation * mask
    return _replace

# Use with intervention
with model.trace(corrupted_text,
                 interventions={"h.24": replace_at_position(clean_layer, 4)}):
    output = model.output.save()
```

!!! note "Which Approach to Use"
    - **Learning the concepts**: Use subject substitution - it clearly shows middle-layer peaks
    - **MLP vs Attention analysis**: Both methods work well
    - **Paper reproduction**: Use Gaussian noise with careful calibration and statistical averaging

---

## Next Steps

- **Tutorial 4: Steering Vectors** - Control model behavior with activation interventions
- **Tutorial 5: Induction Heads** - Understand pattern completion circuits

---

## References

1. Meng, K., et al. (2022). [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262). NeurIPS 2022.

2. Project page: [rome.baulab.info](https://rome.baulab.info/)

3. Related: Geva et al. (2021). [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913).
