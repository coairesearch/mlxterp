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

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

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

## Part 4: Multiple Factual Prompts

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

## Part 5: Exercises

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

---

## Limitations and Caveats

!!! warning "Differences from Paper"
    Our implementation simplifies the original paper's methodology:

    - **Corruption method**: Paper uses Gaussian noise on embeddings; we use subject substitution
    - **Position specificity**: Paper patches at subject's last token; we patch entire sequence
    - **Recovery metric**: Paper measures target-token probability restoration; we use L2 distance on logits
    - **Statistical validation**: Paper averages over many examples; demos use single examples

    For rigorous scientific conclusions, implement the full paper methodology.

---

## Next Steps

- **Tutorial 4: Steering Vectors** - Control model behavior with activation interventions
- **Tutorial 5: Induction Heads** - Understand pattern completion circuits

---

## References

1. Meng, K., et al. (2022). [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262). NeurIPS 2022.

2. Project page: [rome.baulab.info](https://rome.baulab.info/)

3. Related: Geva et al. (2021). [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913).
