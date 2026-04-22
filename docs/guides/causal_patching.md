# Causal Patching Guide

## Overview

Causal patching (also called activation patching or interchange intervention) is the most widely used technique in mechanistic interpretability. It answers: **which model components are causally responsible for a specific behavior?**

The idea: run the model on a clean input and a corrupted input. Then re-run the corrupted input, but *patch* in the clean activation at a specific component. If the model recovers the clean behavior, that component was causally important.

## Quick Start

```python
from mlxterp import InterpretableModel
from mlxterp.causal import activation_patching

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Which layers' MLPs are important for factual recall?
result = activation_patching(
    model,
    clean="The Eiffel Tower is in Paris",
    corrupted="The Colosseum is in Paris",
    component="mlp",
    metric="l2",
)

result.plot()  # Bar chart of per-layer effects
print(result.top_components(k=5))
```

## Components You Can Patch

| Component | Description | Result Shape |
|-----------|-------------|-------------|
| `"resid_post"` | Full layer output (residual stream) | `(n_layers,)` |
| `"attn"` | Attention module output | `(n_layers,)` |
| `"mlp"` | MLP/feed-forward output | `(n_layers,)` |
| `"attn_head"` | Individual attention heads | `(n_layers, n_heads)` |

## Choosing a Metric

| Metric | Best For | Notes |
|--------|----------|-------|
| `"l2"` | General purpose | Normalized recovery, 0-1 scale |
| `"logit_diff"` | IOI, factual recall | Requires `correct_token` and `incorrect_token` |
| `"kl"` | Distribution comparison | Negative KL (higher = better) |
| `"cosine"` | Direction-sensitive tasks | Good for large vocabularies |
| `"ce_diff"` | Loss-based evaluation | Positive = patching reduced loss |

## Complete Example: Factual Recall

```python
from mlxterp import InterpretableModel
from mlxterp.causal import activation_patching
from mlxterp.metrics import logit_diff

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

clean = "The Eiffel Tower is in"
corrupted = "The Colosseum is in"

# Step 1: Find important layers (MLP)
mlp_result = activation_patching(
    model, clean, corrupted,
    component="mlp",
    metric="l2",
)

# Step 2: Find important layers (attention)
attn_result = activation_patching(
    model, clean, corrupted,
    component="attn",
    metric="l2",
)

# Step 3: For the most important layer, find which heads matter
top_layer = mlp_result.top_components(k=1)[0][0]
print(f"Most important MLP layer: {top_layer}")

head_result = activation_patching(
    model, clean, corrupted,
    component="attn_head",
    metric="l2",
    layers=[top_layer - 1, top_layer, top_layer + 1],
)

head_result.plot()  # Heatmap: layer x head
```

## Position-Level Patching

Identify which token positions carry the critical information:

```python
result = activation_patching(
    model, clean, corrupted,
    component="mlp",
    positions=[3, 4, 5],  # Only patch these positions
)
```

## Using CausalTrace for Multi-Patch Experiments

When you want to patch multiple components simultaneously:

```python
with model.causal_trace(clean, corrupted) as ct:
    # Patch MLP at layers 5-7 and attention at layer 9
    ct.patch("layers.5.mlp")
    ct.patch("layers.6.mlp")
    ct.patch("layers.7.mlp")
    ct.patch("layers.9.self_attn")

    # All patches applied at once
    effect = ct.metric("l2")
    print(f"Combined effect: {effect:.4f}")
```

## Using logit_diff Metric

For tasks with a clear correct/incorrect answer:

```python
correct_token = model.tokenizer.encode(" Paris")[-1]
incorrect_token = model.tokenizer.encode(" Rome")[-1]

result = activation_patching(
    model, clean, corrupted,
    component="mlp",
    metric="logit_diff",
    metric_kwargs={
        "correct_token": correct_token,
        "incorrect_token": incorrect_token,
    },
)
```

## Working with Results

Every patching function returns a `PatchingResult`:

```python
# Summary
print(result.summary())

# Top components
for layer, effect in result.top_components(k=5):
    print(f"  Layer {layer}: {effect:.4f}")

# JSON export (for programmatic use)
json_data = result.to_json()

# Markdown report
print(result.to_markdown())

# Raw effect matrix
print(result.effect_matrix)       # mx.array
print(result.effect_matrix.tolist())  # Python list
```

## Interpreting Results

- **High positive effect**: This component is important. Patching the clean activation into the corrupted run recovered the clean behavior.
- **Near-zero effect**: This component doesn't contribute to the difference between clean and corrupted.
- **Negative effect**: Patching this component made things worse (rare, but indicates interference).

!!! info "Rule of Thumb"
    Start with `component="mlp"` and `component="attn"` at all layers to get a broad picture. Then zoom into specific layers with `component="attn_head"` for head-level analysis.
