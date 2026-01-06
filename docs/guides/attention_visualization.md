# Attention Visualization Guide

This guide covers all attention visualization and analysis capabilities in mlxterp.

## Overview

mlxterp captures attention weights during model tracing, enabling:

- **Attention heatmaps**: Visualize which tokens attend to which
- **Pattern detection**: Automatically identify head types (induction, previous token, etc.)
- **Multi-head analysis**: Compare patterns across layers and heads
- **Custom analysis**: Build your own attention-based analyses

## Quick Start

```python
from mlxterp import InterpretableModel
from mlxterp.visualization import (
    get_attention_patterns,
    attention_heatmap,
    attention_from_trace,
    detect_head_types,
)

# Load model
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Trace to capture attention
with model.trace("The cat sat on the mat") as trace:
    pass

# Get attention patterns
patterns = get_attention_patterns(trace)
tokens = model.to_str_tokens("The cat sat on the mat")

# Create visualization
fig = attention_heatmap(patterns[5], tokens, head_idx=0)
```

## Extracting Attention Patterns

### Basic Extraction

```python
from mlxterp.visualization import get_attention_patterns

with model.trace(text) as trace:
    pass

# Get all layers
patterns = get_attention_patterns(trace)
print(f"Captured {len(patterns)} layers")

# Get specific layers
patterns = get_attention_patterns(trace, layers=[0, 5, 10])
```

### Understanding Pattern Shapes

```python
# patterns[layer_idx] has shape: (batch, heads, seq_q, seq_k)
pattern = patterns[0]
print(f"Shape: {pattern.shape}")
# (1, 32, 6, 6) for batch=1, 32 heads, 6 tokens

# Access specific head
head_0 = pattern[0, 0]  # First batch, first head
# Shape: (seq_q, seq_k) = (6, 6)
```

### Token Strings

Use `to_str_tokens` to get readable token labels:

```python
tokens = model.to_str_tokens("Hello world")
print(tokens)  # ['<|begin_of_text|>', 'Hello', ' world']

# Also works with token IDs
token_ids = model.encode("Hello world")
tokens = model.to_str_tokens(token_ids)
```

## Visualization Functions

### Single Heatmap

```python
from mlxterp.visualization import attention_heatmap

fig = attention_heatmap(
    patterns[5],          # Attention from layer 5
    tokens,               # Token labels
    head_idx=0,           # Which head to show
    title="Layer 5, Head 0",
    colorscale="Blues",   # Colormap
    backend="matplotlib", # or "plotly", "circuitsviz"
    mask_upper_tri=True,  # Mask future positions
    figsize=(8, 6)
)
```

### Grid of Multiple Heads

```python
from mlxterp.visualization import attention_from_trace, AttentionVisualizationConfig

config = AttentionVisualizationConfig(
    backend="matplotlib",
    colorscale="Blues",
    mask_upper_tri=True
)

fig = attention_from_trace(
    trace,
    tokens,
    layers=[0, 4, 8, 12],  # 4 layers
    heads=[0, 1, 2, 3],     # 4 heads per layer
    mode="grid",            # Grid layout
    head_notation="LH",     # "L5H3" style titles
    config=config
)
```

### Visualization Backends

mlxterp supports multiple backends:

| Backend | Best For | Installation |
|---------|----------|--------------|
| `matplotlib` | Publication figures, local use | Included |
| `plotly` | Interactive exploration | `pip install plotly` |
| `circuitsviz` | Jupyter notebooks, web | `pip install circuitsviz` |

```python
# Auto-detect best available
config = AttentionVisualizationConfig(backend="auto")

# Force specific backend
config = AttentionVisualizationConfig(backend="plotly")
```

## Pattern Detection

### Detecting Head Types

```python
from mlxterp.visualization import detect_head_types

head_types = detect_head_types(
    model,
    "The quick brown fox jumps over the lazy dog",
    threshold=0.3,
    layers=[0, 5, 10, 15]  # Optional: specific layers
)

print("Head Types Found:")
for head_type, heads in head_types.items():
    if heads:
        print(f"  {head_type}: {len(heads)} heads")
        for layer, head in heads[:3]:
            print(f"    L{layer}H{head}")
```

**Head Types:**

| Type | Description | Typical Location |
|------|-------------|------------------|
| `previous_token` | Attends to position i-1 | Early layers |
| `first_token` | Attends to position 0 (BOS) | All layers |
| `current_token` | Attends to self (diagonal) | Various |
| `induction` | Pattern completion heads | Middle-late layers |

### Detecting Induction Heads

For accurate induction head detection, use repeated random sequences:

```python
from mlxterp.visualization import detect_induction_heads

induction_heads = detect_induction_heads(
    model,
    n_random_tokens=50,  # Length of random sequence
    n_repeats=2,         # Repeat twice
    threshold=0.3,       # Score threshold
    seed=42              # Reproducibility
)

print(f"Found {len(induction_heads)} induction heads")
for head in induction_heads[:5]:
    print(f"  L{head.layer}H{head.head}: {head.score:.3f}")
```

### Computing Pattern Scores

```python
from mlxterp.visualization import (
    induction_score,
    previous_token_score,
    first_token_score,
)
import numpy as np

# Get a single head's pattern
head_pattern = patterns[5][0, 3]  # Layer 5, batch 0, head 3

# Compute scores
prev_score = previous_token_score(head_pattern)
first_score = first_token_score(head_pattern)
print(f"Previous token: {prev_score:.3f}")
print(f"First token: {first_score:.3f}")

# Induction score requires sequence length
ind_score = induction_score(head_pattern, seq_len=5)
print(f"Induction: {ind_score:.3f}")
```

### Using AttentionPatternDetector

For custom analysis with configurable thresholds:

```python
from mlxterp.visualization import AttentionPatternDetector

detector = AttentionPatternDetector(
    induction_threshold=0.4,
    previous_token_threshold=0.5,
    first_token_threshold=0.3,
    current_token_threshold=0.3
)

# Analyze a head
scores = detector.analyze_head(head_pattern)
print(f"All scores: {scores}")

# Classify
types = detector.classify_head(head_pattern)
print(f"Classification: {types}")
```

## Advanced Usage

### Analyzing Specific Token Relationships

```python
# Which tokens does position 5 attend to?
attn_from_pos_5 = head_pattern[5, :]
print(f"Position 5 attends to: {np.argsort(attn_from_pos_5)[::-1][:3]}")

# What attends to position 2?
attn_to_pos_2 = head_pattern[:, 2]
print(f"Positions attending to 2: {np.where(attn_to_pos_2 > 0.1)[0]}")
```

### Custom Pattern Functions

```python
from mlxterp.visualization.patterns import find_attention_pattern

def diagonal_score(pattern):
    """Score for attention to self (main diagonal)."""
    return float(np.mean(np.diag(pattern)))

heads = find_attention_pattern(
    model,
    "Test text",
    pattern_fn=diagonal_score,
    threshold=0.5
)
```

### Comparing Across Prompts

```python
def compare_attention(model, prompts, layer, head):
    """Compare attention patterns across different prompts."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(prompts), figsize=(5*len(prompts), 4))

    for idx, prompt in enumerate(prompts):
        with model.trace(prompt) as trace:
            pass

        patterns = get_attention_patterns(trace, layers=[layer])
        tokens = model.to_str_tokens(prompt)

        attn = patterns[layer][0, head]
        mask = np.triu(np.ones_like(attn), k=1)
        attn_masked = np.where(mask, np.nan, attn)

        ax = axes[idx]
        ax.imshow(attn_masked, cmap='Blues')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f'"{prompt[:20]}..."')

    plt.tight_layout()
    return fig

# Compare attention across prompts
fig = compare_attention(
    model,
    ["The cat sat", "A dog ran", "My car drove"],
    layer=5,
    head=0
)
```

### Aggregating Across Heads

```python
def layer_attention_summary(patterns, layer_idx):
    """Summarize attention across all heads in a layer."""
    attn = patterns[layer_idx]  # (batch, heads, seq_q, seq_k)

    # Mean across heads
    mean_attn = np.mean(attn[0], axis=0)

    # Max across heads
    max_attn = np.max(attn[0], axis=0)

    return mean_attn, max_attn

mean_attn, max_attn = layer_attention_summary(patterns, 5)
```

## Best Practices

### 1. Use Random Tokens for Induction Detection

```python
# Good: Random tokens eliminate semantic confounds
induction_heads = detect_induction_heads(model, n_random_tokens=50)

# Less reliable: Natural text has many patterns
head_types = detect_head_types(model, "natural text")
```

### 2. Check Multiple Layers

```python
# Analyze pattern evolution through layers
for layer in [0, 4, 8, 12, 15]:
    types = detect_head_types(model, text, layers=[layer])
    print(f"Layer {layer}: {sum(len(v) for v in types.values())} classified heads")
```

### 3. Use Appropriate Thresholds

```python
# Start conservative, then relax
for threshold in [0.5, 0.4, 0.3, 0.2]:
    heads = detect_induction_heads(model, threshold=threshold)
    print(f"Threshold {threshold}: {len(heads)} heads")
```

### 4. Validate with Ablation

```python
from mlxterp import interventions as iv

# Confirm head importance by ablating it
text = "The cat sat on the mat. The cat"

# Normal
with model.trace(text) as trace:
    normal_out = model.output.save()

# Ablated
with model.trace(text, interventions={"model.model.layers.8.self_attn": iv.zero_out}):
    ablated_out = model.output.save()

# Compare predictions
normal_pred = model.get_token_predictions(normal_out[0, -1, :], top_k=1)[0]
ablated_pred = model.get_token_predictions(ablated_out[0, -1, :], top_k=1)[0]

print(f"Normal: {model.token_to_str(normal_pred)}")
print(f"Ablated: {model.token_to_str(ablated_pred)}")
```

## Common Issues

### Memory with Long Sequences

For long sequences, attention patterns can be large:

```python
# Shape: (batch, heads, seq_len, seq_len)
# For 32 heads and 1024 tokens: 32 * 1024 * 1024 * 4 bytes = 128MB per layer

# Process specific layers only
patterns = get_attention_patterns(trace, layers=[5, 10])
```

### Causal Masking

Upper triangular entries are masked (future positions):

```python
# Always use mask_upper_tri=True for causal models
config = AttentionVisualizationConfig(mask_upper_tri=True)
```

### Tokenization Alignment

Ensure tokens match the traced sequence:

```python
text = "Hello world"
with model.trace(text) as trace:
    pass

# Use same text for tokenization
tokens = model.to_str_tokens(text)  # Correct
# NOT: tokens = model.to_str_tokens("different text")
```

## See Also

- [Tutorial 5: Induction Heads](../tutorials/05_induction_heads.md) - Comprehensive induction head analysis
- [API Reference: Visualization Module](../API.md#visualization-module) - Complete API documentation
- [Activation Patching Guide](activation_patching.md) - Causal intervention techniques
