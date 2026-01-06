# Tutorial 5: Induction Heads

**Paper**: [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/) by Olsson et al. (Anthropic, 2022)

**Difficulty**: Intermediate-Advanced | **Time**: 2-3 hours

---

## Overview

This tutorial demonstrates how to identify and analyze **induction heads** - attention heads that implement a pattern completion algorithm. Given the pattern `[A][B]...[A]`, an induction head predicts `[B]`.

Induction heads are fundamental to in-context learning, enabling transformers to:
- Copy patterns they've seen before
- Learn from examples in the prompt
- Perform few-shot learning

We'll use mlxterp's visualization module to detect and analyze these heads.

## Prerequisites

- mlxterp installed with visualization dependencies
- Basic understanding of attention mechanisms
- Familiarity with mlxterp tracing (see [Quick Start](../QUICKSTART.md))

```bash
# Install with visualization support
pip install mlxterp[viz]
```

## Part 1: Understanding Induction Heads

### The Induction Algorithm

Induction heads work in two steps:

1. **Previous Token Head**: Attends from `[A]` to the token before `[A]` (i.e., `[A]` attends to what came before the first occurrence of `[A]`)
2. **Induction Head**: Uses the previous token information to attend to `[B]` (the token that followed `[A]` previously)

For example, with the sequence `"The cat sat on the mat. The cat"`:
- When processing the second `"cat"`, the induction head attends to `"sat"` (what came after `"cat"` previously)
- This allows the model to predict `"sat"` as the next token

### Why This Matters

Induction heads are the mechanism behind:
- **Pattern completion**: `A B A → B`
- **In-context learning**: Learning from examples in the prompt
- **Few-shot learning**: Generalizing from limited examples

## Part 2: Detecting Induction Heads

### Setup

```python
from mlxterp import InterpretableModel
from mlxterp.visualization import (
    get_attention_patterns,
    attention_heatmap,
    detect_induction_heads,
    detect_head_types,
    induction_score,
    previous_token_score,
    AttentionVisualizationConfig,
)
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
```

### Method 1: Using `detect_induction_heads`

The most reliable way to detect induction heads is using repeated random sequences:

```python
# Detect induction heads across all layers
induction_heads = detect_induction_heads(
    model,
    n_random_tokens=50,    # Length of random sequence
    n_repeats=2,           # Repeat twice: [random tokens][random tokens]
    threshold=0.3,         # Score threshold
    seed=42                # For reproducibility
)

print(f"Found {len(induction_heads)} potential induction heads\n")

# Show top 10
print("Top Induction Heads:")
print("-" * 40)
for head in induction_heads[:10]:
    print(f"  Layer {head.layer:2d}, Head {head.head:2d}: score = {head.score:.3f}")
```

**Why random tokens?** Using random tokens eliminates semantic patterns that might confuse the detection. The only structure the model can use is the repeated sequence itself.

### Method 2: Using `detect_head_types`

For a broader view including other head types:

```python
# Detect all head types on a natural text
text = "The quick brown fox jumps over the lazy dog"

head_types = detect_head_types(
    model,
    text,
    threshold=0.3,
    layers=list(range(16))  # All layers for Llama-3.2-1B
)

print("Head Type Distribution:")
print("-" * 40)
for head_type, heads in head_types.items():
    if heads:
        print(f"  {head_type}: {len(heads)} heads")
```

## Part 3: Visualizing Attention Patterns

### Basic Attention Heatmap

Let's visualize what an induction head actually looks like:

```python
# Use a repeated sequence to trigger induction behavior
text = "ABC DEF ABC"

with model.trace(text) as trace:
    pass

tokens = model.to_str_tokens(text)
patterns = get_attention_patterns(trace)

# Pick a layer known to have induction heads (from detection above)
# For Llama models, middle-to-late layers often have induction heads
layer_idx = 8

# Get the attention pattern
attn = patterns[layer_idx]
print(f"Tokens: {tokens}")
print(f"Attention shape: {attn.shape}")  # (batch, heads, seq_len, seq_len)
```

### Visualizing a Single Head

```python
# Visualize head 0 from layer 8
fig = attention_heatmap(
    patterns[layer_idx],
    tokens,
    head_idx=0,
    title=f"Layer {layer_idx}, Head 0",
    backend="matplotlib",
    colorscale="Blues"
)
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150)
plt.show()
```

### Grid Visualization of Multiple Heads

```python
from mlxterp.visualization import attention_from_trace

# Visualize multiple heads across layers
config = AttentionVisualizationConfig(
    backend="matplotlib",
    colorscale="Blues",
    mask_upper_tri=True
)

with model.trace(text) as trace:
    pass

# Create grid of attention patterns
fig = attention_from_trace(
    trace,
    tokens,
    layers=[0, 4, 8, 12],  # Sample layers
    heads=[0, 1, 2, 3],     # First 4 heads
    mode="grid",
    config=config
)
plt.tight_layout()
plt.savefig("attention_grid.png", dpi=150)
plt.show()
```

## Part 4: Analyzing Induction Behavior in Detail

### Computing Induction Scores Manually

```python
# Create a repeated random sequence
np.random.seed(42)
vocab_size = model.vocab_size
min_token = int(vocab_size * 0.1)
max_token = int(vocab_size * 0.9)

# Generate random tokens and repeat
n_tokens = 30
random_tokens = np.random.randint(min_token, max_token, size=n_tokens)
repeated_tokens = np.tile(random_tokens, 2)  # Repeat twice

# Run trace with token IDs
import mlx.core as mx
token_ids = mx.array([repeated_tokens.tolist()])

with model.trace(token_ids) as trace:
    pass

# Get attention patterns
patterns = get_attention_patterns(trace)

# Compute induction scores for each head
print("Layer-by-Layer Induction Scores:")
print("-" * 50)

for layer_idx in [0, 4, 8, 12, 15]:
    attn = patterns[layer_idx]
    num_heads = attn.shape[1]

    scores = []
    for head_idx in range(num_heads):
        head_pattern = attn[0, head_idx]  # First batch
        score = induction_score(head_pattern, n_tokens)
        scores.append(score)

    max_score = max(scores)
    max_head = scores.index(max_score)
    print(f"Layer {layer_idx:2d}: max score = {max_score:.3f} (head {max_head})")
```

### Identifying Previous Token Heads

Induction heads require previous token heads to function. Let's find them:

```python
from mlxterp.visualization import AttentionPatternDetector

detector = AttentionPatternDetector(
    previous_token_threshold=0.5,
    first_token_threshold=0.3
)

print("Head Analysis:")
print("-" * 60)

# Analyze first few layers (where previous token heads often appear)
with model.trace("The quick brown fox") as trace:
    pass

patterns = get_attention_patterns(trace, layers=[0, 1, 2, 3])

for layer_idx, attn in patterns.items():
    num_heads = attn.shape[1]
    for head_idx in range(num_heads):
        head_pattern = attn[0, head_idx]
        scores = detector.analyze_head(head_pattern)
        types = detector.classify_head(head_pattern)

        if "previous_token" in types:
            print(f"L{layer_idx}H{head_idx}: prev_token={scores['previous_token']:.2f} ← Previous Token Head")
        elif "first_token" in types:
            print(f"L{layer_idx}H{head_idx}: first_token={scores['first_token']:.2f} ← BOS Head")
```

## Part 5: The Two-Step Composition

### Understanding K-Composition

Induction heads work through **K-composition** (key composition):

1. **Step 1**: A previous token head at position `p` writes information about position `p-1` into the residual stream
2. **Step 2**: The induction head uses this information in its key to attend to the right position

```python
# Demonstrate the composition on a simple repeated pattern
text = "X Y X"  # When at second X, attend to Y

with model.trace(text) as trace:
    pass

tokens = model.to_str_tokens(text)
patterns = get_attention_patterns(trace)

print(f"Tokens: {tokens}")
print()

# Look for the induction pattern:
# At position of second "X", attention should go to "Y"
for layer_idx in [4, 8, 12]:
    attn = patterns[layer_idx]
    num_heads = attn.shape[1]

    # Token indices (adjust based on tokenization)
    # In "X Y X", second X should attend to Y
    second_x_pos = len(tokens) - 1
    y_pos = second_x_pos - 1

    for head_idx in range(num_heads):
        head_pattern = attn[0, head_idx]
        attn_to_y = head_pattern[second_x_pos, y_pos]

        if attn_to_y > 0.3:
            print(f"L{layer_idx}H{head_idx}: Attention to Y = {attn_to_y:.3f}")
```

## Part 6: Ablation Study

To confirm a head is causally responsible for induction, we can ablate it:

```python
from mlxterp import interventions as iv

# Test completion with and without suspected induction head
text = "The cat sat on the mat. The cat"

# Normal completion
with model.trace(text) as trace:
    normal_output = model.output.save()

# Get top prediction
normal_pred = model.get_token_predictions(normal_output[0, -1, :], top_k=1)
normal_token = model.token_to_str(normal_pred[0])
print(f"Normal prediction: '{normal_token}'")

# Ablate suspected induction head (e.g., layer 8)
# Zero out the entire attention output
with model.trace(text, interventions={"model.model.layers.8.self_attn": iv.zero_out}):
    ablated_output = model.output.save()

ablated_pred = model.get_token_predictions(ablated_output[0, -1, :], top_k=1)
ablated_token = model.token_to_str(ablated_pred[0])
print(f"After ablating L8 attention: '{ablated_token}'")
```

### Measuring Induction Loss

A more quantitative approach measures the "induction loss" - how well the model predicts the completion:

```python
import mlx.core as mx

def measure_induction_ability(model, seq_len=50, n_trials=5):
    """
    Measure model's induction ability by testing pattern completion.
    """
    results = []

    for trial in range(n_trials):
        # Generate random sequence and repeat
        np.random.seed(trial)
        vocab_size = model.vocab_size
        random_tokens = np.random.randint(1000, vocab_size - 1000, size=seq_len)
        repeated = np.tile(random_tokens, 2)

        token_ids = mx.array([repeated.tolist()])

        with model.trace(token_ids) as trace:
            logits = model.output.save()

        # Check predictions at positions where we expect induction
        # Position i in second half should predict token at position i+1 in first half
        correct = 0
        total = 0

        for i in range(seq_len, 2 * seq_len - 1):
            target = repeated[i - seq_len + 1]  # What should be predicted

            # Get prediction at position i
            pred_logits = logits[0, i, :]
            pred_token = mx.argmax(pred_logits).item()

            if pred_token == target:
                correct += 1
            total += 1

        accuracy = correct / total
        results.append(accuracy)

    return np.mean(results), np.std(results)

mean_acc, std_acc = measure_induction_ability(model)
print(f"Induction accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
```

## Part 7: Visualizing Induction Patterns

### Creating Publication-Quality Figures

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_induction_analysis(model, layers=[0, 4, 8, 12], n_tokens=50):
    """
    Create a comprehensive visualization of induction head behavior.
    """
    # Generate repeated random sequence
    np.random.seed(42)
    vocab_size = model.vocab_size
    random_tokens = np.random.randint(5000, vocab_size - 5000, size=n_tokens)
    repeated = np.tile(random_tokens, 2)

    token_ids = mx.array([repeated.tolist()])

    with model.trace(token_ids) as trace:
        pass

    patterns = get_attention_patterns(trace, layers=layers)

    # Compute induction scores for all heads
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4))

    for idx, layer_idx in enumerate(layers):
        attn = patterns[layer_idx]
        num_heads = attn.shape[1]

        scores = []
        for head_idx in range(num_heads):
            head_pattern = attn[0, head_idx]
            score = induction_score(head_pattern, n_tokens)
            scores.append(score)

        ax = axes[idx] if len(layers) > 1 else axes
        ax.bar(range(num_heads), scores, color='steelblue', alpha=0.7)
        ax.axhline(y=0.4, color='red', linestyle='--', label='Threshold')
        ax.set_xlabel('Head')
        ax.set_ylabel('Induction Score')
        ax.set_title(f'Layer {layer_idx}')
        ax.set_ylim(0, 1)

    plt.suptitle('Induction Scores by Layer and Head', fontsize=14)
    plt.tight_layout()
    plt.savefig('induction_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run the visualization
plot_induction_analysis(model)
```

### Attention Pattern Comparison

```python
def compare_head_patterns(model, text, head_pairs):
    """
    Compare attention patterns of different heads.

    head_pairs: list of (layer, head) tuples
    """
    with model.trace(text) as trace:
        pass

    tokens = model.to_str_tokens(text)
    patterns = get_attention_patterns(trace)

    fig, axes = plt.subplots(1, len(head_pairs), figsize=(5 * len(head_pairs), 4))

    for idx, (layer, head) in enumerate(head_pairs):
        attn = patterns[layer][0, head]  # First batch

        ax = axes[idx] if len(head_pairs) > 1 else axes

        # Mask upper triangle (future positions)
        mask = np.triu(np.ones_like(attn), k=1)
        attn_masked = np.where(mask, np.nan, attn)

        im = ax.imshow(attn_masked, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f'L{layer}H{head}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

    plt.tight_layout()
    plt.savefig('head_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Compare different head types
compare_head_patterns(
    model,
    "Hello world Hello",
    head_pairs=[(0, 0), (4, 3), (8, 5)]  # Adjust based on your detection
)
```

## Part 8: Full Analysis Pipeline

Here's a complete pipeline that combines all the analyses:

```python
def full_induction_analysis(model_name, save_prefix="induction"):
    """
    Complete induction head analysis for a model.
    """
    print("=" * 60)
    print(f"Induction Head Analysis: {model_name}")
    print("=" * 60)

    # Load model
    print("\n1. Loading model...")
    model = InterpretableModel(model_name)

    # Detect induction heads
    print("\n2. Detecting induction heads...")
    induction_heads = detect_induction_heads(
        model,
        n_random_tokens=50,
        threshold=0.3
    )

    print(f"\n   Found {len(induction_heads)} induction heads:")
    for head in induction_heads[:5]:
        print(f"   - L{head.layer}H{head.head}: {head.score:.3f}")
    if len(induction_heads) > 5:
        print(f"   ... and {len(induction_heads) - 5} more")

    # Detect other head types
    print("\n3. Analyzing head types...")
    head_types = detect_head_types(
        model,
        "The quick brown fox jumps over the lazy dog",
        threshold=0.3
    )

    print("\n   Head type distribution:")
    for head_type, heads in head_types.items():
        if heads:
            print(f"   - {head_type}: {len(heads)} heads")

    # Measure induction ability
    print("\n4. Measuring induction ability...")
    mean_acc, std_acc = measure_induction_ability(model)
    print(f"   Induction accuracy: {mean_acc:.1%} ± {std_acc:.1%}")

    # Create visualizations
    print("\n5. Creating visualizations...")

    # Layer distribution plot
    if induction_heads:
        layers = [h.layer for h in induction_heads]
        plt.figure(figsize=(10, 4))
        plt.hist(layers, bins=range(max(layers) + 2), alpha=0.7, color='steelblue')
        plt.xlabel('Layer')
        plt.ylabel('Number of Induction Heads')
        plt.title('Distribution of Induction Heads Across Layers')
        plt.savefig(f'{save_prefix}_layer_distribution.png', dpi=150)
        plt.close()

    print(f"\n   Saved: {save_prefix}_layer_distribution.png")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return {
        'induction_heads': induction_heads,
        'head_types': head_types,
        'induction_accuracy': (mean_acc, std_acc)
    }

# Run the full analysis
results = full_induction_analysis("mlx-community/Llama-3.2-1B-Instruct-4bit")
```

## Summary

In this tutorial, we learned to:

1. **Understand induction heads**: The mechanism behind in-context learning
2. **Detect induction heads**: Using `detect_induction_heads` with random token sequences
3. **Visualize attention patterns**: Heatmaps and grid visualizations
4. **Analyze head types**: Previous token heads, first token heads, and their roles
5. **Perform ablation studies**: Confirming causal importance
6. **Measure induction ability**: Quantifying pattern completion accuracy

## Key Takeaways

- **Induction heads are universal**: They appear in most transformer models
- **Layer location matters**: Induction heads typically appear in middle-to-late layers
- **Previous token heads are prerequisites**: They appear in early layers and enable induction
- **Repeated random tokens are the gold standard**: For detecting induction heads

## References

1. Olsson, C., et al. (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). Anthropic.

2. Elhage et al. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). Anthropic.

3. Wang et al. (2022). [Interpretability in the Wild: a Circuit for Indirect Object Identification](https://arxiv.org/abs/2211.00593).

## Next Steps

- **Tutorial 6: Sparse Autoencoders** - Decompose neural activations into interpretable features
- **Activation Patching Guide** - Learn more about causal interventions
