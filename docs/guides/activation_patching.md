# Activation Patching Guide

Activation patching is a fundamental technique in mechanistic interpretability for identifying which components of a neural network are important for specific tasks.

## Overview

**Goal**: Determine which layers or components are critical for a task by:

1. Running a "clean" input through the model
2. Running a "corrupted" input through the model
3. Patching clean activations into the corrupted run at different locations
4. Measuring how much this recovers the clean output

**Key insight**: If patching a component significantly recovers the clean output, that component is important for the task.

## Quick Start: Using the Helper Function

The easiest way to perform activation patching is with the built-in `activation_patching()` method:

```python
from mlxterp import InterpretableModel
from mlx_lm import load

# Load model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Find important layers - that's it!
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    plot=True
)

# Analyze
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("Top 3 most important layers:")
for layer_idx, recovery in sorted_results[:3]:
    print(f"  Layer {layer_idx}: {recovery:.1f}% recovery")
```

**The helper function handles all the boilerplate:**
- Running clean and corrupted inputs
- Patching each layer automatically
- Measuring recovery with L2 distance
- Optional visualization

Continue reading for details on interpretation and manual implementation.

## Quick Example

```python
import mlx.core as mx
from mlxterp import InterpretableModel, interventions as iv
from mlx_lm import load

# Load model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Define clean vs corrupted inputs
clean_text = "Paris is the capital of France"
corrupted_text = "London is the capital of France"

# Get baseline outputs
with model.trace(clean_text):
    clean_output = model.output.save()

with model.trace(corrupted_text):
    corrupted_output = model.output.save()

mx.eval(clean_output, corrupted_output)

# Helper function to measure distance
def l2_distance(a, b):
    return float(mx.sqrt(mx.sum((a - b) ** 2)))

baseline = l2_distance(corrupted_output[0, -1], clean_output[0, -1])
print(f"Baseline L2 distance: {baseline:.2f}")

# Patch MLP at layer 10
with model.trace(clean_text) as trace:
    clean_mlp = trace.activations["model.model.layers.10.mlp"]

mx.eval(clean_mlp)

with model.trace(corrupted_text,
                interventions={"layers.10.mlp": iv.replace_with(clean_mlp)}):
    patched_output = model.output.save()

mx.eval(patched_output)

dist = l2_distance(patched_output[0, -1], clean_output[0, -1])
recovery = (baseline - dist) / baseline * 100
print(f"Layer 10 MLP: {recovery:.1f}% recovery")
```

## Complete Procedure

### Step 1: Define Clean and Corrupted Inputs

Choose inputs that differ in exactly the aspect you want to study:

```python
# Factual knowledge task
clean_text = "Paris is the capital of France"
corrupted_text = "London is the capital of France"

# Sentiment task
clean_text = "This movie was amazing"
corrupted_text = "This movie was terrible"

# Grammatical task
clean_text = "The cat sits on the mat"
corrupted_text = "The cat sit on the mat"
```

### Step 2: Get Baseline Measurements

```python
# Get clean output
with model.trace(clean_text):
    clean_output = model.output.save()

# Get corrupted output
with model.trace(corrupted_text):
    corrupted_output = model.output.save()

mx.eval(clean_output, corrupted_output)

# Measure baseline distance
def l2_distance(a, b):
    """L2 (Euclidean) distance between output logits"""
    return float(mx.sqrt(mx.sum((a - b) ** 2)))

baseline = l2_distance(corrupted_output[0, -1], clean_output[0, -1])
```

### Step 3: Patch Each Component

```python
results = {}

for layer_idx in range(len(model.layers)):
    # Get clean activation for this component
    with model.trace(clean_text) as trace:
        clean_mlp = trace.activations[f"model.model.layers.{layer_idx}.mlp"]

    mx.eval(clean_mlp)

    # Patch into corrupted run
    with model.trace(corrupted_text,
                    interventions={f"layers.{layer_idx}.mlp": iv.replace_with(clean_mlp)}):
        patched_output = model.output.save()

    mx.eval(patched_output)

    # Measure recovery
    dist = l2_distance(patched_output[0, -1], clean_output[0, -1])
    recovery = (baseline - dist) / baseline * 100
    results[layer_idx] = recovery

    print(f"Layer {layer_idx:2d}: {recovery:6.1f}% recovery")
```

### Step 4: Analyze Results

```python
# Sort by importance
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\nMost important layers:")
for layer_idx, recovery in sorted_results[:5]:
    print(f"  Layer {layer_idx:2d}: {recovery:5.1f}% recovery")
```

## Understanding Results

### Positive Recovery

**High positive recovery** (e.g., +40%) = **Important layer**

- Patching this component significantly recovers clean output
- This layer is critical for the task
- Often seen in early layers (feature extraction) and final layers (output formation)

Example:
```
Layer  0: +43.1% recovery  ← Very important!
Layer 15: +24.2% recovery  ← Important
```

### Negative Recovery

**Negative recovery** (e.g., -20%) = **Encodes corruption**

- Patching makes output WORSE than corrupted baseline
- This layer actively encodes the corrupted information
- This is expected and informative!

Example:
```
Layer  7: -18.4% recovery  ← Encodes "London"
Layer 10: -23.5% recovery  ← Strongly encodes corruption
```

### Near-Zero Recovery

**~0% recovery** = **Not relevant**

- Patching has minimal effect
- Layer doesn't significantly contribute to this specific task
- Might be important for other tasks

## Components to Patch

You can patch different granularities:

### Full MLP

```python
# Patch entire MLP block
interventions={"layers.{i}.mlp": iv.replace_with(clean_mlp)}
```

### MLP Sub-components

```python
# Gate projection
interventions={"layers.{i}.mlp.gate_proj": iv.replace_with(clean_gate)}

# Up projection
interventions={"layers.{i}.mlp.up_proj": iv.replace_with(clean_up)}

# Down projection
interventions={"layers.{i}.mlp.down_proj": iv.replace_with(clean_down)}
```

### Attention Components

```python
# Full attention
interventions={"layers.{i}.self_attn": iv.replace_with(clean_attn)}

# Query projection
interventions={"layers.{i}.self_attn.q_proj": iv.replace_with(clean_q)}

# Key projection
interventions={"layers.{i}.self_attn.k_proj": iv.replace_with(clean_k)}

# Value projection
interventions={"layers.{i}.self_attn.v_proj": iv.replace_with(clean_v)}

# Output projection
interventions={"layers.{i}.self_attn.o_proj": iv.replace_with(clean_o)}
```

## Common Pitfalls

### ❌ DON'T: Patch Entire Layer Output

```python
# WRONG - patches entire residual stream
with model.trace(corrupted_text,
                interventions={"layers.10": iv.replace_with(clean_act)}):
    pass
```

**Why wrong**: This replaces the entire residual stream, affecting ALL downstream layers. You'll get perfect recovery for every layer.

### ❌ DON'T: Use Lambda Closures

```python
# WRONG - lambda closure bug
for i in range(num_layers):
    with model.trace(clean_text):
        clean_act = model.layers[i].output.save()

    # Bug: all interventions use the SAME activation
    with model.trace(corrupted_text,
                    interventions={f"layers.{i}": lambda x: clean_act}):
        pass
```

**Why wrong**: Lambda captures `clean_act` by reference, so all interventions end up using the last layer's activation.

### ✅ DO: Use iv.replace_with()

```python
# CORRECT
with model.trace(corrupted_text,
                interventions={"layers.10.mlp": iv.replace_with(clean_mlp)}):
    pass
```

### ✅ DO: Choose the Right Distance Metric

The `activation_patching()` helper supports three distance metrics. Choose based on your model's vocabulary size:

```python
# For small/medium models (< 50k vocab)
results = model.activation_patching(
    clean_text="...",
    corrupted_text="...",
    metric="l2"  # Default - Euclidean distance
)

# For large vocabulary models (> 100k vocab)
results = model.activation_patching(
    clean_text="...",
    corrupted_text="...",
    metric="mse"  # Most stable for huge models
)
```

**Why**: KL divergence can give NaN, and L2 can overflow on large vocabularies. See [Distance Metrics](#distance-metrics) section below.

## Advanced: Position-Specific Patching

Patch activations only at specific token positions:

```python
# Patch only the last token's MLP activation
with model.trace(clean_text) as trace:
    clean_mlp = trace.activations["model.model.layers.10.mlp"]

# Create patched activation: clean for last token, corrupted for others
def selective_patch(corrupted_activation):
    patched = corrupted_activation.copy()
    patched[0, -1, :] = clean_mlp[0, -1, :]  # Patch last token only
    return patched

with model.trace(corrupted_text,
                interventions={"layers.10.mlp": selective_patch}):
    patched_output = model.output.save()
```

## Distance Metrics

The `activation_patching()` helper uses distance metrics to measure how different the outputs are. Choosing the right metric is crucial, especially for large models.

### Available Metrics

#### 1. L2 Distance (Euclidean) - Default

**Formula**:
```
d(a, b) = √(Σ(aᵢ - bᵢ)²)
```

**When to use**: Small to medium models (vocabulary < 50k tokens)

**Implementation**:
```python
def l2_distance(a, b):
    diff = a - b
    # Use float32 for accumulation to prevent overflow
    diff_f32 = diff.astype(mx.float32)
    squared_sum = mx.sum(diff_f32 * diff_f32)

    # Check for overflow
    if mx.isinf(squared_sum):
        # Fallback to MSE-based calculation
        mse = mx.mean(diff_f32 * diff_f32)
        return float(mx.sqrt(mse) * mx.sqrt(float(diff.size)))

    return float(mx.sqrt(squared_sum))
```

**Why it can fail**: With large vocabularies (e.g., 150k tokens), summing 150k squared differences can overflow to `inf`, especially in float16.

**Example**:
```python
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    metric="l2"  # Default
)
```

#### 2. Cosine Distance

**Formula**:
```
d(a, b) = 1 - (a · b) / (||a|| × ||b||)

where:
  a · b = Σ(aᵢ × bᵢ)           # Dot product
  ||a|| = √(Σ aᵢ²)             # L2 norm
```

**When to use**: Medium to large models (50k - 150k tokens), or when you want direction-based similarity

**Implementation**:
```python
def cosine_distance(a, b):
    a_f32 = a.astype(mx.float32)
    b_f32 = b.astype(mx.float32)

    a_norm = mx.sqrt(mx.sum(a_f32 * a_f32))
    b_norm = mx.sqrt(mx.sum(b_f32 * b_f32))

    if mx.isinf(a_norm) or mx.isinf(b_norm):
        # Fallback: normalize by mean instead of sum
        a_normalized = a_f32 / mx.sqrt(mx.mean(a_f32 * a_f32))
        b_normalized = b_f32 / mx.sqrt(mx.mean(b_f32 * b_f32))
        return float(1.0 - mx.mean(a_normalized * b_normalized))

    a_normalized = a_f32 / a_norm
    b_normalized = b_f32 / b_norm
    return float(1.0 - mx.sum(a_normalized * b_normalized))
```

**Why it's better for large models**: Normalization prevents overflow by dividing before accumulation.

**Example**:
```python
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    metric="cosine"
)
```

#### 3. Mean Squared Error (MSE) - Most Stable

**Formula**:
```
d(a, b) = (1/N) × Σ(aᵢ - bᵢ)²

where N = number of elements
```

**When to use**: Very large models (vocabulary > 100k tokens), or when numerical stability is critical

**Implementation**:
```python
def mse_distance(a, b):
    diff = a.astype(mx.float32) - b.astype(mx.float32)
    return float(mx.mean(diff * diff))
```

**Why it's most stable**: Averages over all elements instead of summing, preventing overflow even with millions of dimensions.

**Example**:
```python
# Recommended for Qwen (151k vocab), GPT-4 scale models
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    metric="mse"  # Most stable
)
```

### Metric Selection Guide

| Model Characteristics | Recommended Metric | Reason |
|----------------------|-------------------|---------|
| **Vocab < 50k tokens**<br>(e.g., Llama-3.2-1B) | `"l2"` (default) | Fast, accurate, no overflow risk |
| **Vocab 50k - 100k**<br>(e.g., Llama-3-8B) | `"l2"` or `"cosine"` | L2 with overflow protection works well |
| **Vocab > 100k tokens**<br>(e.g., Qwen-30B: 151k) | `"mse"` or `"cosine"` | Most numerically stable |
| **Direction matters**<br>(studying vector directions) | `"cosine"` | Measures angle, not magnitude |
| **Magnitude matters**<br>(studying activation sizes) | `"l2"` or `"mse"` | Measures absolute difference |

### Real-World Example: Qwen Model

The Qwen3-30B model has **151,936 tokens**. Here's what happens with each metric:

```python
from mlxterp import InterpretableModel
from mlx_lm import load

base_model, tokenizer = load('mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)
```

**With L2 (without protection)**:
```
Output shape: (1, 6, 151936)  # 151k logits!
Baseline: inf                  # Overflow!
Recovery: nan, nan, nan...     # All NaN
```

**With MSE** ✅:
```
Baseline: 0.6480
Layer 10: 17.9% recovery  ← Works perfectly!
Layer 30:  7.5% recovery
Layer  0: -298.6% recovery
```

**With Cosine** ✅:
```
Baseline: 0.0079
Layer 10: 11.9% recovery  ← Also works!
Layer 40:  9.1% recovery
Layer  0: -45.0% recovery
```

### Recovery Calculation

Recovery percentage is computed as:

```python
baseline_dist = distance(corrupted_output, clean_output)
patched_dist = distance(patched_output, clean_output)

recovery = (baseline_dist - patched_dist) / baseline_dist * 100
```

**Interpretation**:
- **High positive %**: Patching reduced distance significantly → layer is important
- **Negative %**: Patching increased distance → layer encodes the corruption
- **~0%**: Patching had no effect → layer is not relevant

### Why Not KL Divergence?

KL divergence is commonly used in research papers, but it has numerical issues:

```python
# KL Divergence (NOT recommended)
def kl_divergence(p, q):
    p = mx.softmax(p, axis=-1)
    q = mx.softmax(q, axis=-1)
    return mx.sum(p * (mx.log(p) - mx.log(q)))  # NaN from log(0)!
```

**Problems**:
1. `log(0)` produces `-inf`
2. Very small probabilities (< 1e-7) cause numerical instability
3. Requires adding epsilon: `log(p + ε)` - but what epsilon?
4. With 150k vocab, many probabilities are ~0

**Better alternatives**: L2, MSE, or cosine distance are more robust.

## Interpreting Example Results

```
Layer  0 MLP: +43.1% recovery
Layer  2 MLP: +16.7% recovery
Layer  6 MLP: +17.6% recovery
Layer  7 MLP: -18.4% recovery
Layer 10 MLP: -23.5% recovery
Layer 15 MLP: +24.2% recovery
```

**Interpretation**:

1. **Layer 0** (43% recovery): Critical for early feature extraction
2. **Layers 2, 6** (16-17% recovery): Contribute to task but not critical
3. **Layers 7, 10** (negative): Encode the corruption ("London")
4. **Layer 15** (24% recovery): Important for final output formation

**Insight**: The model processes the factual knowledge primarily in early (Layer 0) and late (Layer 15) layers, while middle layers (7-10) encode the specific entity mentioned ("London").

## Complete Working Example

See `examples/activation_patching_example.py` for a complete, tested implementation.

## References

- Classic paper: [Causal Tracing for GPT-2](https://arxiv.org/abs/2202.05262)
- TransformerLens: Similar techniques in PyTorch
- nnsight: Generic activation patching framework
