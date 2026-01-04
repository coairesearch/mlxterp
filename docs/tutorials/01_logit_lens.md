# Tutorial 1: The Logit Lens

**Paper**: [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) by nostalgebraist (2020)

**Difficulty**: Beginner | **Time**: 1-2 hours

---

!!! note "Model Choice"
    The original paper analyzed GPT-2. This tutorial uses Llama-3.2-1B-Instruct to demonstrate the technique on a modern model. The core findings (predictions crystallizing in later layers, early layers predicting related tokens) generalize across transformer architectures, though specific layer numbers may differ. For exact paper reproduction, use a GPT-2 model via `InterpretableModel("gpt2")`.

---

## Learning Objectives

By the end of this tutorial, you will:

1. Understand the residual stream concept in transformers
2. Know how to project intermediate hidden states to token predictions
3. Observe how predictions evolve across layers
4. Use mlxterp's built-in `logit_lens()` method
5. Interpret logit lens visualizations

---

## Introduction

### What is the Logit Lens?

The **logit lens** is a technique for peeking inside transformer models to see what they're "thinking" at each layer. The key insight is that hidden states at intermediate layers can be projected through the output embedding matrix (the "unembedding" matrix) to produce token logits (unnormalized scores that can be converted to probabilities via softmax).

!!! info "Logits vs Probabilities"
    Throughout this tutorial, "scores" refer to **logits** (raw model outputs before softmax), not probabilities. Higher logits indicate stronger predictions. To get probabilities, apply `mx.softmax(logits)`. The relative ranking of tokens is preserved either way.

### Why does this work?

Transformers use a **residual stream** architecture. Each layer reads from and writes to a running sum:

```
x_0 = embedding(input)
x_1 = x_0 + attention_1(x_0) + mlp_1(x_0)
x_2 = x_1 + attention_2(x_1) + mlp_2(x_1)
...
x_n = x_{n-1} + attention_n(x_{n-1}) + mlp_n(x_{n-1})
output = unembed(layer_norm(x_n))
```

The logit lens asks: **What if we applied the unembedding at intermediate layers?**

```python
# At any layer i:
intermediate_prediction = unembed(layer_norm(x_i))
```

This reveals how the model iteratively refines its predictions from input to output.

!!! warning "Approximation Caveat"
    The logit lens is an **approximation**. The final layer norm and unembedding matrix are trained only on the final layer's output, not intermediate layers. This means early layer predictions may be systematically biased. The [Tuned Lens](02_tuned_lens.md) addresses this by learning layer-specific corrections.

### Key Finding from the Paper

The original paper found that:

> "The model doesn't seem to be encoding the output distribution in some clever, inscrutable way that gets decoded at the very end. Instead, the residual stream appears to contain something like a 'running probability distribution' over tokens, which each layer refines."

---

## Prerequisites

```python
# Install mlxterp if you haven't already
# pip install mlxterp

from mlxterp import InterpretableModel
import mlx.core as mx
```

---

## Part 1: Manual Implementation

Let's first understand the logit lens by implementing it manually.

### Step 1: Load a Model

```python
from mlxterp import InterpretableModel

# Load a model (any MLX-compatible model works)
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

print(f"Model loaded with {len(model.layers)} layers")
```

### Step 2: Get Hidden States at Each Layer

```python
# Our test prompt
prompt = "The capital of France is"

# Trace through the model to capture all layer outputs
with model.trace(prompt) as trace:
    pass

# Show what activations we captured
print(f"Captured {len(trace.activations)} activation tensors")
print("\nLayer activation keys:")
for key in sorted(trace.activations.keys()):
    if 'layers' in key and key.endswith(('.self_attn', '.mlp', '.input_layernorm')):
        continue  # Skip component-level activations for now
    shape = trace.activations[key].shape if hasattr(trace.activations[key], 'shape') else 'N/A'
    print(f"  {key}: {shape}")
```

### Step 3: Project to Token Predictions

The key operation is projecting hidden states through the unembedding:

```python
def manual_logit_lens(model, hidden_state):
    """
    Manually apply the logit lens to a hidden state.

    Args:
        model: InterpretableModel instance
        hidden_state: Hidden state tensor of shape (hidden_dim,) or (seq_len, hidden_dim)

    Returns:
        Top-k token predictions with probabilities
    """
    # Step 1: Apply final layer normalization
    # This is critical - the model expects normalized inputs to the output projection
    final_norm = model._module_resolver.get_final_norm()
    normalized = final_norm(hidden_state)

    # Step 2: Get predictions through the output projection (unembedding)
    predictions = model.get_token_predictions(normalized, top_k=5, return_scores=True)

    # Step 3: Convert to readable format
    results = []
    for token_id, score in predictions:
        token_str = model.token_to_str(token_id)
        results.append((token_id, score, token_str))

    return results
```

### Step 4: Apply at Each Layer

```python
# Get tokens
tokens = model.encode(prompt)
print(f"Input tokens: {[model.token_to_str(t) for t in tokens]}")

# Analyze the last token position (where the model predicts the next token)
print("\n" + "="*60)
print("Logit Lens: Predictions at each layer")
print("="*60)

# Find layer activations
from mlxterp.core.module_resolver import find_layer_key_pattern

for layer_idx in range(0, len(model.layers), 4):  # Every 4th layer
    layer_key = find_layer_key_pattern(trace.activations, layer_idx)
    if layer_key is None:
        continue

    layer_output = trace.activations[layer_key]  # Shape: (1, seq_len, hidden_dim)
    last_token_hidden = layer_output[0, -1, :]   # Shape: (hidden_dim,)

    # Apply logit lens
    predictions = manual_logit_lens(model, last_token_hidden)

    # Show top prediction
    top_token = predictions[0][2]
    top_score = predictions[0][1]
    print(f"Layer {layer_idx:2d}: '{top_token}' (score: {top_score:.3f})")
```

---

## Part 2: Using the Built-in Method

mlxterp provides a built-in `logit_lens()` method that handles all of this automatically:

```python
# Simple usage
results = model.logit_lens(
    "The capital of France is",
    layers=[0, 4, 8, 12, 15],  # Analyze specific layers
    top_k=3                     # Get top 3 predictions
)

# Print results
print("\nLogit Lens Results (built-in method):")
print("-" * 50)
for layer_idx in sorted(results.keys()):
    layer_preds = results[layer_idx]
    last_pos_preds = layer_preds[-1]  # Last token position

    print(f"\nLayer {layer_idx}:")
    for i, (token_id, score, token_str) in enumerate(last_pos_preds[:3]):
        print(f"  {i+1}. '{token_str}' (score: {score:.4f})")
```

### Visualization

The `logit_lens()` method can generate visualizations:

```python
# Generate a heatmap visualization
results = model.logit_lens(
    "The capital of France is",
    plot=True,              # Enable visualization
    max_display_tokens=10,  # Limit tokens shown
    figsize=(14, 8)
)
```

This produces a heatmap showing:
- X-axis: Token positions in the input
- Y-axis: Layer index
- Color: Confidence of the top prediction
- Cell text: Top predicted token

---

## Part 3: Reproducing Paper Findings

### Finding 1: Predictions Crystallize in Later Layers

The paper observed that correct predictions often appear suddenly in middle-to-late layers:

```python
# Test with a factual prompt
prompt = "The Eiffel Tower is located in"

results = model.logit_lens(prompt, top_k=1)

print("When does the correct answer ('Paris') appear?")
print("-" * 50)

for layer_idx in sorted(results.keys()):
    top_pred = results[layer_idx][-1][0][2]  # Top prediction at last position
    marker = " <-- CORRECT" if 'Paris' in top_pred else ""
    print(f"Layer {layer_idx:2d}: {top_pred:15s}{marker}")
```

### Finding 2: Early Layers Often Predict Related Tokens

```python
# Early layers often predict semantically related but incorrect tokens
prompt = "Barack Obama was the 44th president of the United"

results = model.logit_lens(prompt, top_k=5)

print("\nEarly vs Late Layer Predictions:")
print("=" * 60)

early_layer = min(results.keys())
late_layer = max(results.keys())

print(f"\nEarly Layer ({early_layer}):")
for token_id, score, token_str in results[early_layer][-1][:5]:
    print(f"  '{token_str}': {score:.4f}")

print(f"\nLate Layer ({late_layer}):")
for token_id, score, token_str in results[late_layer][-1][:5]:
    print(f"  '{token_str}': {score:.4f}")
```

### Finding 3: Position-Dependent Evolution

Different positions evolve differently:

```python
prompt = "The quick brown fox jumps over the lazy"

results = model.logit_lens(prompt, layers=list(range(0, len(model.layers), 2)))
tokens = model.encode(prompt)

print("\nEvolution at different positions:")
print("=" * 60)

# Compare early vs late position
positions_to_check = [2, len(tokens)-1]  # "brown" and final position

for pos in positions_to_check:
    token_str = model.token_to_str(tokens[pos]) if pos < len(tokens) else "?"
    print(f"\nPosition {pos} ('{token_str}'):")

    for layer_idx in sorted(results.keys())[:4]:
        if pos < len(results[layer_idx]):
            top_pred = results[layer_idx][pos][0][2]
            print(f"  Layer {layer_idx:2d}: '{top_pred}'")
```

---

## Part 4: Exercises

### Exercise 1: Compare Different Prompt Types

Try the logit lens on different types of prompts:

```python
prompts = [
    # Factual
    "The largest planet in our solar system is",
    # Linguistic
    "She went to the store and bought some",
    # Arithmetic
    "Two plus two equals",
    # Code
    "def hello_world():\n    print(",
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    results = model.logit_lens(prompt, layers=[0, 8, 15])
    final_pred = results[max(results.keys())][-1][0][2]
    print(f"Final prediction: '{final_pred}'")
```

### Exercise 2: Find the "Crystallization Point"

Write a function that finds the layer where the correct answer first appears:

```python
def find_crystallization_layer(model, prompt, expected_token):
    """Find the first layer where the expected token becomes the top prediction."""
    results = model.logit_lens(prompt)

    for layer_idx in sorted(results.keys()):
        top_pred = results[layer_idx][-1][0][2]
        if expected_token.lower() in top_pred.lower():
            return layer_idx

    return None

# Test it
layer = find_crystallization_layer(
    model,
    "The capital of Japan is",
    "Tokyo"
)
print(f"'Tokyo' crystallizes at layer: {layer}")
```

### Exercise 3: Confidence Evolution

Plot how the confidence of the correct prediction changes across layers:

```python
import matplotlib.pyplot as plt

def plot_confidence_evolution(model, prompt, target_token):
    """Plot confidence in a specific token across layers."""
    results = model.logit_lens(prompt, top_k=100)  # Get more predictions

    layers = sorted(results.keys())
    confidences = []

    for layer_idx in layers:
        preds = results[layer_idx][-1]
        # Find confidence of target token
        conf = 0.0
        for token_id, score, token_str in preds:
            if target_token.lower() in token_str.lower():
                conf = score
                break
        confidences.append(conf)

    plt.figure(figsize=(10, 5))
    plt.plot(layers, confidences, 'b-o')
    plt.xlabel('Layer')
    plt.ylabel(f'Confidence in "{target_token}"')
    plt.title(f'Confidence Evolution: "{prompt}"')
    plt.grid(True, alpha=0.3)
    plt.show()

# Try it
# plot_confidence_evolution(model, "The capital of France is", "Paris")
```

---

## Summary

In this tutorial, you learned:

1. **The residual stream concept**: Transformers maintain a running representation that each layer refines
2. **How the logit lens works**: Project intermediate hidden states through the unembedding matrix
3. **Key finding**: Models iteratively refine predictions, with correct answers often "crystallizing" in middle-to-late layers
4. **How to use mlxterp**: Both manual implementation and the built-in `logit_lens()` method

---

## Next Steps

- **Tutorial 2: Tuned Lens** - Learn how to improve logit lens predictions with learned probes
- **Tutorial 3: Causal Tracing** - Use activation patching to localize factual knowledge

---

## References

1. nostalgebraist (2020). [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). LessWrong.

2. Elhage et al. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). Anthropic.
