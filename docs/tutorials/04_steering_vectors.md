# Tutorial 4: Steering Vectors (CAA)

**Paper**: [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) by Rimsky et al. (ACL 2024)

**Difficulty**: Intermediate | **Time**: 2-3 hours

---

## Learning Objectives

By the end of this tutorial, you will:

1. Understand contrastive activation addition (CAA)
2. Compute steering vectors from contrastive examples
3. Apply steering vectors to control model behavior
4. Experiment with different steering strengths
5. Create your own behavioral steering vectors

---

## Introduction

### What is Steering?

**Steering** is a technique to control model behavior by adding vectors to activations during inference. Unlike fine-tuning, steering:
- Requires no gradient computation
- Can be applied/removed instantly
- Allows continuous control via strength parameter
- Works with frozen model weights

### The Contrastive Activation Addition Method

The CAA paper introduces a principled approach:

1. **Collect contrastive examples**: Pairs of prompts eliciting opposite behaviors
2. **Extract activations**: Get hidden states for both positive and negative examples
3. **Compute steering vector**: Subtract negative mean from positive mean
4. **Apply during inference**: Add the vector to intermediate activations

```
steering_vector = mean(positive_activations) - mean(negative_activations)
steered_output = model(input, hidden += steering_vector * strength)
```

### Why This Works

The paper hypothesizes that behavioral traits are encoded as directions in activation space. By identifying these directions (via contrast), we can amplify or suppress behaviors.

---

## Prerequisites

```python
from mlxterp import InterpretableModel
from mlxterp import interventions as iv
import mlx.core as mx
```

!!! note "Model Choice"
    The original CAA paper used Llama 2 models. This tutorial uses Llama-3.2-1B-Instruct for accessibility and speed. The methodology generalizes across transformer models, though optimal layers and strengths may differ.

---

## Part 1: Basic Steering Example

### Step 1: Define Contrastive Prompts

```python
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Contrastive examples for sentiment steering
positive_prompts = [
    "I think this is great because",
    "I love the way this works since",
    "This makes me happy because",
    "I'm excited about this because",
    "This is wonderful because",
]

negative_prompts = [
    "I think this is terrible because",
    "I hate the way this works since",
    "This makes me sad because",
    "I'm worried about this because",
    "This is awful because",
]
```

!!! warning "Simplified Contrastive Pairs"
    The paper's methodology uses **matched** contrastive pairs where prompts differ only in the target behavior (e.g., same question with honest vs. sycophantic response). Our examples above use different prompts, which may capture some topic/wording differences along with sentiment. For rigorous steering, use the paper's matched-pair templates.

### Step 2: Extract Activations

```python
# Choose a layer for steering (middle layers often work well)
layer_idx = len(model.layers) // 2

positive_acts = []
negative_acts = []

# Collect positive activations
for prompt in positive_prompts:
    with model.trace(prompt) as trace:
        pass

    # Get layer output activation
    layer_key = None
    for key in trace.activations:
        if key.endswith(f"layers.{layer_idx}"):
            layer_key = key
            break

    if layer_key:
        act = trace.activations[layer_key]
        # Take last token position
        positive_acts.append(act[0, -1, :])

# Collect negative activations
for prompt in negative_prompts:
    with model.trace(prompt) as trace:
        pass

    layer_key = None
    for key in trace.activations:
        if key.endswith(f"layers.{layer_idx}"):
            layer_key = key
            break

    if layer_key:
        act = trace.activations[layer_key]
        negative_acts.append(act[0, -1, :])

# Ensure we have activations
mx.eval(positive_acts)
mx.eval(negative_acts)
```

### Step 3: Compute Steering Vector

```python
# Stack and compute means
positive_stack = mx.stack(positive_acts)
negative_stack = mx.stack(negative_acts)

positive_mean = mx.mean(positive_stack, axis=0)
negative_mean = mx.mean(negative_stack, axis=0)

# Steering vector: direction from negative to positive
steering_vector = positive_mean - negative_mean

mx.eval(steering_vector)
print(f"Steering vector shape: {steering_vector.shape}")
print(f"Steering vector norm: {float(mx.sqrt(mx.sum(steering_vector ** 2))):.4f}")
```

### Step 4: Apply Steering

```python
# Helper to get predictions from logits
def get_top_from_logits(model, logits, top_k=5):
    if len(logits.shape) == 3:
        logits = logits[0, -1, :]
    top_indices = mx.argsort(logits)[-top_k:][::-1]
    top_scores = logits[top_indices]
    mx.eval(top_indices, top_scores)
    return [(int(i), float(s)) for i, s in zip(top_indices.tolist(), top_scores.tolist())]

# Test prompt
test_prompt = "This product is"

# Build intervention key
intervention_key = f"layers.{layer_idx}"

# Without steering
with model.trace(test_prompt) as trace:
    normal_output = model.output.save()

mx.eval(normal_output)
normal_pred = get_top_from_logits(model, normal_output, top_k=5)
print("Normal predictions:")
for token_id, score in normal_pred:
    print(f"  '{model.token_to_str(token_id)}': {score:.2f}")

# With positive steering
strength = 2.0  # Scaling factor
with model.trace(test_prompt,
                 interventions={intervention_key: iv.add_vector(steering_vector * strength)}):
    steered_output = model.output.save()

mx.eval(steered_output)
steered_pred = get_top_from_logits(model, steered_output, top_k=5)
print(f"\nPositive steering (strength={strength}):")
for token_id, score in steered_pred:
    print(f"  '{model.token_to_str(token_id)}': {score:.2f}")

# With negative steering (opposite direction)
with model.trace(test_prompt,
                 interventions={intervention_key: iv.add_vector(-steering_vector * strength)}):
    neg_steered_output = model.output.save()

mx.eval(neg_steered_output)
neg_pred = get_top_from_logits(model, neg_steered_output, top_k=5)
print(f"\nNegative steering (strength={-strength}):")
for token_id, score in neg_pred:
    print(f"  '{model.token_to_str(token_id)}': {score:.2f}")
```

---

## Part 2: Steering Strength Analysis

Different strengths produce different effects:

```python
strengths = [0.0, 0.5, 1.0, 2.0, 4.0]

print(f"Test prompt: '{test_prompt}'")
print("-" * 50)

for strength in strengths:
    if strength == 0.0:
        with model.trace(test_prompt) as trace:
            output = model.output.save()
    else:
        with model.trace(test_prompt,
                         interventions={intervention_key: iv.add_vector(steering_vector * strength)}):
            output = model.output.save()

    mx.eval(output)
    pred = get_top_from_logits(model, output, top_k=1)
    token_str = model.token_to_str(pred[0][0])
    print(f"Strength {strength:4.1f}: '{token_str}'")
```

!!! note "Finding the Right Strength"
    - Too low: No effect
    - Too high: Gibberish or repetition
    - Typical range: 1.0-3.0 (heuristic, varies by model and behavior)

---

## Part 3: Multi-Layer Steering

The paper found that steering multiple layers can be more effective:

```python
# Compute steering vectors for multiple layers
steering_vectors = {}
layers_to_steer = [4, 6, 8, 10]

for layer_idx in layers_to_steer:
    pos_acts = []
    neg_acts = []

    for prompt in positive_prompts:
        with model.trace(prompt) as trace:
            pass
        for key in trace.activations:
            if key.endswith(f"layers.{layer_idx}"):
                pos_acts.append(trace.activations[key][0, -1, :])
                break

    for prompt in negative_prompts:
        with model.trace(prompt) as trace:
            pass
        for key in trace.activations:
            if key.endswith(f"layers.{layer_idx}"):
                neg_acts.append(trace.activations[key][0, -1, :])
                break

    mx.eval(pos_acts)
    mx.eval(neg_acts)

    if pos_acts and neg_acts:
        pos_mean = mx.mean(mx.stack(pos_acts), axis=0)
        neg_mean = mx.mean(mx.stack(neg_acts), axis=0)
        steering_vectors[layer_idx] = pos_mean - neg_mean
        mx.eval(steering_vectors[layer_idx])

# Apply multi-layer steering
strength = 1.5
interventions = {
    f"layers.{layer_idx}": iv.add_vector(vec * strength)
    for layer_idx, vec in steering_vectors.items()
}

with model.trace(test_prompt, interventions=interventions):
    multi_steered = model.output.save()

mx.eval(multi_steered)
multi_pred = get_top_from_logits(model, multi_steered, top_k=5)
print(f"\nMulti-layer steering ({list(steering_vectors.keys())}):")
for token_id, score in multi_pred:
    print(f"  '{model.token_to_str(token_id)}': {score:.2f}")
```

---

## Part 4: Different Behavioral Dimensions

The CAA paper tested sycophancy, corrigibility, and other safety-relevant traits. Here are **additional examples** of behavioral dimensions you can experiment with (not from the paper):

### Honesty/Sycophancy Steering

```python
# Prompts that elicit honest vs. sycophantic responses
honest_prompts = [
    "I need to give you honest feedback:",
    "Let me tell you the truth about this:",
    "To be completely honest with you,",
]

sycophantic_prompts = [
    "You're absolutely right! Let me agree that",
    "What a brilliant point! I think",
    "I couldn't agree more! This is",
]
```

### Formal/Casual Steering

```python
formal_prompts = [
    "I would like to formally address the matter of",
    "It is my professional opinion that",
    "Upon careful consideration of the evidence,",
]

casual_prompts = [
    "Hey so basically like",
    "Yo check it out",
    "Dude you gotta see this",
]
```

### Cautious/Confident Steering

```python
cautious_prompts = [
    "I'm not entirely sure but I think",
    "It might be possible that",
    "This could potentially be",
]

confident_prompts = [
    "I am absolutely certain that",
    "There is no doubt that",
    "It is definitely true that",
]
```

---

## Part 5: Exercises

### Exercise 1: Create Your Own Steering Vector

Design contrastive prompts for a behavior you want to control:

```python
# Your custom behavioral dimension
your_positive_prompts = [
    # Add prompts that exhibit the desired behavior
]

your_negative_prompts = [
    # Add prompts that exhibit the opposite behavior
]

# Compute and test your steering vector
# ... your code here ...
```

### Exercise 2: Optimal Layer Selection

Find which layer is most effective for steering:

```python
# Test each layer individually and compare effects
for layer_idx in range(len(model.layers)):
    # Compute steering vector for this layer
    # Apply and measure effect
    # ... your code here ...
    pass
```

### Exercise 3: Steering with Generation

Apply steering during text generation:

```python
# Generate text with steering applied
# Compare generations with different steering strengths
# ... your code here ...
```

---

## Summary

In this tutorial, you learned:

1. **Contrastive Activation Addition**: Compute steering vectors from contrastive examples
2. **Applying Steering**: Add vectors to intermediate activations to control behavior
3. **Strength Tuning**: Balance between no effect and disruption
4. **Multi-Layer Steering**: Apply to multiple layers for stronger effects
5. **Behavioral Dimensions**: Different contrasts target different behaviors

---

## Limitations and Notes

!!! warning "Important Caveats"
    - **Prompt Sensitivity**: Steering effectiveness depends heavily on contrastive prompt quality
    - **Model Specificity**: Optimal layers and strengths vary by model
    - **Behavior Complexity**: Complex behaviors may not reduce to single directions
    - **Side Effects**: Steering one behavior may affect others
    - **Evaluation Scope**: This tutorial evaluates next-token predictions only; the paper evaluates full generated completions

    For rigorous evaluation, use the paper's standardized prompt templates and metrics.

---

## Next Steps

- **Tutorial 5: Induction Heads** - Understand pattern completion circuits
- **Tutorial 6: Sparse Autoencoders** - Feature decomposition

---

## References

1. Rimsky, N., et al. (2024). [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681). ACL 2024.

2. Turner, A., et al. (2023). [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248).

3. Related: [Representation Engineering](https://arxiv.org/abs/2310.01405) by Zou et al.
