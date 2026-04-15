# Text Generation with Interventions

## Overview

mlxterp supports token-by-token text generation with interventions applied at every step. This enables studying how interventions (steering vectors, ablations, activation modifications) affect the model's generated text — not just individual forward passes.

## Basic Generation

```python
from mlxterp import InterpretableModel

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Greedy generation (deterministic)
result = model.generate(
    "The capital of France is",
    max_tokens=20,
    temperature=0.0,
)
print(result.text)
print(f"Generated {len(result.tokens)} tokens")
```

## Sampling Strategies

### Greedy (temperature=0)

Always picks the most likely next token. Deterministic.

```python
result = model.generate("Once upon a time", max_tokens=30, temperature=0.0)
```

### Temperature Sampling

Higher temperature = more randomness.

```python
result = model.generate("Once upon a time", max_tokens=30, temperature=0.8)
```

### Top-k Sampling

Only consider the top k most likely tokens at each step.

```python
result = model.generate("Once upon a time", max_tokens=30, temperature=0.8, top_k=40)
```

### Nucleus (Top-p) Sampling

Keep the smallest set of tokens whose cumulative probability exceeds p.

```python
result = model.generate("Once upon a time", max_tokens=30, temperature=0.8, top_p=0.95)
```

## Generation with Interventions

The key feature: apply interventions at every generation step.

### Steering with Vectors

```python
from mlxterp.core.intervention import add_vector

# Add a steering vector to layer 5 at every generation step
result = model.generate(
    "I think this restaurant is",
    max_tokens=30,
    temperature=0.7,
    interventions={"layers.5": add_vector(positive_sentiment_vector)},
)
```

### Ablation During Generation

```python
from mlxterp.core.intervention import zero_out, scale

# Zero out a specific MLP
result = model.generate(
    "The capital of France is",
    max_tokens=20,
    interventions={"layers.10.mlp": zero_out},
)

# Scale down an attention layer
result = model.generate(
    "The capital of France is",
    max_tokens=20,
    interventions={"layers.5.self_attn": scale(0.1)},
)
```

### Comparing With and Without Interventions

```python
# Normal generation
normal = model.generate("The president of the United States is", max_tokens=20)

# With knowledge-related MLP ablated
ablated = model.generate(
    "The president of the United States is",
    max_tokens=20,
    interventions={"layers.12.mlp": zero_out},
)

print(f"Normal: {normal.text}")
print(f"Ablated: {ablated.text}")
```

## Stop Tokens and Callbacks

### Stop Tokens

Generation stops when a stop token is produced:

```python
result = model.generate(
    prompt,
    max_tokens=100,
    stop_tokens=[model.tokenizer.eos_token_id],
)
```

### Callbacks

Monitor or control generation step-by-step:

```python
def monitor(step, token_id, logits):
    token_str = model.tokenizer.decode([token_id])
    confidence = float(mx.softmax(logits)[token_id])
    print(f"Step {step}: '{token_str}' (p={confidence:.3f})")
    return False  # Return True to stop

result = model.generate(prompt, max_tokens=50, callback=monitor)
```

Stop after a condition:

```python
def stop_at_period(step, token_id, logits):
    return model.tokenizer.decode([token_id]).strip() == "."

result = model.generate(prompt, max_tokens=100, callback=stop_at_period)
```

## Working with Results

```python
result = model.generate("The answer is", max_tokens=20)

# Access generated data
result.text           # Full generated text string
result.tokens         # List of token IDs
result.token_logits   # Per-token logit distributions (mx.array)
result.prompt         # Original prompt

# Structured output
print(result.summary())
print(result.to_json())
print(result.to_markdown())
```

## Input Formats

Generation accepts multiple input formats:

```python
# String (requires tokenizer)
result = model.generate("Hello world", max_tokens=10)

# Token IDs as list
result = model.generate([1, 2, 3, 4], max_tokens=10)

# Token IDs as mx.array
result = model.generate(mx.array([[1, 2, 3, 4]]), max_tokens=10)
```
