# Using mlxterp in Jupyter Notebooks

## Quick Start

### 1. Installation

In your notebook, first install mlx-lm if you haven't already:

```python
# Run this once
!uv add mlx-lm
```

### 2. Import

```python
from mlxterp import InterpretableModel, interventions as iv
from mlx_lm import load
import mlx.core as mx
import mlx.nn as nn
```

## Working with Real Models

mlxterp now **fully supports** real mlx-lm models with comprehensive activation capture!

```python
# Load a real model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Run a forward pass and capture ALL activations
with model.trace("Hello, how are you?") as trace:
    pass  # Automatically captures ~196 activations!

# Access any captured activation by name
print(f"Captured {len(trace.activations)} activations")

# Access specific modules
layer_5_attn = trace.activations['model.model.layers.5.self_attn']
q_proj_3 = trace.activations['model.model.layers.3.self_attn.q_proj']
mlp_7 = trace.activations['model.model.layers.7.mlp']
output = trace.activations['__model_output__']

print(f"Layer 5 attention: {layer_5_attn.shape}")
print(f"Layer 3 Q projection: {q_proj_3.shape}")
print(f"Output: {output.shape}")
```

### Available Activations

Real models capture fine-grained activations including:

- **Embeddings**: `model.model.embed_tokens`
- **Layer outputs**: `model.model.layers.{i}`
- **Attention components**:
  - `model.model.layers.{i}.self_attn`
  - `model.model.layers.{i}.self_attn.q_proj`
  - `model.model.layers.{i}.self_attn.k_proj`
  - `model.model.layers.{i}.self_attn.v_proj`
  - `model.model.layers.{i}.self_attn.o_proj`
  - `model.model.layers.{i}.self_attn.rope`
- **MLP components**:
  - `model.model.layers.{i}.mlp`
  - `model.model.layers.{i}.mlp.gate_proj`
  - `model.model.layers.{i}.mlp.up_proj`
  - `model.model.layers.{i}.mlp.down_proj`
- **Layer norms**:
  - `model.model.layers.{i}.input_layernorm`
  - `model.model.layers.{i}.post_attention_layernorm`
- **Final output**: `__model_output__`

## Exploring Captured Activations

```python
# See what was captured
with model.trace("Hello world") as trace:
    pass

# List all captured activation names
print("Captured activations:")
for name in list(trace.activations.keys())[:20]:
    act = trace.activations[name]
    print(f"  {name}: {act.shape}")
```

## Interventions on Real Models

You can intervene on any captured module:

```python
# Baseline
with model.trace("The capital of France is") as baseline:
    baseline_output = baseline.activations['__model_output__']

# Scale down attention in layer 5
with model.trace("The capital of France is",
                 interventions={'model.model.layers.5.self_attn': iv.scale(0.5)}) as modified:
    modified_output = modified.activations['__model_output__']

# Compare
diff = mx.linalg.norm(baseline_output - modified_output)
print(f"Output difference: {diff:.4f}")
```

### Available Interventions

```python
from mlxterp import interventions as iv

# Scale activations
iv.scale(0.5)

# Zero out
iv.zero_out

# Add steering vector
steering = mx.random.normal((2048,))
iv.add_vector(steering)

# Replace with value
iv.replace_with(1.0)

# Clamp to range
iv.clamp(-1.0, 1.0)

# Add noise
iv.noise(std=0.1)

# Compose multiple
combined = iv.compose() \
    .add(iv.scale(0.8)) \
    .add(iv.noise(0.1)) \
    .build()
```

## Custom Models

Simple custom models also work perfectly:

```python
# Define a simple transformer
class MyTransformer(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        return x

# Wrap it
model = InterpretableModel(MyTransformer())

# Trace it
input_data = mx.random.normal((1, 64))
with model.trace(input_data) as trace:
    pass

# Access activations
layer_0 = trace.activations['model.layers.0']
print(f"Layer 0 shape: {layer_0.shape}")
```

## Complete Example for Mechanistic Interpretability

```python
from mlxterp import InterpretableModel, interventions as iv
from mlx_lm import load
import mlx.core as mx

# 1. Load model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# 2. Run clean forward pass
prompt = "The Eiffel Tower is located in"
with model.trace(prompt) as clean:
    clean_attn_8 = clean.activations['model.model.layers.8.self_attn']
    clean_output = clean.activations['__model_output__']

# 3. Run with corrupted input
corrupted_prompt = "The Big Ben is located in"
with model.trace(corrupted_prompt) as corrupted:
    corrupted_attn_8 = corrupted.activations['model.model.layers.8.self_attn']

# 4. Activation patching: restore clean attention in layer 8
from mlxterp import interventions as iv

# Evaluate clean activation first
mx.eval(clean_attn_8)

with model.trace(corrupted_prompt,
                 interventions={'layers.8.self_attn': iv.replace_with(clean_attn_8)}) as patched:
    patched_output = patched.activations['__model_output__']

# 5. Analyze results
clean_logits = clean_output[0, -1, :]
patched_logits = patched_output[0, -1, :]

print("Top predictions (clean):", mx.argmax(clean_logits))
print("Top predictions (patched):", mx.argmax(patched_logits))
print(f"Logit difference: {mx.linalg.norm(clean_logits - patched_logits):.4f}")
```

## Common Patterns

### Activation Patching

Replace activations from one run into another:

```python
# Get clean activation
with model.trace("clean input") as clean:
    clean_mlp = clean.activations['model.model.layers.8.mlp']

# Patch into corrupted run
with model.trace("corrupted input",
                 interventions={'model.model.layers.8.mlp': lambda x: clean_mlp}) as patched:
    result = patched.activations['__model_output__']
```

### Steering Vectors

Add directional vectors to guide model behavior:

```python
# Compute steering vector (difference between contrasting examples)
with model.trace("I love this") as pos:
    pos_h = pos.activations['model.model.layers.10']

with model.trace("I hate this") as neg:
    neg_h = neg.activations['model.model.layers.10']

steering_vector = pos_h - neg_h

# Apply steering
with model.trace("This movie is",
                 interventions={'model.model.layers.10': iv.add_vector(steering_vector)}) as steered:
    steered_output = steered.activations['__model_output__']
```

### Probing Multiple Layers

Collect activations from multiple layers for analysis:

```python
with model.trace("Input text") as trace:
    pass

# Extract layer representations
layer_activations = []
for i in range(16):  # For Llama-3.2-1B
    layer_name = f'model.model.layers.{i}'
    if layer_name in trace.activations:
        layer_activations.append(trace.activations[layer_name])

# Analyze representations
for i, act in enumerate(layer_activations):
    norm = mx.linalg.norm(act)
    print(f"Layer {i} norm: {norm:.4f}")
```

## Tips

1. **Check what's available**: Always run once and inspect `trace.activations.keys()` to see exactly what was captured

2. **Use specific module names**: Target interventions precisely (e.g., `model.model.layers.5.self_attn.q_proj` instead of just `layers.5`)

3. **Memory management**: Traces store all activations - clear them when done:
   ```python
   with model.trace(input) as trace:
       act = trace.activations['some.module']
   # trace.activations is cleared after exiting context
   ```

4. **Model downloads**: First run downloads ~1-2GB model, cached at `~/.cache/huggingface/`

## Current Status

✅ **Fully Working**:
- Real mlx-lm models (Llama, Mistral, etc.)
- Custom simple models
- Fine-grained activation capture (~196 per forward pass)
- All intervention types
- Activation patching
- Steering vectors

🎯 **Best Practices**:
- Use `trace.activations` dict for direct access by name
- Target specific submodules for precise interventions
- Cache clean activations when doing multiple patching experiments

## Troubleshooting

**Issue**: Model download fails
**Fix**: Check internet connection, models download on first use

**Issue**: `KeyError` when accessing activation
**Fix**: Print `trace.activations.keys()` to see available names

**Issue**: Out of memory
**Fix**: Use smaller batch sizes or clear traces after use

**Issue**: Interventions don't affect output
**Fix**: Verify intervention targets correct module name, check effect magnitude
