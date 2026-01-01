# mlxterp

**Mechanistic Interpretability for MLX on Apple Silicon**

A clean, intuitive library for mechanistic interpretability research on Apple Silicon Macs, powered by [MLX](https://github.com/ml-explore/mlx). Inspired by [nnsight](https://nnsight.net) and [nnterp](https://github.com/Butanium/nnterp/), mlxterp brings their elegant API design to the MLX ecosystem.

## Why mlxterp?

- **🎯 Simple & Clean API**: Context managers and direct activation access - no verbose method chains
- **🔌 Model Agnostic**: Works with ANY MLX model - no model-specific implementations needed
- **🔬 Fine-Grained Access**: Captures ~196 activations per forward pass (Q/K/V, MLP, attention, etc.)
- **🚀 Apple Silicon Optimized**: Leverages MLX's unified memory and Metal acceleration
- **📦 Minimal Boilerplate**: Lean codebase focused on accessibility
- **🔧 Flexible Interventions**: Easy activation patching, steering, and modification

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/coairesearch/mlxterp
cd mlxterp
uv sync

# Install mlx-lm for real models
uv add mlx-lm
```

### Using pip

```bash
git clone https://github.com/coairesearch/mlxterp
cd mlxterp
pip install -e .
pip install mlx-lm  # For loading real models
```

## Quick Start

### With Real Models

```python
from mlxterp import InterpretableModel
from mlx_lm import load

# Load any mlx-lm model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Run forward pass and capture ALL activations
with model.trace("Hello, how are you?") as trace:
    pass  # Automatically captures ~196 activations!

# Access any activation by name
print(f"Captured {len(trace.activations)} activations")

# Fine-grained access to model internals
layer_5_attn = trace.activations['model.model.layers.5.self_attn']
q_proj_3 = trace.activations['model.model.layers.3.self_attn.q_proj']
mlp_7_gate = trace.activations['model.model.layers.7.mlp.gate_proj']
output = trace.activations['__model_output__']

print(f"Layer 5 attention: {layer_5_attn.shape}")
print(f"Layer 3 Q projection: {q_proj_3.shape}")
print(f"Output: {output.shape}")
```

### With Custom Models

```python
from mlxterp import InterpretableModel
import mlx.core as mx
import mlx.nn as nn

# Create your own model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(64, 64) for _ in range(4)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Wrap with InterpretableModel
model = InterpretableModel(SimpleModel())

# Trace execution
input_data = mx.random.normal((1, 64))
with model.trace(input_data) as trace:
    pass

# Access activations
layer_0 = trace.activations['model.layers.0']
print(f"Layer 0 shape: {layer_0.shape}")
```

## Core Features

### 1. Comprehensive Activation Capture

mlxterp captures fine-grained activations from real models:

```python
with model.trace("Input text") as trace:
    pass

# Available activations (for transformer models):
# - Embeddings: model.model.embed_tokens
# - Layer outputs: model.model.layers.{i}
# - Attention: model.model.layers.{i}.self_attn
# - Q/K/V projections: model.model.layers.{i}.self_attn.{q,k,v}_proj
# - Output projection: model.model.layers.{i}.self_attn.o_proj
# - RoPE: model.model.layers.{i}.self_attn.rope
# - MLP: model.model.layers.{i}.mlp
# - MLP gates: model.model.layers.{i}.mlp.{gate,up,down}_proj
# - Layer norms: model.model.layers.{i}.{input,post_attention}_layernorm
# - Final output: __model_output__

# Explore what was captured
for name in list(trace.activations.keys())[:10]:
    print(f"{name}: {trace.activations[name].shape}")
```

### 2. Tokenizer Integration

Work seamlessly with text and tokens:

```python
# Encode text to tokens
text = "The capital of France is"
tokens = model.encode(text)
print(f"Tokens: {tokens}")

# Decode tokens back to text
decoded = model.decode(tokens)
print(f"Decoded: '{decoded}'")

# Analyze individual tokens
for i, token_id in enumerate(tokens):
    token_str = model.token_to_str(token_id)
    print(f"Position {i}: {token_id} -> '{token_str}'")

# Get vocabulary info
print(f"Vocabulary size: {model.vocab_size}")

# Find token position and extract its activation
with model.trace(text) as trace:
    layer_8_out = trace.activations['model.model.layers.8']

# Extract activation at "France" token position
for i, token_id in enumerate(tokens):
    if "France" in model.token_to_str(token_id):
        france_activation = layer_8_out[0, i, :]
        print(f"Activation at 'France': {france_activation.shape}")
```

### 3. Interventions

Modify activations during forward passes:

```python
from mlxterp import interventions as iv

# Scale down attention in layer 5
with model.trace("Test input",
                 interventions={'model.model.layers.5.self_attn': iv.scale(0.5)}) as trace:
    output = trace.activations['__model_output__']

# Zero out MLP in layer 3
with model.trace("Test", interventions={'model.model.layers.3.mlp': iv.zero_out}):
    output = model.output.save()

# Add steering vectors
steering_vector = mx.random.normal((2048,))
with model.trace("Test",
                 interventions={'model.model.layers.10': iv.add_vector(steering_vector)}):
    steered_output = model.output.save()

# Compose multiple interventions
combined = iv.compose().add(iv.scale(0.5)).add(iv.noise(std=0.1)).build()
with model.trace("Test", interventions={'model.model.layers.3': combined}):
    output = model.output.save()
```

**Available interventions:**
- `iv.zero_out` - Set to zero
- `iv.scale(factor)` - Multiply by factor
- `iv.add_vector(vector)` - Add steering vector
- `iv.replace_with(value)` - Replace with value
- `iv.clamp(min, max)` - Clamp to range
- `iv.noise(std)` - Add Gaussian noise
- `iv.compose()` - Chain multiple interventions

### 4. Logit Lens & Predictions

Decode hidden states to tokens and see what each layer predicts at each position:

```python
# Get token predictions from any layer
with model.trace("The capital of France is") as trace:
    layer_6 = trace.activations["model.model.layers.6"]

# Decode last token's hidden state
predictions = model.get_token_predictions(layer_6[0, -1, :], top_k=5)
for token_id in predictions:
    print(model.token_to_str(token_id))

# Logit lens: see what each layer predicts at each position
text = "The capital of France is"
results = model.logit_lens(text, layers=[0, 5, 10, 15])

# Show how predictions for the LAST position evolve
for layer_idx in [0, 5, 10, 15]:
    last_pos_pred = results[layer_idx][-1][0][2]  # Top token at last position
    print(f"Layer {layer_idx}: '{last_pos_pred}'")

# Output shows progressive refinement:
# Layer  0: ' the'
# Layer  5: ' a'
# Layer 10: ' Paris'
# Layer 15: ' Paris'

# Visualize with heatmap (shows predictions at ALL positions)
results = model.logit_lens(
    "The Eiffel Tower is located in the city of",
    plot=True,
    max_display_tokens=15,
    figsize=(16, 10)
)
# X-axis: Input token positions
# Y-axis: Model layers
# Cells: Predicted tokens at each (layer, position)
```

**Features:**
- `get_token_predictions()` - Decode any hidden state to tokens
- `logit_lens()` - Analyze predictions at every position across layers
- Visualization with heatmap plots showing full prediction matrix
- Works with quantized models
- Handles weight-tied embeddings automatically

### 5. Activation Patching

Identify important layers with one function call:

```python
# Find which layers are critical for factual recall
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    plot=True
)

# Analyze results
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\nMost important layers:")
for layer_idx, recovery in sorted_results[:3]:
    print(f"  Layer {layer_idx}: {recovery:.1f}% recovery")

# Output:
# Layer  0: +43.1% recovery  ← Very important!
# Layer 15: +24.2% recovery  ← Important
# Layer  6: +17.6% recovery  ← Somewhat important

# Test different components
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="self_attn",  # Test attention instead of MLP
    layers=[0, 5, 10, 15]
)
```

**Recovery interpretation:**
- Positive % = layer is important for the task
- Negative % = layer encodes the corruption
- ~0% = layer is not relevant

### 4. Steering Vectors

Compute and apply steering vectors:

```python
# Compute steering vector from contrastive examples
with model.trace("I love this") as pos:
    pos_h = pos.activations['model.model.layers.10']

with model.trace("I hate this") as neg:
    neg_h = neg.activations['model.model.layers.10']

steering_vector = pos_h - neg_h

# Apply steering to guide model behavior
with model.trace("This movie is",
                 interventions={'model.model.layers.10': iv.add_vector(steering_vector)}) as steered:
    steered_output = steered.activations['__model_output__']
```

### 5. Layer Analysis

Analyze representations across layers:

```python
with model.trace("Input text") as trace:
    pass

# Collect all layer outputs
layer_norms = []
for i in range(16):  # For Llama-3.2-1B
    layer_name = f'model.model.layers.{i}'
    if layer_name in trace.activations:
        layer_norms.append(mx.linalg.norm(trace.activations[layer_name]).item())

# Visualize layer norms
for i, norm in enumerate(layer_norms):
    print(f"Layer {i}: {'#' * int(norm / 100)}")
```

## Advanced Usage

### Working with Specific Components

Target specific attention heads or MLP components:

```python
with model.trace("Analyze this text") as trace:
    pass

# Access Q/K/V projections
q_proj_5 = trace.activations['model.model.layers.5.self_attn.q_proj']
k_proj_5 = trace.activations['model.model.layers.5.self_attn.k_proj']
v_proj_5 = trace.activations['model.model.layers.5.self_attn.v_proj']

# Access MLP components
gate_proj_3 = trace.activations['model.model.layers.3.mlp.gate_proj']
up_proj_3 = trace.activations['model.model.layers.3.mlp.up_proj']
down_proj_3 = trace.activations['model.model.layers.3.mlp.down_proj']

print(f"Q projection shape: {q_proj_5.shape}")
print(f"Gate projection shape: {gate_proj_3.shape}")
```

### Intervention on Specific Components

```python
# Zero out Q projections in layer 5
with model.trace("Test",
                 interventions={'model.model.layers.5.self_attn.q_proj': iv.zero_out}):
    output = model.output.save()

# Scale MLP gate in layer 3
with model.trace("Test",
                 interventions={'model.model.layers.3.mlp.gate_proj': iv.scale(0.5)}):
    output = model.output.save()
```

### Custom Interventions

Create your own intervention functions:

```python
def my_custom_intervention(activation: mx.array) -> mx.array:
    """Custom intervention: normalize and scale"""
    normalized = activation / mx.linalg.norm(activation)
    return normalized * 10.0

with model.trace("Input",
                 interventions={'model.model.layers.5': my_custom_intervention}):
    output = model.output.save()
```

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Installation](docs/installation.md)** - Detailed installation instructions
- **[Jupyter Guide](docs/JUPYTER_GUIDE.md)** - Using mlxterp in notebooks
- **[API Reference](docs/API.md)** - Complete API documentation
- **[Examples](docs/examples.md)** - Usage examples and patterns
- **[Architecture](docs/architecture.md)** - Design and implementation details

## Testing

mlxterp includes comprehensive tests. See [tests/README.md](tests/README.md) for details.

```bash
# Run main integration test
uv run python tests/test_comprehensive.py

# Test activation validity
uv run python tests/test_activation_validity.py

# Test interventions
uv run python tests/test_interventions.py
```

## Design Philosophy

mlxterp follows these principles:

1. **Context Managers for Tracing**: Clean, Pythonic API
2. **Comprehensive Activation Capture**: Every module is intercepted automatically
3. **Generic Model Wrapping**: Works with ANY MLX model via recursive discovery
4. **Direct Dict Access**: `trace.activations['name']` for simplicity
5. **Lazy Evaluation**: Leverages MLX's lazy computation
6. **Minimal Abstractions**: Simple, maintainable codebase

## Implementation Highlights

- **Recursive Module Discovery**: Uses MLX's `.children()` to find all submodules
- **Composition-Based Wrapping**: Lightweight wrappers that delegate to original modules
- **Cycle Detection**: Handles circular references in module trees
- **Fine-Grained Interception**: Captures Q/K/V projections, MLP components, layer norms, etc.

## Current Status

✅ **Fully Working**:
- Real mlx-lm models (Llama, Mistral, etc.)
- Custom simple models
- Fine-grained activation capture (~196 activations per forward pass)
- All intervention types
- Activation patching
- Steering vectors

🎯 **Production Ready**:
- Stable API
- Comprehensive test suite
- Clean documentation
- No known issues

## Comparison with Other Libraries

| Feature | mlxterp | TransformerLens | nnsight |
|---------|---------|-----------------|---------|
| Framework | MLX | PyTorch | PyTorch |
| Model-Specific Code | ❌ No | ✅ Yes | ❌ No |
| Apple Silicon Native | ✅ Yes | ❌ No | ❌ No |
| Unified Memory | ✅ Yes | ❌ No | ❌ No |
| Fine-Grained Access | ✅ Yes (~196) | ✅ Yes | ⚠️ Limited |
| Context Managers | ✅ Yes | ❌ No | ✅ Yes |

## Contributing

Contributions welcome! Areas for contribution:

- Additional intervention types
- Visualization tools
- More examples and tutorials
- Support for additional model architectures
- Performance optimizations

Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## Citation

If you use mlxterp in your research, it would be nice if you could please cite:

```bibtex
@software{mlxterp,
  title = {mlxterp: Mechanistic Interpretability for MLX},
  author = {Sigurd Schacht, COAI Research},
  year = {2025},
  url = {https://github.com/coairesearch/mlxterp}
}
```

## Related Projects

- [nnsight](https://nnsight.net): Inspiration for the generic wrapping approach
- [nnterp](https://github.com/Butanium/nnterp/): Clean API design inspiration
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens): Comprehensive interpretability library
- [MLX](https://github.com/ml-explore/mlx): Apple's ML framework

## License

MIT License - see [LICENSE](LICENSE) for details.
