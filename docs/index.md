# mlxterp

**Mechanistic Interpretability for MLX on Apple Silicon**

A clean, intuitive library for mechanistic interpretability research on Apple Silicon Macs, powered by [MLX](https://github.com/ml-explore/mlx). Inspired by [nnsight](https://nnsight.net) and [nnterp](https://github.com/Butanium/nnterp/), mlxterp brings their elegant API design to the MLX ecosystem.

## Why mlxterp?

- **🎯 Simple & Clean API**: Context managers and direct attribute access - no verbose method chains
- **🔌 Model Agnostic**: Works with ANY MLX model - no model-specific implementations needed
- **🚀 Apple Silicon Optimized**: Leverages MLX's unified memory and Metal acceleration
- **📦 Minimal Boilerplate**: Lean codebase focused on accessibility
- **🔧 Flexible Interventions**: Easy activation patching, steering, and modification

## Quick Example

```python
from mlxterp import InterpretableModel
import mlx.core as mx

# Load any MLX model
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Trace execution and capture activations
with model.trace("The capital of France is"):
    # Direct attribute access to layers
    attn_3 = model.layers[3].attn.output.save()
    mlp_8 = model.layers[8].mlp.output.save()
    logits = model.output.save()

print(f"Attention layer 3 output shape: {attn_3.shape}")
```

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/coairesearch/mlxterp
cd mlxterp

# Create environment and install
uv sync

# Activate the environment
source .venv/bin/activate
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/coairesearch/mlxterp
cd mlxterp

# Install in development mode
pip install -e .
```

## Features at a Glance

### Context Manager API

```python
with model.trace("Hello world") as trace:
    layer_5 = model.layers[5].output.save()

# Access saved values after trace
print(trace.saved_values)
```

### Activation Collection

```python
from mlxterp import get_activations

activations = get_activations(
    model,
    prompts=["Hello", "World", "Test"],
    layers=[3, 8, 12],
    positions=-1  # Last token
)
```

### Interventions

```python
from mlxterp import interventions as iv

# Scale activations
with model.trace("Test", interventions={"layers.3": iv.scale(0.5)}):
    output = model.output.save()

# Add steering vectors
steering_vector = mx.random.normal((hidden_dim,))
with model.trace("Test", interventions={"layers.5": iv.add_vector(steering_vector)}):
    output = model.output.save()
```

### Activation Patching

Find important layers with one function call:

```python
# Identify which layers are critical for factual recall
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    plot=True
)

# Analyze results
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for layer_idx, recovery in sorted_results[:3]:
    print(f"Layer {layer_idx}: {recovery:.1f}% recovery")

# Output:
# Layer  0: +43.1% recovery  ← Very important!
# Layer 15: +24.2% recovery  ← Important
# Layer  6: +17.6% recovery  ← Somewhat important
```

The `activation_patching()` helper automates the entire workflow. Learn more in the [Activation Patching Guide](guides/activation_patching.md) for comprehensive coverage including interpretation and best practices.

## Next Steps

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [API Reference](API.md) - Complete API documentation
- [Examples](examples.md) - Detailed usage examples
- [Activation Patching Guide](guides/activation_patching.md) - In-depth guide with interpretation
- [Architecture](architecture.md) - Design principles and implementation

## Community

- **Issues**: [GitHub Issues](https://github.com/coairesearch/mlxterp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/coairesearch/mlxterp/discussions)

## Citation

```bibtex
@software{mlxterp,
  title = {mlxterp: Mechanistic Interpretability for MLX},
  author = {Sigurd Schacht},
  year = {2025},
  url = {https://github.com/coairesearch/mlxterp}
}
```

## License

MIT License - see [LICENSE](https://github.com/coairesearch/mlxterp/blob/main/LICENSE) for details.
