# mlxterp Quick Start Guide

Get started with mlxterp in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlxterp
cd mlxterp

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[viz]"
```

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- MLX framework (`pip install mlx`)

## Your First Script

Create a file `test_mlxterp.py`:

```python
from mlxterp import InterpretableModel
import mlx.core as mx
import mlx.nn as nn

# 1. Create a simple model
class SimpleTransformer(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.layers = [
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(4)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        return x

# 2. Wrap with InterpretableModel
base_model = SimpleTransformer()
model = InterpretableModel(base_model)

# 3. Create input
input_data = mx.random.normal((1, 10, 64))  # (batch, seq, hidden)

# 4. Trace execution
with model.trace(input_data):
    layer_0 = model.layers[0].output.save()
    layer_2 = model.layers[2].output.save()

print(f"Layer 0 shape: {layer_0.shape}")
print(f"Layer 2 shape: {layer_2.shape}")
```

Run it:
```bash
python test_mlxterp.py
```

## Common Use Cases

### 1. Load a Real Model

```python
from mlxterp import InterpretableModel

# Automatically loads model and tokenizer
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Use with text
with model.trace("Hello world"):
    attn = model.layers[5].attn.output.save()
```

### 2. Work with Tokenizers

```python
# Encode text to tokens
tokens = model.encode("Hello world")
print(tokens)  # [128000, 9906, 1917]

# Decode tokens to text
text = model.decode(tokens)
print(text)  # "Hello world"

# Analyze individual tokens
for i, token_id in enumerate(tokens):
    token_str = model.token_to_str(token_id)
    print(f"Token {i}: '{token_str}'")

# Get vocabulary size
print(f"Vocab size: {model.vocab_size}")
```

### 3. Collect Activations

```python
from mlxterp import get_activations

acts = get_activations(
    model,
    prompts=["Hello", "World"],
    layers=[3, 8, 12],
    positions=-1  # last token
)

print(acts["layer_3"].shape)  # (2, hidden_dim)
```

### 4. Apply Interventions

```python
from mlxterp import interventions as iv

# Scale down layer 4
with model.trace("Test", interventions={"layers.4": iv.scale(0.5)}):
    output = model.output.save()

# Add steering vector
steering = mx.random.normal((hidden_dim,))
with model.trace("Test", interventions={"layers.5": iv.add_vector(steering)}):
    output = model.output.save()
```

### 5. Logit Lens

See what each layer predicts at each token position:

```python
# Analyze predictions across layers
text = "The capital of France is"
results = model.logit_lens(text, layers=[0, 5, 10, 15])

# See how prediction at LAST position evolves through layers
for layer_idx in [0, 5, 10, 15]:
    last_pos_pred = results[layer_idx][-1][0][2]
    print(f"Layer {layer_idx}: '{last_pos_pred}'")

# Output:
# Layer  0: ' the'
# Layer  5: ' a'
# Layer 10: ' Paris'
# Layer 15: ' Paris'

# Visualize with heatmap showing ALL positions
results = model.logit_lens(
    "The Eiffel Tower is located in the city of",
    plot=True,
    max_display_tokens=15
)
# Shows: Input tokens (x-axis) × Layers (y-axis) with predicted tokens
```

### 6. Activation Patching

```python
# Get clean activation
with model.trace("The capital of France is"):
    clean_act = model.layers[8].output.save()

# Patch into different input
with model.trace("The capital of Spain is",
                interventions={"layers.8": lambda x: clean_act}):
    patched = model.output.save()
```

## What's Next?

1. **Read the full [README.md](README.md)** for detailed examples
2. **Check [API.md](API.md)** for complete API reference
3. **Explore [examples/](examples/)** for more use cases
4. **Read [CLAUDE.md](CLAUDE.md)** to understand the architecture

## Troubleshooting

### "String input provided but no tokenizer available"

**Problem:** You're passing text but no tokenizer was loaded.

**Solution:** Pass a tokenizer or use token arrays:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model-name")
model = InterpretableModel(base_model, tokenizer=tokenizer)
```

### "Model does not have 'layers' attribute"

**Problem:** Your model uses a different attribute name.

**Solution:** Specify the correct attribute:
```python
# For GPT-2
model = InterpretableModel(gpt2_model, layer_attr="h")

# For custom models
model = InterpretableModel(custom_model, layer_attr="transformer.blocks")
```

### Import Error

**Problem:** `ModuleNotFoundError: No module named 'mlxterp'`

**Solution:** Install the package:
```bash
cd mlxterp
pip install -e .
```

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/mlxterp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/mlxterp/discussions)
- **Examples**: See `examples/` directory

## Next Steps

Try these examples in order:

1. ✅ Basic tracing (you just did this!)
2. 📝 [examples/basic_usage.py](examples/basic_usage.py) - More comprehensive examples
3. 🔍 Explore your own models
4. 🧪 Experiment with interventions
5. 📊 Analyze activation patterns

Happy exploring! 🎉
