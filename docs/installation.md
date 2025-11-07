# Installation

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9 or higher
- MLX framework

## Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust.

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Install mlxterp

```bash
# Clone the repository
git clone https://github.com/yourusername/mlxterp
cd mlxterp

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate

# Verify installation
python -c "import mlxterp; print(mlxterp.__version__)"
```

## Using pip

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlxterp
cd mlxterp

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### With Optional Dependencies

```bash
# Install with visualization tools
pip install -e ".[viz]"

# Install with development tools
pip install -e ".[dev]"

# Install everything
pip install -e ".[dev,viz]"
```

## From PyPI (Future)

Once published to PyPI:

```bash
pip install mlxterp

# With extras
pip install mlxterp[viz]
```

## Verify Installation

```python
import mlxterp
import mlx.core as mx
import mlx.nn as nn

# Check version
print(f"mlxterp version: {mlxterp.__version__}")

# Test basic functionality
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(32, 32) for _ in range(3)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = mlxterp.InterpretableModel(SimpleModel())
print(f"Model has {len(model.layers)} layers")
print("✓ Installation successful!")
```

## Dependencies

### Core Dependencies

- **mlx** (>=0.0.1): Apple's ML framework
- **Python** (>=3.9): Minimum Python version

### Optional Dependencies

#### Visualization (`viz`)

- **matplotlib** (>=3.5): For plotting
- **plotly** (>=5.0): Interactive visualizations

#### Development (`dev`)

- **pytest** (>=7.0): Testing framework
- **black** (>=22.0): Code formatting
- **flake8** (>=4.0): Linting
- **mypy** (>=0.950): Type checking
- **mkdocs-material**: Documentation
- **mkdocstrings**: API documentation generation

## Development Setup

For contributing to mlxterp:

### Using uv

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mlxterp
cd mlxterp

# Install with all dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Build docs
uv run mkdocs serve
```

### Using pip

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mlxterp
cd mlxterp

# Install in editable mode with dev dependencies
pip install -e ".[dev,viz]"

# Run tests
pytest

# Build docs
mkdocs serve
```

## Troubleshooting

### MLX Not Found

**Problem**: `ModuleNotFoundError: No module named 'mlx'`

**Solution**: Install MLX:

```bash
pip install mlx
```

### Apple Silicon Required

**Problem**: MLX requires Apple Silicon

**Solution**: mlxterp only works on Macs with M1/M2/M3/M4 chips. For Intel Macs or other platforms, consider using PyTorch-based alternatives like nnsight or TransformerLens.

### Python Version

**Problem**: `Python 3.9+ required`

**Solution**: Upgrade Python:

```bash
# Using Homebrew
brew install python@3.11

# Or download from python.org
```

### Permission Errors

**Problem**: Permission denied during installation

**Solution**: Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Uninstalling

### With uv

```bash
cd mlxterp
uv pip uninstall mlxterp
```

### With pip

```bash
pip uninstall mlxterp
```

## Next Steps

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [Examples](examples.md) - Learn by example
- [API Reference](API.md) - Complete API documentation
