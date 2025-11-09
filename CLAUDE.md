# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mlxterp** is a mechanistic interpretability library for Apple Silicon, leveraging the MLX framework. The goal is to provide researchers with tools similar to nnsight and nnterp but optimized for Apple Silicon Macs.

**Core Philosophy:**
- Simplicity first - make mechanistic interpretability accessible to many researchers
- Lean codebase with minimal boilerplate
- Clean architecture and good maintainability
- Functional-first API leveraging MLX's strengths

## Architecture Design

The library follows a **functional-first paradigm** that aligns with MLX's design:

### Core Components

1. **core/** - Computation graph tracing, intervention system, activation caching, hook registration
   - Generic model wrapping that works with ANY MLX model (not model-specific implementations)
   - Automatic layer detection and naming
2. **analysis/** - Attention patterns, activation analysis, circuit discovery, attribution methods
3. **interpretability/** - Logit lens, patching utilities, probing tools, sparse autoencoder support
4. **visualization/** - Attention/activation visualization and interactive dashboard
5. **utils/** - Data utilities, metrics, Metal-specific optimizations

### Key Design Decisions

**1. Context Manager Pattern for Tracing (nnterp style)**
- Use context managers for clean, intuitive tracing: `with model.trace("text"):`
- Direct attribute access to layer outputs: `model.layers[3].attn.output`
- `.save()` method to capture values for later use

**2. Direct Attribute Access with Proxy Objects**
- Access layers and modules naturally: `model.layers[i].attn.output`
- No need for verbose method calls like `.get_activation("layer_3.attn")`
- Proxy objects handle the wrapping transparently

**3. Lazy Evaluation with Explicit Materialization**
- Leverage MLX's lazy evaluation for efficiency
- Use `.save()` to capture values when needed

**5. Unified Memory Exploitation**
- Take advantage of MLX's unified memory model for zero-copy operations
- Design for efficient caching without device transfers

**6. Generic Model Wrapping (nnsight principle)**
- Works with ANY MLX model, not specific model implementations
- Automatically discovers and wraps model layers at runtime
- No need for model-specific subclasses (unlike TransformerLens's HookedTransformer approach)

## Development Guidelines

### Repository Cleanliness (CRITICAL)

**ALWAYS keep the repository clean and professional:**

**DO:**
- Place all examples in `examples/` directory
- Place all tests in `tests/` directory
- Place all documentation in `docs/` directory
- Use temporary variables in code instead of creating debug files
- Clean up any temporary files immediately after debugging
- Add generated files (`.mlx/`, `wandb/`, etc.) to `.gitignore`

**DON'T:**
- Create debug scripts in root directory (e.g., `debug_*.py`, `test_*.py`, `check_*.py`)
- Create temporary documentation files in root (e.g., `*_SUMMARY.md`, `*_FIX.md`, `*_GUIDE.md`)
- Leave test artifacts (`.mlx` model files, cache directories)
- Scatter analysis scripts across the repository
- Create multiple redundant documentation files

**Root directory should ONLY contain:**
- `CLAUDE.md` - This file (development guidelines)
- `README.md` - Main documentation
- `SAE_ROADMAP.md` - SAE development phases
- `TROUBLESHOOTING.md` - User troubleshooting guide
- Configuration files (`pyproject.toml`, `mkdocs.yml`, etc.)
- Essential project files (`.gitignore`, `LICENSE`, etc.)

**For debugging:**
- Use Python debugger or print statements
- Create temporary functions, not files
- If you must create a debug script, delete it immediately after use
- Use notebooks for interactive exploration (but clean them up)

**For documentation:**
- Add to `docs/` directory and update `mkdocs.yml`
- Don't create summary files for internal communication
- Consolidate related information instead of creating many small files

### When Creating New Components

**DO:**
- Use functional APIs that return new values
- Leverage MLX's lazy evaluation by default
- Add type hints to all function signatures
- Design for Metal optimization opportunities
- Keep the API simple and composable
- Keep the repository clean and organized

**DON'T:**
- Create model-specific implementations (e.g., HookedLlama, HookedMistral) - use generic wrapping
- Add stateful hooks without strong justification
- Create unnecessary abstractions or boilerplate
- Duplicate functionality that exists in MLX
- Add features that compromise the "simple and accessible" goal
- Scatter debug files and temporary documentation across the repo

### Code Style Preferences

- **Direct attribute access** pattern inspired by nnterp (not verbose method calls)
- Context managers for tracing (`with model.trace():`)
- Proxy objects for intuitive layer access (`model.layers[i].attn.output`)
- Simple utility functions over complex class hierarchies
- Use dataclasses for configuration when needed

### MLX-Specific Considerations

- **Unified Memory**: Arrays live in shared memory - no CPU/GPU transfers needed
- **Lazy Evaluation**: Computations aren't executed until `mx.eval()` is called
- **Dynamic Graphs**: Computation graphs are built dynamically (easier debugging)
- **Metal Shaders**: Opportunities for custom Metal kernels in performance-critical paths

### Naming Conventions

**Main API:**
- Main class: `InterpretableModel` (not NNMLX)
- Package name: `mlxterp` (aligns with nnterp)

**Internal components:**
- Proxy objects: `LayerProxy`, `ModuleProxy` for attribute access
- Interventions: Simple functions, not complex class hierarchies
- Utilities: `get_activations()`, `run_prompts()`, etc.

## Development Setup

### Using uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/yourusername/mlxterp
cd mlxterp

# Install with all dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Serve documentation
uv run mkdocs serve
```

### Using pip

```bash
pip install -e ".[dev,docs,viz]"
pytest
mkdocs serve
```

## Documentation

Documentation is built with **mkdocs** and **mkdocs-material**:

- **Source**: `docs/` directory
- **Config**: `mkdocs.yml`
- **Build**: `mkdocs build`
- **Serve**: `mkdocs serve` (http://localhost:8000)

### Documentation Structure

```
docs/
├── index.md          - Home/landing page
├── QUICKSTART.md     - 5-minute guide
├── installation.md   - Installation instructions
├── examples.md       - Usage examples
├── API.md           - Complete API reference
├── architecture.md  - Design and implementation
├── contributing.md  - Contribution guidelines
├── license.md       - License information
└── citation.md      - How to cite
```

## Implementation Status

**✅ Phase 1: Core Infrastructure** (Complete)
- ✅ Tracing system with context managers
- ✅ Comprehensive intervention framework (7 intervention types)
- ✅ Activation caching and storage
- ✅ Generic model wrapping using recursive module discovery
- ✅ Composition-based module wrapping (handles any MLX model)
- ✅ Cycle detection for complex module graphs
- ✅ Fine-grained activation capture (~196 per forward pass)

**✅ Phase 2: Analysis Tools** (Complete)
- ✅ Activation patching
- ✅ Steering vectors (add_vector intervention)
- ✅ Direct activation access via trace.activations dict
- ✅ Fine-grained component access (Q/K/V, MLP gates, layer norms)
- ✅ Utility functions (get_activations, batch_get_activations)

**✅ Documentation** (Complete)
- ✅ mkdocs-based documentation
- ✅ Complete API reference
- ✅ Usage examples and guides (README, JUPYTER_GUIDE)
- ✅ Test documentation (tests/README.md)
- ✅ Clean repository structure

**✅ Testing** (Complete)
- ✅ Comprehensive integration tests
- ✅ Real model tests (Llama-3.2-1B)
- ✅ Activation validity tests
- ✅ Intervention tests
- ✅ Nested module access tests

**🔄 Phase 3: Advanced Features** (Future)
- SAE integration
- Attention pattern analysis tools
- Advanced circuit discovery
- Logit lens utilities
- Metal kernel optimizations
- Visualization dashboard

**🔄 Phase 4: Ecosystem** (Future)
- PyPI package publication
- CI/CD pipeline
- Benchmarks and performance analysis
- Model zoo with pre-computed features
- Community examples and tutorials
- Integration with other interpretability tools

## Current Capabilities

**What Works Now:**
1. **Real Model Support**: Full support for mlx-lm models (Llama, Mistral, etc.)
2. **Comprehensive Capture**: ~196 activations captured per forward pass
3. **Fine-Grained Access**: Q/K/V projections, MLP components, layer norms, attention, etc.
4. **All Intervention Types**: zero_out, scale, add_vector, replace_with, clamp, noise, compose
5. **Activation Patching**: Full support for replacing activations between runs
6. **Steering Vectors**: Add directional biases to layer outputs
7. **Custom Models**: Works with any MLX nn.Module
8. **Production Ready**: Stable API, comprehensive tests, clean documentation

## Example API Usage (Target Design)

**Simple and clean API inspired by nnterp:**

```python
from mlxterp import InterpretableModel
import mlx.core as mx

# Load any MLX model with automatic wrapping
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Simple tracing with context manager
with model.trace("The capital of France is"):
    # Direct attribute access to layer outputs
    attn_3 = model.layers[3].attn.output.save()
    mlp_8 = model.layers[8].mlp.output.save()
    logits = model.output.save()

# Get activations with utility functions
from mlxterp.utils import get_activations

activations = get_activations(
    model,
    prompts=["Hello world", "The capital"],
    layers=[3, 8, 12],
    positions=-1  # last token
)

# Built-in intervention methods
with model.trace("Text to analyze"):
    # Zero out a layer
    model.layers[4].attn.output = mx.zeros_like(model.layers[4].attn.output)
    result = model.output.save()

# Activation patching
with model.trace("Clean input") as clean:
    clean_mlp = model.layers[8].mlp.output.save()

with model.trace("Corrupted input"):
    model.layers[8].mlp.output = clean_mlp  # Patch in clean activation
    patched_output = model.output.save()

# Steering vectors
model.steer(
    layers=[5, 6, 7],
    vector=steering_vector,
    strength=2.0
)
```

**Key principles:**
- Context manager pattern for tracing
- Direct attribute access (`model.layers[i].attn.output`)
- Utility functions for common tasks
- Built-in methods for interventions

## Related Projects

- **nnsight** (https://nnsight.net): PyTorch-based interpretability framework
  - **Design inspiration**: We follow nnsight's principle of generic model wrapping rather than model-specific implementations
- **nnterp** (https://github.com/Butanium/nnterp/): Lightweight interpretability tools
- **TransformerLens**: Comprehensive interpretability library for transformers
  - **Key difference**: TransformerLens uses model-specific implementations (HookedTransformer, HookedLlama, etc.). We use generic wrapping that works with any MLX model.

This library adapts their core interpretability concepts for Apple Silicon with MLX optimization, prioritizing simplicity and model-agnostic design.
