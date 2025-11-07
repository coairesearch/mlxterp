# mlxterp Tests

This directory contains tests and inspection scripts for mlxterp.

## Directory Structure

```
tests/
├── README.md                      # This file
├── test_real_model.py            # Main integration test with real models
├── test_comprehensive.py         # Comprehensive activation capture test
├── test_activation_validity.py   # Validates captured activations
├── test_interventions.py         # Tests intervention functionality
├── test_tokenizer_methods.py     # Tests tokenizer convenience methods
├── test_logit_lens.py            # Tests logit lens and token prediction helpers
├── test_nested_modules.py        # Tests nested module access
├── test_patching.py              # Tests low-level patching mechanisms
└── inspection/                   # Inspection utilities
    ├── inspect_model.py          # Inspect mlx-lm model structure
    ├── inspect_forward.py        # Inspect forward pass implementation
    └── inspect_mlx_modules.py    # Inspect MLX module hierarchy
```

## Required Tests

### Core Functionality Tests

These tests verify that mlxterp works correctly with real models:

#### 1. `test_comprehensive.py` ⭐ **ESSENTIAL**

Tests complete activation capture with real models.

```bash
uv run python tests/test_comprehensive.py
```

**What it verifies:**
- Loads real mlx-lm model (Llama-3.2-1B-Instruct-4bit)
- Captures all activations (should capture ~196 activations)
- Tests nested module access (Q/K/V projections, MLP, attention)
- Tests interventions work correctly

**Expected output:**
```
✅ Captured 196 activations!
✅ Layer 5 attention: (1, 7, 2048)
✅ Layer 3 Q projection: (1, 7, 2048)
✅ Model output: (1, 7, 128256)
✅ Intervention worked!
```

#### 2. `test_real_model.py`

Tests basic functionality with both simple and real models.

```bash
uv run python tests/test_real_model.py
```

**What it verifies:**
- Simple custom models work (4 layers)
- Real mlx-lm models work (16 layers)
- Layer activations are captured correctly

#### 3. `test_activation_validity.py`

Validates that captured activations are numerically valid.

```bash
uv run python tests/test_activation_validity.py
```

**What it verifies:**
- No NaN values in activations
- No Inf values in activations
- Reasonable value ranges
- Correct dtypes

#### 4. `test_interventions.py`

Tests that interventions actually affect model behavior.

```bash
uv run python tests/test_interventions.py
```

**What it verifies:**
- Baseline forward pass works
- Scaling intervention changes outputs
- Intervention effects propagate through model

#### 5. `test_tokenizer_methods.py`

Tests tokenizer convenience methods on InterpretableModel.

```bash
uv run python tests/test_tokenizer_methods.py
```

**What it verifies:**
- `encode()` converts text to tokens
- `decode()` converts tokens to text (works with lists and mx.array)
- `encode_batch()` handles multiple texts
- `token_to_str()` decodes individual tokens
- `vocab_size` property returns correct vocabulary size
- Direct `tokenizer` attribute is accessible
- Token-position specific activation extraction works

**Expected output:**
```
✅ ALL TOKENIZER METHODS WORK!
✅ Successfully extracted token-specific activation!
```

#### 6. `test_logit_lens.py`

Tests logit lens and token prediction helper methods.

```bash
uv run python tests/test_logit_lens.py
```

**What it verifies:**
- `get_token_predictions()` decodes hidden states to tokens
- `get_token_predictions()` returns scores correctly
- `logit_lens()` analyzes predictions at all positions
- `logit_lens()` works with specific layer subsets
- Quantized embeddings are handled correctly
- Predictions show progressive refinement through layers

**Expected output:**
```
✅ All tests passed!
```

Shows logit lens output analyzing predictions at each position:
```
Layer  0: ' the'
Layer  5: ' a'
Layer 10: ' Paris'
Layer 15: ' Paris'
```

#### 7. `test_logit_lens_plot.py`

Tests logit lens visualization functionality.

```bash
uv run python tests/test_logit_lens_plot.py
```

**What it verifies:**
- `logit_lens()` with `plot=True` generates heatmap visualization
- Customizable figsize, colormap, and max_display_tokens parameters
- Visualization shows predictions at each (layer, position) coordinate
- Input tokens displayed on x-axis, layers on y-axis
- Predicted tokens shown in each cell with color coding

**Expected output:**
- Displays matplotlib heatmap with structure:
  - **X-axis**: Input token positions
  - **Y-axis**: Model layers
  - **Cells**: Top predicted token at each (layer, position)
  - **Colors**: Different colors for different predictions
- Returns results dict as usual

**Note:** Requires matplotlib: `uv add matplotlib` or `pip install matplotlib`

### Development Tests

These tests help understand and debug the internals:

#### 8. `test_patching.py`

Tests low-level patching mechanism.

```bash
uv run python tests/test_patching.py
```

**What it verifies:**
- Instance `__call__` patching doesn't work (expected)
- Understanding of Python's method resolution

#### 9. `test_nested_modules.py`

Tests nested module structure and access.

```bash
uv run python tests/test_nested_modules.py
```

## Inspection Scripts

Located in `inspection/`, these are utilities for understanding model structure:

### `inspect_model.py`

Inspects the structure of mlx-lm models:

```bash
uv run python tests/inspection/inspect_model.py
```

Shows:
- Model type and attributes
- Whether `layers` is a property
- Module hierarchy (model.model.layers)

### `inspect_forward.py`

Shows the source code of model forward passes:

```bash
uv run python tests/inspection/inspect_forward.py
```

Useful for understanding how models process data.

### `inspect_mlx_modules.py`

Inspects MLX module structure and child discovery:

```bash
uv run python tests/inspection/inspect_mlx_modules.py
```

Shows:
- How to discover child modules
- Available submodules (self_attn, mlp, etc.)
- Module attributes

## Running All Tests

To run all essential tests:

```bash
uv run python tests/test_comprehensive.py && \
uv run python tests/test_real_model.py && \
uv run python tests/test_activation_validity.py && \
uv run python tests/test_interventions.py
```

## Requirements

All tests require:
- `mlx-lm` installed: `uv add mlx-lm`
- Working internet connection (for model downloads on first run)
- Models are cached after first download

## Expected Behavior

### ✅ Success Indicators

- **test_comprehensive.py**: Should capture ~196 activations
- **test_real_model.py**: Both simple and real models should pass
- **test_activation_validity.py**: No NaN/Inf values
- **test_interventions.py**: Interventions should change outputs

### ❌ Common Issues

**Issue**: `ModuleNotFoundError: No module named 'mlx_lm'`
**Fix**: `uv add mlx-lm`

**Issue**: `ValueError: Could not load model`
**Fix**: Check internet connection, model will download on first run

**Issue**: Model download is slow
**Note**: Models are 1-2GB, cached after first download at `~/.cache/huggingface/`

## Test Output Filtering

Tests output model download progress bars. To filter these:

```bash
uv run python tests/test_comprehensive.py 2>&1 | grep -v "Fetching\|files:"
```

## Adding New Tests

When adding new tests:

1. Place in `tests/` directory
2. Name with `test_` prefix for consistency
3. Add documentation to this README
4. Include in "Required Tests" if essential for CI/CD

## Future: pytest Integration

Currently tests are standalone scripts. Future plans:

- Convert to pytest format
- Add fixtures for model loading
- Add CI/CD integration
- Add coverage reports
- Add performance benchmarks
