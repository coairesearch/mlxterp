# Contributing to mlxterp

Thank you for your interest in contributing to mlxterp! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Make your changes
5. Submit a pull request

## Development Setup

### Using uv (Recommended)

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mlxterp
cd mlxterp

# Install with all dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Start development server for docs
uv run mkdocs serve
```

### Using pip

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mlxterp
cd mlxterp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev,viz]"

# Run tests
pytest
```

## Code Style

### Python Code

We use **Black** for formatting and **flake8** for linting:

```bash
# Format code
black mlxterp/ examples/

# Check linting
flake8 mlxterp/

# Type checking
mypy mlxterp/
```

### Guidelines

- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints for all functions
- Write docstrings (Google style)

Example:

```python
def my_function(x: mx.array, scale: float = 1.0) -> mx.array:
    """
    Brief description of function.

    Args:
        x: Input array
        scale: Scaling factor

    Returns:
        Scaled array

    Example:
        >>> result = my_function(mx.array([1, 2, 3]), scale=2.0)
    """
    return x * scale
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_proxy.py

# Run with coverage
pytest --cov=mlxterp --cov-report=html
```

### Writing Tests

Place tests in `tests/` directory:

```python
# tests/test_feature.py
import mlxterp
import mlx.core as mx

def test_my_feature():
    """Test description"""
    model = mlxterp.InterpretableModel(...)
    # Your test
    assert result == expected
```

## Documentation

### Building Docs

```bash
# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

### Writing Documentation

- Use Markdown format
- Include code examples
- Add type hints to examples
- Update API.md for new functions

### Documentation Structure

```
docs/
├── index.md          - Home page
├── QUICKSTART.md     - Getting started
├── installation.md   - Install instructions
├── examples.md       - Usage examples
├── API.md           - API reference
├── architecture.md  - Design docs
└── contributing.md  - This file
```

## Pull Request Process

### Before Submitting

1. **Run tests**: Ensure all tests pass
2. **Format code**: Run `black` on changed files
3. **Update docs**: Add documentation for new features
4. **Write tests**: Include tests for new functionality
5. **Update CHANGELOG**: Add entry for your changes

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted (black)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages are clear

### Commit Messages

Use clear, descriptive commit messages:

```
Add intervention composition feature

- Implement InterventionComposer class
- Add tests for composition
- Update documentation with examples
```

### PR Title Format

```
[Type] Brief description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- test: Add/update tests
- chore: Maintenance
```

## Types of Contributions

### Bug Reports

Open an issue with:

- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment (Python version, macOS version, chip type)
- Minimal code example

### Feature Requests

Open an issue with:

- Use case description
- Proposed API (if applicable)
- Example usage
- Rationale

### Code Contributions

Areas for contribution:

1. **Core Features**
   - New intervention types
   - Performance optimizations
   - Better error messages

2. **Utilities**
   - Activation analysis functions
   - Visualization tools
   - Data processing helpers

3. **Documentation**
   - More examples
   - Tutorials
   - API clarifications

4. **Testing**
   - Increase coverage
   - Add edge case tests
   - Performance benchmarks

## Design Principles

When contributing, keep these principles in mind:

1. **Simplicity**: Prefer simple solutions over complex ones
2. **Composability**: Make components work together naturally
3. **MLX Native**: Leverage MLX's features (lazy eval, unified memory)
4. **Clean API**: Maintain intuitive, Pythonic interfaces
5. **Minimal Abstractions**: Avoid unnecessary layers

### Example: Adding an Intervention

Good:

```python
def my_intervention(x: mx.array) -> mx.array:
    """Simple, functional intervention"""
    return mx.clip(x, -1, 1)
```

Avoid:

```python
class MyInterventionClass:
    """Unnecessary class wrapper"""
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, x):
        return mx.clip(x, self.min_val, self.max_val)
```

## Code Review Process

1. Maintainer reviews your PR
2. Feedback provided (if needed)
3. Make requested changes
4. PR approved and merged

### Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Design consistency
- Performance impact

## Community Guidelines

- Be respectful and welcoming
- Help others learn
- Give constructive feedback
- Credit others' work

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/coairesearch/mlxterp/issues)


## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Don't hesitate to ask questions in Issues or Discussions. We're here to help!

Thank you for contributing to mlxterp! 🎉
