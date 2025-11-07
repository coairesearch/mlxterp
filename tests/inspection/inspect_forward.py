"""
Inspect how mlx-lm models do their forward pass.
"""

from mlx_lm import load
import inspect

# Load a model
model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')

print("=== Model's __call__ method ===")
print(inspect.getsource(model.__call__))
print()

print("=== Model.model's __call__ method (if exists) ===")
if hasattr(model.model, '__call__'):
    try:
        print(inspect.getsource(model.model.__call__))
    except:
        print("Can't get source (might be inherited)")
        print(f"__call__ is: {model.model.__call__}")
