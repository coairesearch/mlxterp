"""
Inspect the structure of mlx-lm models to understand where layers live.
"""

from mlx_lm import load
import mlx.nn as nn

# Load a model
model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')

print("Model type:", type(model))
print("Model attributes:", [attr for attr in dir(model) if not attr.startswith('_')])
print()

# Check what 'layers' actually is
print("model.layers type:", type(model.layers))
print("Is property?", isinstance(type(model).layers, property))
print()

# Check if model has a 'model' attribute (common pattern)
if hasattr(model, 'model'):
    print("model.model exists!")
    print("model.model type:", type(model.model))
    print("model.model attributes:", [attr for attr in dir(model.model) if not attr.startswith('_')])
    print()

    if hasattr(model.model, 'layers'):
        print("model.model.layers type:", type(model.model.layers))
        print()

        # Check if layers from property are same objects as model.model.layers
        layers_from_property = model.layers
        layers_from_model = model.model.layers

        print(f"Layers from property: {type(layers_from_property)}")
        print(f"Layers from model.model: {type(layers_from_model)}")
        print(f"Are they the same object? {layers_from_property is layers_from_model}")
        print()

        # Check individual layer objects
        if len(layers_from_property) > 0 and len(layers_from_model) > 0:
            print(f"First layer from property: {id(layers_from_property[0])}")
            print(f"First layer from model.model: {id(layers_from_model[0])}")
            print(f"Are they the same? {layers_from_property[0] is layers_from_model[0]}")
