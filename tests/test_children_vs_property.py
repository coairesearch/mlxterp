"""
Check if .children() returns same list as property
"""

from mlx_lm import load

# Load model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')

# Get layers via property
layers_via_property = base_model.model.layers
print(f"Via property:  ID={id(layers_via_property)}, type={type(layers_via_property)}")

# Get layers via .children()
children = base_model.model.children()
print(f"\n.children() returns: {type(children)}")
print(f"Keys: {list(children.keys())}")

if 'layers' in children:
    layers_via_children = children['layers']
    print(f"\nVia children['layers']: ID={id(layers_via_children)}, type={type(layers_via_children)}")
    print(f"\nSame list? {layers_via_property is layers_via_children}")

    # Check individual items
    if len(layers_via_property) > 0 and len(layers_via_children) > 0:
        print(f"\nFirst item via property:  ID={id(layers_via_property[0])}")
        print(f"First item via children:  ID={id(layers_via_children[0])}")
        print(f"Same item? {layers_via_property[0] is layers_via_children[0]}")
