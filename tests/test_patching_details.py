"""
Check what modules are patched vs skipped during tracing.
"""

from mlxterp import InterpretableModel
from mlx_lm import load

# Temporarily enable the debug print
import mlxterp.core.trace as trace_module

# Monkey-patch to enable logging
original_patch = trace_module.Trace._patch_model_layers

def patched_patch_model_layers(self):
    """Wrapper that logs patching statistics"""
    # Store original method
    result = original_patch(self)

    # Count what was patched
    patched_by_type = {}
    for name, (parent, attr, module) in self._patched_modules.items():
        module_type = type(module).__name__
        if module_type not in patched_by_type:
            patched_by_type[module_type] = []
        patched_by_type[module_type].append(name)

    print(f"\n{'='*60}")
    print(f"PATCHING SUMMARY")
    print(f"{'='*60}")
    print(f"Total modules patched: {len(self._patched_modules)}")
    print(f"\nBreakdown by module type:")
    for module_type, names in sorted(patched_by_type.items()):
        print(f"  {module_type}: {len(names)} modules")
        if len(names) <= 5:
            for name in names:
                print(f"    - {name}")
        else:
            print(f"    - {names[0]}")
            print(f"    - {names[1]}")
            print(f"    ... ({len(names) - 4} more)")
            print(f"    - {names[-2]}")
            print(f"    - {names[-1]}")

    return result

trace_module.Trace._patch_model_layers = patched_patch_model_layers

# Load model
print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print("\nRunning trace to analyze patching...")
with model.trace("Hello, who are you?") as trace:
    # Check a specific layer
    layer_0 = model.layers[0].output.save()

print(f"\n{'='*60}")
print(f"ACTIVATION CAPTURE")
print(f"{'='*60}")
print(f"Total activations captured: {len(trace.activations)}")
print(f"\nSample activation names:")
for i, name in enumerate(list(trace.activations.keys())[:10]):
    print(f"  {i+1}. {name}")

print("\n✅ Analysis complete!")
