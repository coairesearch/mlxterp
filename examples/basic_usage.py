"""
Basic usage examples for mlxterp.

This example demonstrates the core functionality of the library.
"""

import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel, interventions as iv


def example_1_basic_tracing():
    """Example 1: Basic tracing and activation capture"""
    print("\n=== Example 1: Basic Tracing ===\n")

    # For this example, we'll use a simple custom model
    # In practice, you'd load a real model like:
    # model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_dim=64, num_layers=4):
            super().__init__()
            self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
                x = nn.relu(x)
            return x

    # Create and wrap model
    base_model = SimpleTransformer()
    model = InterpretableModel(base_model)

    # Create dummy input
    input_tokens = mx.random.normal((1, 10, 64))  # (batch, seq_len, hidden_dim)

    # Trace execution
    with model.trace(input_tokens) as trace:
        # Access layer outputs
        layer_0_out = model.layers[0].output.save()
        layer_2_out = model.layers[2].output.save()

    print(f"Layer 0 output shape: {layer_0_out.shape}")
    print(f"Layer 2 output shape: {layer_2_out.shape}")
    print(f"Saved values: {list(trace.saved_values.keys())}")


def example_2_interventions():
    """Example 2: Using interventions to modify activations"""
    print("\n=== Example 2: Interventions ===\n")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(32, 32) for _ in range(3)]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    base_model = SimpleModel()
    model = InterpretableModel(base_model)

    input_data = mx.random.normal((1, 32))

    # Normal forward pass
    with model.trace(input_data) as trace_normal:
        normal_output = model.output.save()

    print(f"Normal output mean: {mx.mean(normal_output):.4f}")

    # With scaling intervention
    with model.trace(input_data, interventions={"layers.1": iv.scale(0.5)}) as trace_scaled:
        scaled_output = model.output.save()

    print(f"Scaled output mean: {mx.mean(scaled_output):.4f}")

    # With zero-out intervention
    with model.trace(input_data, interventions={"layers.1": iv.zero_out}) as trace_zero:
        zeroed_output = model.output.save()

    print(f"Zeroed output mean: {mx.mean(zeroed_output):.4f}")


def example_3_activation_patching():
    """Example 3: Activation patching for causal analysis"""
    print("\n=== Example 3: Activation Patching ===\n")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(32, 32) for _ in range(3)]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    base_model = SimpleModel()
    model = InterpretableModel(base_model)

    # Clean input
    clean_input = mx.ones((1, 32))

    # Corrupted input (with noise)
    corrupted_input = mx.ones((1, 32)) + mx.random.normal((1, 32)) * 0.5

    # Get clean activation
    with model.trace(clean_input) as clean_trace:
        clean_layer_1 = model.layers[1].output.save()

    print(f"Clean activation mean: {mx.mean(clean_layer_1):.4f}")

    # Patch clean activation into corrupted run
    with model.trace(
        corrupted_input,
        interventions={"layers.1": lambda x: clean_layer_1}
    ) as patched_trace:
        patched_output = model.output.save()

    # Compare with unpatch corrupted
    with model.trace(corrupted_input) as corrupted_trace:
        corrupted_output = model.output.save()

    print(f"Corrupted output mean: {mx.mean(corrupted_output):.4f}")
    print(f"Patched output mean: {mx.mean(patched_output):.4f}")


def example_4_steering_vectors():
    """Example 4: Using steering vectors"""
    print("\n=== Example 4: Steering Vectors ===\n")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(32, 32) for _ in range(3)]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    base_model = SimpleModel()
    model = InterpretableModel(base_model)

    input_data = mx.zeros((1, 32))

    # Create a steering vector
    steering_vector = mx.random.normal((32,)) * 2.0

    # Apply steering
    with model.trace(
        input_data,
        interventions={"layers.1": iv.add_vector(steering_vector)}
    ) as steered_trace:
        steered_output = model.output.save()

    # Without steering
    with model.trace(input_data) as normal_trace:
        normal_output = model.output.save()

    print(f"Normal output norm: {mx.linalg.norm(normal_output):.4f}")
    print(f"Steered output norm: {mx.linalg.norm(steered_output):.4f}")


def example_5_composed_interventions():
    """Example 5: Composing multiple interventions"""
    print("\n=== Example 5: Composed Interventions ===\n")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(32, 32) for _ in range(3)]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    base_model = SimpleModel()
    model = InterpretableModel(base_model)

    input_data = mx.random.normal((1, 32))

    # Compose multiple interventions
    combined = iv.compose() \
        .add(iv.scale(0.8)) \
        .add(iv.noise(std=0.1)) \
        .add(iv.clamp(-2.0, 2.0)) \
        .build()

    with model.trace(input_data, interventions={"layers.1": combined}) as trace:
        output = model.output.save()

    print(f"Output after composed intervention: {output.shape}")
    print(f"Output range: [{mx.min(output):.4f}, {mx.max(output):.4f}]")


if __name__ == "__main__":
    print("mlxterp Basic Usage Examples")
    print("=" * 50)

    example_1_basic_tracing()
    example_2_interventions()
    example_3_activation_patching()
    example_4_steering_vectors()
    example_5_composed_interventions()

    print("\n" + "=" * 50)
    print("Examples completed!")
