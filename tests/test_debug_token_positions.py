"""
Debug why only the first token has non-zero activations.
"""

from mlxterp import InterpretableModel
from mlx_lm import load
import mlx.core as mx

# Load model
print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Test input
text = "Hello, who are you?"
tokens = model.encode(text)
print(f"\nInput: '{text}'")
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Run trace
print("\nRunning trace...")
with model.trace(text) as trace:
    pass

# Check the MLP activation
mlp_key = "model.model.layers.0.mlp"
if mlp_key in trace.activations:
    mlp_activation = trace.activations[mlp_key]
    print(f"\n{'='*60}")
    print(f"Analyzing: {mlp_key}")
    print(f"{'='*60}")
    print(f"Shape: {mlp_activation.shape}")
    print(f"Dtype: {mlp_activation.dtype}")

    # Force evaluation
    mx.eval(mlp_activation)

    print(f"\nChecking each position:")
    # Correct indexing: [batch, sequence, hidden]
    if len(mlp_activation.shape) == 3:
        batch_size, seq_len, hidden_dim = mlp_activation.shape
        print(f"Batch size: {batch_size}, Seq len: {seq_len}, Hidden dim: {hidden_dim}")

        for i in range(seq_len):
            # Index as [batch=0, position=i, :]
            token_activation = mlp_activation[0, i, :]
            max_val = float(mx.max(token_activation))
            min_val = float(mx.min(token_activation))
            mean_val = float(mx.mean(token_activation))

            # Check if it's all zeros
            is_zero = float(mx.sum(mx.abs(token_activation))) < 1e-6

            print(f"  Position {i}: max={max_val:.4f}, min={min_val:.4f}, mean={mean_val:.4f}, all_zero={is_zero}")
    else:
        print(f"Unexpected shape! Expected 3D tensor (batch, seq, hidden)")

# Check other layers too
print(f"\n{'='*60}")
print("Checking other activations:")
print(f"{'='*60}")

keys_to_check = [
    "model.model.layers.0",
    "model.model.layers.0.self_attn",
    "model.model.layers.0.self_attn.q_proj",
    "model.model.layers.1.mlp",
]

for key in keys_to_check:
    if key in trace.activations:
        act = trace.activations[key]
        print(f"\n{key}:")
        print(f"  Shape: {act.shape}")

        if len(act.shape) == 3:
            # Check first and last token positions
            first_token = act[0, 0, :]
            last_token = act[0, -1, :]

            first_max = float(mx.max(mx.abs(first_token)))
            last_max = float(mx.max(mx.abs(last_token)))

            print(f"  First token max: {first_max:.4f}")
            print(f"  Last token max: {last_max:.4f}")

# Check the model output
if "__model_output__" in trace.activations:
    output = trace.activations["__model_output__"]
    print(f"\n{'='*60}")
    print("Model output:")
    print(f"{'='*60}")
    print(f"Shape: {output.shape}")

    if len(output.shape) == 3:
        for i in range(output.shape[1]):
            token_out = output[0, i, :]
            max_val = float(mx.max(mx.abs(token_out)))
            print(f"  Position {i}: max={max_val:.4f}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("If only position 0 has values, this suggests:")
print("1. Possible issue with causal masking in the model")
print("2. Possible issue with how activations are captured")
print("3. Or this could be expected behavior for this specific model")
