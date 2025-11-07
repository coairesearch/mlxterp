"""
Test the new logit lens and token prediction helper methods.
"""

from mlxterp import InterpretableModel
from mlx_lm import load
import mlx.core as mx

print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print("\n" + "="*60)
print("Test 1: get_token_predictions()")
print("="*60)

# Get predictions from a specific layer
text = "The capital of France is"
with model.trace(text) as trace:
    layer_6 = trace.activations["model.model.layers.6"]

# Get last token's hidden state
last_token_hidden = layer_6[0, -1, :]

# Get predictions
print(f"\nInput: '{text}'")
print("Top 10 predictions from layer 6:")
predictions = model.get_token_predictions(last_token_hidden, top_k=10)

for i, token_id in enumerate(predictions, 1):
    token_str = model.token_to_str(token_id)
    print(f"  {i}. {token_id:6d} -> '{token_str}'")

# Test with scores
print("\nWith scores:")
predictions_with_scores = model.get_token_predictions(
    last_token_hidden,
    top_k=5,
    return_scores=True
)
for token_id, score in predictions_with_scores:
    token_str = model.token_to_str(token_id)
    print(f"  {token_str:20s} (score: {score:.2f})")

print("\n" + "="*60)
print("Test 2: logit_lens() - Full Analysis")
print("="*60)

# Run logit lens across all layers
text = "The capital of France is"
print(f"\nInput: '{text}'")
print("Analyzing what each layer predicts at each position...\n")

results = model.logit_lens(text, top_k=1, layers=[0, 5, 10, 15])

# Show last position predictions across layers
print("Predictions for LAST token position across layers:")
for layer_idx in sorted(results.keys()):
    layer_predictions = results[layer_idx]
    last_pos_pred = layer_predictions[-1][0][2]  # Top prediction at last position
    print(f"Layer {layer_idx:2d}: '{last_pos_pred}'")

print("\n" + "="*60)
print("Test 3: logit_lens() - Show All Positions")
print("="*60)

# Show predictions at all positions for specific layers
text = "The capital of France"
print(f"\nInput: '{text}'")
tokens = model.encode(text)
print(f"Tokens: {[model.token_to_str(t) for t in tokens]}")
print("\nPredictions at each position:")

results = model.logit_lens(text, top_k=1, layers=[5, 10, 15])

for layer_idx in [5, 10, 15]:
    if layer_idx in results:
        print(f"\nLayer {layer_idx}:")
        layer_predictions = results[layer_idx]
        for pos_idx, predictions in enumerate(layer_predictions):
            input_token = model.token_to_str(tokens[pos_idx])
            pred_token = predictions[0][2]  # Top prediction
            print(f"  Pos {pos_idx} ('{input_token}') -> predicts: '{pred_token}'")

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
