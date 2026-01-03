#!/usr/bin/env python3
"""
Tutorial 1: The Logit Lens

This script demonstrates the logit lens technique based on:
"interpreting GPT: the logit lens" by nostalgebraist (2020)
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

The logit lens reveals how transformer predictions evolve across layers by
projecting intermediate hidden states through the output embedding matrix.

NOTE: The original paper analyzed GPT-2. We use Llama-3.2-1B-Instruct to
demonstrate the technique on a modern model. The core findings generalize
across transformer architectures, though specific layer numbers may differ.

NOTE: Scores returned by get_token_predictions() are LOGITS (unnormalized),
not probabilities. Higher logits indicate stronger predictions.

Run this script:
    python examples/tutorials/01_logit_lens/logit_lens_tutorial.py
"""

from mlxterp import InterpretableModel
from mlxterp.core.module_resolver import find_layer_key_pattern
import mlx.core as mx


def manual_logit_lens(model, hidden_state):
    """
    Manually apply the logit lens to a hidden state.

    This implements the core logit lens operation:
    1. Apply final layer normalization
    2. Project through the unembedding matrix
    3. Get token predictions

    Args:
        model: InterpretableModel instance
        hidden_state: Hidden state tensor of shape (hidden_dim,)

    Returns:
        List of (token_id, logit_score, token_str) tuples
        Note: logit_score is an unnormalized score, not a probability
    """
    # Apply final layer normalization
    final_norm = model._module_resolver.get_final_norm()
    normalized = final_norm(hidden_state)

    # Get predictions through the output projection
    predictions = model.get_token_predictions(normalized, top_k=5, return_scores=True)

    # Convert to readable format
    results = []
    for token_id, score in predictions:
        token_str = model.token_to_str(token_id)
        results.append((token_id, score, token_str))

    return results


def experiment_1_basic_logit_lens(model):
    """
    Experiment 1: Basic Logit Lens

    Demonstrates how predictions evolve across layers for a simple factual prompt.
    This reproduces the core finding that models iteratively refine predictions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Basic Logit Lens")
    print("=" * 70)

    prompt = "The capital of France is"
    print(f"\nPrompt: '{prompt}'")

    # Trace the model
    with model.trace(prompt) as trace:
        pass

    tokens = model.encode(prompt)
    print(f"Tokens: {[model.token_to_str(t) for t in tokens]}")

    print("\n" + "-" * 50)
    print("Layer-by-layer predictions (at last token position):")
    print("-" * 50)

    # Manual implementation
    print("\n[Manual Implementation]")
    for layer_idx in range(0, len(model.layers), 4):
        layer_key = find_layer_key_pattern(trace.activations, layer_idx)
        if layer_key is None:
            continue

        layer_output = trace.activations[layer_key]
        last_token_hidden = layer_output[0, -1, :]

        predictions = manual_logit_lens(model, last_token_hidden)
        top_token = predictions[0][2]
        top_score = predictions[0][1]

        marker = " <-- CORRECT" if "Paris" in top_token else ""
        print(f"Layer {layer_idx:2d}: '{top_token:12s}' (score: {top_score:.3f}){marker}")

    # Built-in method
    print("\n[Built-in logit_lens() method]")
    results = model.logit_lens(prompt, layers=list(range(0, len(model.layers), 4)))

    for layer_idx in sorted(results.keys()):
        top_pred = results[layer_idx][-1][0]  # Last position, top prediction
        marker = " <-- CORRECT" if "Paris" in top_pred[2] else ""
        print(f"Layer {layer_idx:2d}: '{top_pred[2]:12s}' (score: {top_pred[1]:.3f}){marker}")


def experiment_2_crystallization(model):
    """
    Experiment 2: Finding the Crystallization Point

    The paper observed that correct predictions often "crystallize" suddenly
    at a specific layer. This experiment identifies when this happens.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Prediction Crystallization")
    print("=" * 70)

    test_cases = [
        ("The Eiffel Tower is located in", "Paris"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("Water is made of hydrogen and", "oxygen"),
        ("Albert Einstein developed the theory of", "relat"),  # relativity
    ]

    for prompt, expected in test_cases:
        print(f"\nPrompt: '{prompt}'")
        print(f"Expected: contains '{expected}'")

        results = model.logit_lens(prompt)
        crystallization_layer = None

        for layer_idx in sorted(results.keys()):
            top_pred = results[layer_idx][-1][0][2]
            if expected.lower() in top_pred.lower():
                crystallization_layer = layer_idx
                break

        if crystallization_layer is not None:
            print(f"Crystallizes at layer: {crystallization_layer}")
        else:
            # Show what it predicted instead
            final_pred = results[max(results.keys())][-1][0][2]
            print(f"Did not crystallize. Final prediction: '{final_pred}'")


def experiment_3_early_vs_late(model):
    """
    Experiment 3: Early vs Late Layer Comparison

    The paper found that early layers often predict semantically related
    but incorrect tokens, while late layers converge on the correct answer.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Early vs Late Layer Predictions")
    print("=" * 70)

    prompt = "Barack Obama was the 44th president of the United"
    print(f"\nPrompt: '{prompt}'")

    results = model.logit_lens(prompt, top_k=5)

    early_layer = min(results.keys())
    late_layer = max(results.keys())

    print(f"\nEarly Layer ({early_layer}) - Top 5 predictions:")
    for i, (token_id, score, token_str) in enumerate(results[early_layer][-1][:5]):
        print(f"  {i+1}. '{token_str}': {score:.4f}")

    print(f"\nLate Layer ({late_layer}) - Top 5 predictions:")
    for i, (token_id, score, token_str) in enumerate(results[late_layer][-1][:5]):
        print(f"  {i+1}. '{token_str}': {score:.4f}")

    print("\n[Observation]")
    print("Early layers often show related concepts (countries, places)")
    print("Late layers converge on 'States' to complete 'United States'")


def experiment_4_multiple_prompts(model):
    """
    Experiment 4: Different Prompt Types

    Test the logit lens on various types of prompts to see
    how the crystallization pattern differs.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Different Prompt Types")
    print("=" * 70)

    prompts = [
        ("Factual", "The largest ocean on Earth is the"),
        ("Linguistic", "She walked to the store and"),
        ("Arithmetic", "Two plus two equals"),
        ("Code", "def hello():\n    print("),
    ]

    for category, prompt in prompts:
        print(f"\n[{category}]")
        print(f"Prompt: '{prompt}'")

        results = model.logit_lens(prompt, layers=[0, len(model.layers)//2, len(model.layers)-1])

        for layer_idx in sorted(results.keys()):
            top_pred = results[layer_idx][-1][0]
            print(f"  Layer {layer_idx:2d}: '{top_pred[2]}' (score: {top_pred[1]:.3f})")


def experiment_5_visualization(model):
    """
    Experiment 5: Visualization

    Generate a heatmap visualization showing prediction evolution.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Visualization")
    print("=" * 70)

    prompt = "The quick brown fox jumps over the lazy"
    print(f"\nPrompt: '{prompt}'")
    print("\nGenerating visualization...")

    try:
        # This will display a matplotlib figure if available
        results = model.logit_lens(
            prompt,
            plot=True,
            max_display_tokens=10,
            figsize=(14, 8)
        )
        print("Visualization displayed (or saved if running non-interactively)")
    except Exception as e:
        print(f"Visualization failed (matplotlib may not be available): {e}")
        # Fallback to text representation
        results = model.logit_lens(prompt)
        tokens = model.encode(prompt)

        print("\nText representation:")
        print("-" * 60)

        header = "Layer | " + " | ".join(f"Pos {i}" for i in range(min(5, len(tokens))))
        print(header)
        print("-" * len(header))

        for layer_idx in list(sorted(results.keys()))[::4]:
            row = f"{layer_idx:5d} | "
            for pos in range(min(5, len(results[layer_idx]))):
                top_pred = results[layer_idx][pos][0][2][:6]
                row += f"{top_pred:6s} | "
            print(row)


def main():
    """Run all experiments."""
    print("=" * 70)
    print("LOGIT LENS TUTORIAL")
    print("Reproducing: 'interpreting GPT: the logit lens' (nostalgebraist, 2020)")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
    print(f"Model loaded: {len(model.layers)} layers")

    # Run experiments
    experiment_1_basic_logit_lens(model)
    experiment_2_crystallization(model)
    experiment_3_early_vs_late(model)
    experiment_4_multiple_prompts(model)
    experiment_5_visualization(model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings reproduced:

1. The logit lens reveals that transformers iteratively refine predictions
   across layers, not just at the final layer.

2. Correct predictions often "crystallize" suddenly at a specific layer,
   rather than gradually improving.

3. Early layers tend to predict semantically related tokens, while late
   layers converge on the contextually correct answer.

4. This pattern holds across different types of prompts (factual,
   linguistic, arithmetic, code), though the crystallization layer varies.

These findings support the residual stream hypothesis: transformers maintain
a running representation that each layer reads from and writes to, gradually
building up the correct prediction.
    """)


if __name__ == "__main__":
    main()
