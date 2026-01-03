#!/usr/bin/env python3
"""
Tutorial 2: The Tuned Lens

This script demonstrates the tuned lens technique based on:
"Eliciting Latent Predictions from Transformers with the Tuned Lens"
by Belrose et al. (NeurIPS 2023)
https://arxiv.org/abs/2303.08112

The tuned lens improves upon the logit lens by learning layer-specific
affine transformations that correct for coordinate system mismatches.

Run this script:
    python examples/tutorials/01b_tuned_lens/tuned_lens_tutorial.py
"""

from mlxterp import InterpretableModel, TunedLens, train_tuned_lens
import mlx.core as mx


def experiment_1_identity_initialization(model):
    """
    Experiment 1: Verify Identity Initialization

    The tuned lens is initialized to identity, meaning an untrained
    tuned lens should behave exactly like the logit lens.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Identity Initialization")
    print("=" * 70)

    # Get dimensions from the model
    num_layers = len(model.layers)

    # Get hidden_dim by running a trace and checking activation shape
    # This is more robust than accessing internal model attributes
    from mlxterp.core.module_resolver import find_layer_key_pattern
    with model.trace("test") as trace:
        pass
    layer_key = find_layer_key_pattern(trace.activations, 0)
    hidden_dim = trace.activations[layer_key].shape[-1]

    print(f"Model has {num_layers} layers with hidden_dim={hidden_dim}")

    # Create untrained tuned lens with model dimensions
    tuned_lens = TunedLens(num_layers=num_layers, hidden_dim=hidden_dim)

    # Check that weights are identity
    for i, translator in enumerate(tuned_lens.translators[:3]):
        is_identity = mx.allclose(translator.weight, mx.eye(hidden_dim), atol=1e-6)
        is_zero_bias = mx.allclose(translator.bias, mx.zeros(hidden_dim), atol=1e-6)
        print(f"Layer {i}: weight=identity: {bool(is_identity.item())}, bias=zeros: {bool(is_zero_bias.item())}")

    print("\n[Conclusion]")
    print("Untrained tuned lens has identity weights, so it behaves like logit lens.")


def experiment_2_train_tuned_lens(model):
    """
    Experiment 2: Train a Tuned Lens

    Train a tuned lens on sample text and observe the loss decrease.

    NOTE: This demo uses reduced settings (50 steps, small dataset) for speed.
    For paper-accurate results, use 250+ steps and 1000+ diverse text samples.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Training a Tuned Lens")
    print("=" * 70)

    # Sample training data (in practice, use much more diverse text)
    # The paper recommends diverse data: Wikipedia, news, code, conversations
    training_texts = [
        "The capital of France is Paris, a beautiful city known for the Eiffel Tower.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn.",
        "Water is composed of two hydrogen atoms and one oxygen atom, forming H2O.",
        "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        "Shakespeare wrote many famous plays including Hamlet, Macbeth, and Romeo and Juliet.",
        "The human brain contains approximately 86 billion neurons connected by synapses.",
        "Python is a popular programming language known for its readability and versatility.",
        "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
        "The Great Wall of China is one of the most impressive architectural feats in history.",
        "Albert Einstein developed the theory of relativity which revolutionized physics.",
    ] * 10  # Repeat for more training data

    print(f"Training on {len(training_texts)} text samples...")
    print("(Note: Paper uses 250 steps; 1000+ samples is a pragmatic recommendation)")

    losses = []

    def callback(step, loss):
        losses.append(loss)
        if step % 10 == 0:
            print(f"  Step {step}: loss = {loss:.4f}")

    # Demo uses reduced settings for speed
    # Paper settings: num_steps=250, learning_rate=1.0, momentum=0.9
    tuned_lens = train_tuned_lens(
        model,
        training_texts,
        num_steps=50,  # Paper recommends 250 for convergence
        max_seq_len=128,
        learning_rate=1.0,
        verbose=False,
        callback=callback
    )

    print(f"\nTraining complete!")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

    return tuned_lens


def experiment_3_compare_predictions(model, tuned_lens):
    """
    Experiment 3: Compare Tuned Lens vs. Logit Lens

    The paper's key finding: tuned lens predictions are more accurate than
    logit lens, especially in early layers.

    NOTE: With reduced demo settings, you may not see clear improvements.
    The paper used 250 steps and diverse training data for reliable results.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Tuned Lens vs. Logit Lens Comparison")
    print("=" * 70)
    print("(Note: With demo settings, improvements may be subtle or absent)")

    test_prompts = [
        ("The capital of France is", "Paris"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("Water freezes at zero degrees", "Celsius"),
    ]

    for prompt, expected in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print(f"Expected: contains '{expected}'")
        print("-" * 50)

        # Get predictions from both methods
        logit_results = model.logit_lens(prompt, top_k=1)
        tuned_results = model.tuned_lens(prompt, tuned_lens, top_k=1)

        print(f"{'Layer':>6} | {'Logit Lens':>15} | {'Tuned Lens':>15}")
        print("-" * 45)

        for layer_idx in [0, 4, 8, 12, max(logit_results.keys())]:
            if layer_idx not in logit_results or layer_idx not in tuned_results:
                continue

            logit_pred = logit_results[layer_idx][-1][0][2]
            tuned_pred = tuned_results[layer_idx][-1][0][2]

            logit_mark = "*" if expected.lower() in logit_pred.lower() else ""
            tuned_mark = "*" if expected.lower() in tuned_pred.lower() else ""

            print(f"{layer_idx:>6} | {logit_pred:>14s}{logit_mark} | {tuned_pred:>14s}{tuned_mark}")

        print("(* = contains expected token)")


def experiment_4_early_layer_improvement(model, tuned_lens):
    """
    Experiment 4: Early Layer Improvement

    Focus on early layers where tuned lens provides the most benefit
    according to the paper.

    NOTE: The paper shows early layers benefit most from tuning because
    their representations are most different from the final layer's
    coordinate system. With full training (250 steps, diverse data),
    you should see clearer token predictions in early layers.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Early Layer Improvement")
    print("=" * 70)
    print("(Paper shows early layers benefit most from tuning)")

    prompt = "The Eiffel Tower is located in the city of"
    early_layers = [0, 1, 2, 3, 4, 5]

    print(f"\nPrompt: '{prompt}'")
    print("\nEarly layer predictions (where tuned lens helps most):")
    print("-" * 60)

    logit_results = model.logit_lens(prompt, layers=early_layers, top_k=3)
    tuned_results = model.tuned_lens(prompt, tuned_lens, layers=early_layers, top_k=3)

    for layer in early_layers:
        print(f"\nLayer {layer}:")

        print("  Logit Lens: ", end="")
        for _, score, token in logit_results[layer][-1][:3]:
            print(f"'{token}' ({score:.2f})", end=" ")

        print("\n  Tuned Lens: ", end="")
        for _, score, token in tuned_results[layer][-1][:3]:
            print(f"'{token}' ({score:.2f})", end=" ")
        print()


def experiment_5_save_and_load(model, tuned_lens):
    """
    Experiment 5: Save and Load Tuned Lens

    Demonstrate persistence of trained tuned lens.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Save and Load Tuned Lens")
    print("=" * 70)

    import tempfile
    import os

    # Save to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_tuned_lens")

        print(f"Saving tuned lens to: {save_path}")
        tuned_lens.save(save_path)

        # Check files exist
        npz_exists = os.path.exists(save_path + ".npz")
        json_exists = os.path.exists(save_path + ".json")
        print(f"  .npz file exists: {npz_exists}")
        print(f"  .json file exists: {json_exists}")

        # Load and compare
        print("\nLoading tuned lens...")
        loaded_lens = TunedLens.load(save_path)

        print(f"  Original layers: {tuned_lens.num_layers}")
        print(f"  Loaded layers: {loaded_lens.num_layers}")
        print(f"  Original dim: {tuned_lens.hidden_dim}")
        print(f"  Loaded dim: {loaded_lens.hidden_dim}")

        # Verify weights match
        weights_match = all(
            mx.allclose(orig.weight, loaded.weight, atol=1e-6).item()
            for orig, loaded in zip(tuned_lens.translators, loaded_lens.translators)
        )
        print(f"  Weights match: {weights_match}")


def main():
    """Run all experiments."""
    print("=" * 70)
    print("TUNED LENS TUTORIAL")
    print("Demonstrating: 'Eliciting Latent Predictions with the Tuned Lens'")
    print("Belrose et al., NeurIPS 2023")
    print("=" * 70)

    # Load model first (needed for all experiments)
    print("\nLoading model...")
    model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
    print(f"Model loaded: {len(model.layers)} layers")

    # Experiment 1: Identity initialization (uses model dimensions)
    experiment_1_identity_initialization(model)

    # Run experiments
    tuned_lens = experiment_2_train_tuned_lens(model)
    experiment_3_compare_predictions(model, tuned_lens)
    experiment_4_early_layer_improvement(model, tuned_lens)
    experiment_5_save_and_load(model, tuned_lens)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key concepts from the paper (Belrose et al., NeurIPS 2023):

1. The tuned lens learns affine transformations (Wx + b) for each layer
   that correct for coordinate system mismatches.

2. Untrained tuned lens (identity initialization) behaves like logit lens.
   [Demonstrated above]

3. Training minimizes KL divergence between each layer's prediction and
   the model's final output.

4. The paper shows tuned lens provides more accurate predictions,
   especially in early layers where the logit lens is most biased.
   (NOTE: Demo uses reduced settings - for paper-accurate results,
   train with 250 steps and 1000+ diverse text samples.)

5. Trained tuned lenses can be saved and loaded for reuse.
   [Demonstrated above]

Paper hyperparameters: SGD with Nesterov momentum, lr=1.0, 250 steps.
    """)


if __name__ == "__main__":
    main()
