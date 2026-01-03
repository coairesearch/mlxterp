#!/usr/bin/env python3
"""
Tutorial 3: Causal Tracing (ROME)

This script demonstrates the causal tracing technique based on:
"Locating and Editing Factual Associations in GPT"
by Meng et al. (NeurIPS 2022)
https://arxiv.org/abs/2202.05262

Causal tracing reveals where factual knowledge is stored in transformers
by corrupting inputs and measuring which layer activations restore correct outputs.

NOTE: This is a simplified implementation. The paper uses:
- Gaussian noise on embeddings (we use subject substitution)
- Position-specific patching (we patch entire sequences)
- Statistical averaging over many examples (we use single examples)

Run this script:
    python examples/tutorials/02_causal_tracing/causal_tracing_tutorial.py
"""

from mlxterp import InterpretableModel
from mlxterp import interventions as iv
import mlx.core as mx


def get_top_predictions_from_logits(model, logits, top_k=3):
    """
    Get top-k predictions from logits (already projected output).

    Unlike get_token_predictions() which expects hidden states,
    this works directly with logits from model.output.
    """
    # Get last position if sequence
    if len(logits.shape) == 3:
        logits = logits[0, -1, :]
    elif len(logits.shape) == 2:
        logits = logits[-1, :]

    # Get top-k indices and scores
    top_indices = mx.argsort(logits)[-top_k:][::-1]
    top_scores = logits[top_indices]

    mx.eval(top_indices, top_scores)

    return [(int(idx), float(score)) for idx, score in zip(top_indices.tolist(), top_scores.tolist())]


def experiment_1_manual_causal_tracing(model):
    """
    Experiment 1: Manual Causal Tracing

    Understand the methodology by implementing it step-by-step.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Manual Causal Tracing")
    print("=" * 70)

    clean_text = "The Eiffel Tower is located in the city of"
    corrupted_text = "The Louvre Museum is located in the city of"

    print(f"\nClean text: '{clean_text}'")
    print(f"Corrupted text: '{corrupted_text}'")

    # Step 1: Get clean and corrupted baselines
    print("\n[Step 1] Getting baseline outputs...")

    with model.trace(clean_text) as trace:
        clean_output = model.output.save()

    mx.eval(clean_output)
    # model.output is logits, so use our helper function
    clean_pred = get_top_predictions_from_logits(model, clean_output, top_k=3)
    print(f"  Clean prediction: {[(model.token_to_str(t), f'{s:.2f}') for t, s in clean_pred]}")

    with model.trace(corrupted_text) as trace:
        corrupted_output = model.output.save()

    mx.eval(corrupted_output)
    corrupted_pred = get_top_predictions_from_logits(model, corrupted_output, top_k=3)
    print(f"  Corrupted prediction: {[(model.token_to_str(t), f'{s:.2f}') for t, s in corrupted_pred]}")

    # Sanity check: verify corruption actually changes prediction
    if clean_pred[0][0] == corrupted_pred[0][0]:
        print("  WARNING: Corruption did not change top prediction!")
        print("  Results may not be meaningful. Try a different corrupted prompt.")

    # Step 2: Patch middle layer and measure recovery
    print("\n[Step 2] Patching middle layer MLP...")

    middle_layer = len(model.layers) // 2

    # Find MLP output key for this layer (not subcomponents like gate_proj)
    with model.trace(clean_text) as trace:
        pass

    mlp_key = None
    # Look for the MLP module output (ends with .mlp, not .mlp.gate_proj etc.)
    for key in sorted(trace.activations.keys()):
        # Match "layers.X.mlp" but not "layers.X.mlp.subcomponent"
        if f"layers.{middle_layer}.mlp" in key:
            # Check it's the MLP output, not a subcomponent
            after_mlp = key.split(f"layers.{middle_layer}.mlp")[-1]
            if after_mlp == "" or after_mlp.startswith(".") is False:
                mlp_key = key
                break

    if mlp_key is None:
        # Fallback: find key ending with just ".mlp"
        for key in trace.activations:
            if key.endswith(f"layers.{middle_layer}.mlp"):
                mlp_key = key
                break

    if mlp_key is None:
        # Last fallback: layer output
        for key in trace.activations:
            if key.endswith(f"layers.{middle_layer}"):
                mlp_key = key
                break

    if mlp_key:
        clean_mlp = trace.activations[mlp_key]
        mx.eval(clean_mlp)

        # Build intervention key (remove model prefixes)
        if mlp_key.startswith("model.model."):
            intervention_key = mlp_key[12:]
        elif mlp_key.startswith("model."):
            intervention_key = mlp_key[6:]
        else:
            intervention_key = mlp_key

        print(f"  Patching key: {intervention_key}")

        with model.trace(corrupted_text,
                         interventions={intervention_key: iv.replace_with(clean_mlp)}):
            patched_output = model.output.save()

        mx.eval(patched_output)
        # model.output is logits, use our helper
        patched_pred = get_top_predictions_from_logits(model, patched_output, top_k=3)
        print(f"  Patched prediction: {[(model.token_to_str(t), f'{s:.2f}') for t, s in patched_pred]}")

        # Calculate recovery using L2 distance on logit vectors
        # NOTE: The paper uses target-token probability recovery, not L2 distance.
        # L2 distance is a simpler proxy metric for this tutorial.
        clean_vec = clean_output[0, -1].astype(mx.float32)
        corrupted_vec = corrupted_output[0, -1].astype(mx.float32)
        patched_vec = patched_output[0, -1].astype(mx.float32)

        baseline_dist = float(mx.sqrt(mx.sum((corrupted_vec - clean_vec) ** 2)))
        patched_dist = float(mx.sqrt(mx.sum((patched_vec - clean_vec) ** 2)))

        if baseline_dist > 1e-6:
            recovery = (baseline_dist - patched_dist) / baseline_dist * 100
        else:
            recovery = 0.0

        print(f"\n  Recovery: {recovery:.1f}%")
        print(f"  (Higher = patching helped restore correct output)")
        print(f"  (Note: Using L2 distance metric, not paper's target-token probability)")
    else:
        print("  Warning: Could not find MLP activation key")

    print("\n[Conclusion]")
    print("Manual causal tracing shows how patching specific layers can")
    print("restore correct outputs after corruption.")


def experiment_2_automated_patching(model):
    """
    Experiment 2: Automated Activation Patching

    Use mlxterp's built-in activation_patching() method.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Automated Activation Patching")
    print("=" * 70)

    clean_text = "The Eiffel Tower is located in the city of"
    corrupted_text = "The Louvre Museum is located in the city of"

    print(f"\nClean: '{clean_text}'")
    print(f"Corrupted: '{corrupted_text}'")

    print("\n[MLP Component Patching]")
    results = model.activation_patching(
        clean_text=clean_text,
        corrupted_text=corrupted_text,
        component="mlp",
        plot=False
    )

    # Show results
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print("\nMost important MLP layers:")
    for layer, recovery in sorted_results[:5]:
        bar = "+" * int(max(0, recovery) / 5)
        print(f"  Layer {layer:2d}: {recovery:6.1f}% {bar}")

    print("\nLeast important MLP layers:")
    for layer, recovery in sorted_results[-3:]:
        print(f"  Layer {layer:2d}: {recovery:6.1f}%")


def experiment_3_mlp_vs_attention(model):
    """
    Experiment 3: MLP vs. Attention Comparison

    The paper found MLPs are more important than attention for factual recall.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: MLP vs. Attention Comparison")
    print("=" * 70)
    print("(Paper finding: MLPs store factual knowledge more than attention)")

    clean_text = "Albert Einstein developed the theory of"
    corrupted_text = "Isaac Newton developed the theory of"

    print(f"\nClean: '{clean_text}'")
    print(f"Corrupted: '{corrupted_text}'")

    # MLP patching
    print("\nPatching MLP components...")
    mlp_results = model.activation_patching(
        clean_text=clean_text,
        corrupted_text=corrupted_text,
        component="mlp",
        plot=False
    )

    # Attention patching
    print("Patching attention components...")
    attn_results = model.activation_patching(
        clean_text=clean_text,
        corrupted_text=corrupted_text,
        component="self_attn",
        plot=False
    )

    # Compare
    print("\n" + "-" * 50)
    print("Layer | MLP Recovery | Attn Recovery | Winner")
    print("-" * 50)

    mlp_wins = 0
    attn_wins = 0

    for layer in sorted(mlp_results.keys()):
        mlp_rec = mlp_results.get(layer, 0)
        attn_rec = attn_results.get(layer, 0)

        if mlp_rec > attn_rec:
            winner = "MLP"
            mlp_wins += 1
        elif attn_rec > mlp_rec:
            winner = "Attn"
            attn_wins += 1
        else:
            winner = "Tie"

        print(f"{layer:5d} | {mlp_rec:10.1f}% | {attn_rec:10.1f}% | {winner}")

    print("-" * 50)
    print(f"\nMLP wins: {mlp_wins} layers")
    print(f"Attention wins: {attn_wins} layers")

    total_mlp = sum(mlp_results.values())
    total_attn = sum(attn_results.values())
    print(f"\nTotal recovery - MLP: {total_mlp:.1f}%, Attention: {total_attn:.1f}%")

    if total_mlp > total_attn:
        print("\n[Result] MLPs contribute more to factual recall (consistent with paper)")
    else:
        print("\n[Result] Attention contributes more (may vary by model/prompt)")


def experiment_4_layer_distribution(model):
    """
    Experiment 4: Layer Distribution Analysis

    The paper found factual knowledge concentrates in middle layers.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Layer Distribution Analysis")
    print("=" * 70)
    print("(Paper finding: Middle layers store factual associations)")

    clean_text = "The capital of France is"
    corrupted_text = "The capital of Germany is"

    print(f"\nClean: '{clean_text}'")
    print(f"Corrupted: '{corrupted_text}'")

    print("\nRunning activation patching...")
    results = model.activation_patching(
        clean_text=clean_text,
        corrupted_text=corrupted_text,
        component="mlp",
        plot=False
    )

    # Analyze distribution
    layers = sorted(results.keys())
    n_layers = len(layers)

    early_end = n_layers // 3
    late_start = 2 * n_layers // 3

    early_layers = layers[:early_end]
    middle_layers = layers[early_end:late_start]
    late_layers = layers[late_start:]

    def avg_recovery(layer_list):
        if not layer_list:
            return 0.0
        return sum(results.get(l, 0) for l in layer_list) / len(layer_list)

    early_avg = avg_recovery(early_layers)
    middle_avg = avg_recovery(middle_layers)
    late_avg = avg_recovery(late_layers)

    print(f"\nLayer distribution (total {n_layers} layers):")
    print("-" * 45)

    print(f"Early (0-{early_end-1}):  avg {early_avg:6.1f}%  ", end="")
    print("=" * int(max(0, early_avg) / 3))

    print(f"Middle ({early_end}-{late_start-1}): avg {middle_avg:6.1f}%  ", end="")
    print("=" * int(max(0, middle_avg) / 3))

    print(f"Late ({late_start}-{n_layers-1}):  avg {late_avg:6.1f}%  ", end="")
    print("=" * int(max(0, late_avg) / 3))

    # Find peak
    peak_layer = max(results, key=results.get) if results else 0
    peak_recovery = results.get(peak_layer, 0)
    print(f"\nPeak: Layer {peak_layer} ({peak_recovery:.1f}% recovery)")

    # Determine which region has highest recovery
    max_avg = max(early_avg, middle_avg, late_avg)
    if max_avg == middle_avg:
        print("\n[Result] Middle layers show highest recovery (consistent with paper)")
    elif max_avg == early_avg:
        print("\n[Result] Early layers show highest recovery (may vary by prompt)")
    else:
        print("\n[Result] Late layers show highest recovery (may vary by prompt)")


def experiment_5_multiple_facts(model):
    """
    Experiment 5: Multiple Factual Prompts

    Test localization across different types of facts.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Multiple Factual Prompts")
    print("=" * 70)

    test_cases = [
        ("The Eiffel Tower is in", "The Statue of Liberty is in", "Location"),
        ("The capital of France is", "The capital of Japan is", "Capital city"),
        ("Water freezes at zero degrees", "Water boils at zero degrees", "Science fact"),
        ("Shakespeare wrote", "Dickens wrote", "Authorship"),
    ]

    print("\nAnalyzing peak MLP layer for different fact types:")
    print("-" * 60)

    for clean, corrupted, fact_type in test_cases:
        print(f"\n[{fact_type}]")
        print(f"  Clean: '{clean}'")

        results = model.activation_patching(
            clean_text=clean,
            corrupted_text=corrupted,
            component="mlp",
            plot=False
        )

        if results:
            peak = max(results, key=results.get)
            recovery = results[peak]
            print(f"  Peak layer: {peak} ({recovery:.1f}% recovery)")
        else:
            print("  No results")

    print("\n[Observation]")
    print("Different facts may be stored at different layers,")
    print("but middle layers tend to be most important overall.")


def main():
    """Run all experiments."""
    print("=" * 70)
    print("CAUSAL TRACING TUTORIAL")
    print("Reproducing: 'Locating and Editing Factual Associations in GPT'")
    print("Meng et al., NeurIPS 2022")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("NOTE: This is a simplified implementation for educational purposes.")
    print("The paper uses Gaussian noise corruption and position-specific patching.")
    print("We use subject substitution and full-sequence patching for simplicity.")
    print("-" * 70)

    # Load model
    print("\nLoading model...")
    model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
    print(f"Model loaded: {len(model.layers)} layers")

    # Run experiments
    experiment_1_manual_causal_tracing(model)
    experiment_2_automated_patching(model)
    experiment_3_mlp_vs_attention(model)
    experiment_4_layer_distribution(model)
    experiment_5_multiple_facts(model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key concepts from the paper (Meng et al., NeurIPS 2022):

1. Causal tracing identifies where factual knowledge is stored by
   measuring which layer activations restore correct outputs after
   input corruption.

2. The paper found factual associations are stored primarily in
   MLP modules at middle layers.

3. Knowledge is localized at the subject's last token position
   (our simplified implementation patches the full sequence).

4. This technique enables model editing: surgically updating specific
   facts without full retraining.

Limitations of our implementation vs. the paper:
- Corruption: Subject substitution instead of Gaussian noise on embeddings
- Patching: Entire sequences instead of position-specific (subject's last token)
- Metric: L2 distance on logits instead of target-token probability recovery
- Statistics: Single examples instead of averaging over many prompts

These simplifications make the demo faster and clearer but may affect
the quantitative results. For rigorous scientific conclusions, implement
the full paper methodology.
    """)


if __name__ == "__main__":
    main()
