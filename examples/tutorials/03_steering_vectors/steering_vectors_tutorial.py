#!/usr/bin/env python3
"""
Tutorial 4: Steering Vectors (Contrastive Activation Addition)

This script demonstrates the steering vector technique based on:
"Steering Llama 2 via Contrastive Activation Addition"
by Rimsky et al. (ACL 2024)
https://arxiv.org/abs/2312.06681

Steering vectors allow controlling model behavior by adding learned
directional vectors to intermediate activations during inference.

NOTE: Model choice
- The paper used Llama 2 models; this tutorial uses Llama-3.2-1B-Instruct
  for accessibility. The methodology generalizes across transformers.

NOTE: Simplified contrastive pairs
- The paper uses matched pairs where prompts differ only in target behavior.
  Our examples use different prompts for simplicity, which may capture some
  topic/wording differences. For rigorous steering, use matched templates.

NOTE: Evaluation scope
- This tutorial evaluates next-token predictions only.
  The paper evaluates behavioral changes in full generated completions.

Run this script:
    python examples/tutorials/03_steering_vectors/steering_vectors_tutorial.py
"""

from mlxterp import InterpretableModel
from mlxterp import interventions as iv
import mlx.core as mx


def get_top_from_logits(model, logits, top_k=5):
    """Get top-k predictions from logits."""
    if len(logits.shape) == 3:
        logits = logits[0, -1, :]
    elif len(logits.shape) == 2:
        logits = logits[-1, :]

    top_indices = mx.argsort(logits)[-top_k:][::-1]
    top_scores = logits[top_indices]
    mx.eval(top_indices, top_scores)
    return [(int(i), float(s)) for i, s in zip(top_indices.tolist(), top_scores.tolist())]


def find_layer_key(trace_activations, layer_idx):
    """Find the activation key for a given layer."""
    for key in trace_activations:
        if key.endswith(f"layers.{layer_idx}"):
            return key
    return None


def collect_activations(model, prompts, layer_idx):
    """Collect activations from prompts at a specific layer."""
    activations = []

    for prompt in prompts:
        with model.trace(prompt) as trace:
            pass

        layer_key = find_layer_key(trace.activations, layer_idx)
        if layer_key:
            act = trace.activations[layer_key]
            activations.append(act[0, -1, :])  # Last token position

    if activations:
        mx.eval(activations)

    return activations


def compute_steering_vector(positive_acts, negative_acts):
    """Compute steering vector from contrastive activations."""
    if not positive_acts or not negative_acts:
        return None

    positive_stack = mx.stack(positive_acts)
    negative_stack = mx.stack(negative_acts)

    positive_mean = mx.mean(positive_stack, axis=0)
    negative_mean = mx.mean(negative_stack, axis=0)

    steering_vector = positive_mean - negative_mean
    mx.eval(steering_vector)

    return steering_vector


def experiment_1_basic_steering(model):
    """
    Experiment 1: Basic Sentiment Steering

    Demonstrates the core CAA technique with sentiment as the behavioral dimension.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Basic Sentiment Steering")
    print("=" * 70)

    # Contrastive prompts for sentiment
    positive_prompts = [
        "I think this is great because",
        "I love the way this works since",
        "This makes me happy because",
        "I'm excited about this because",
        "This is wonderful because",
    ]

    negative_prompts = [
        "I think this is terrible because",
        "I hate the way this works since",
        "This makes me sad because",
        "I'm worried about this because",
        "This is awful because",
    ]

    layer_idx = len(model.layers) // 2
    print(f"\nUsing layer {layer_idx} (middle of {len(model.layers)} layers)")

    # Collect activations
    print("Collecting positive activations...")
    positive_acts = collect_activations(model, positive_prompts, layer_idx)
    print(f"  Collected {len(positive_acts)} activations")

    print("Collecting negative activations...")
    negative_acts = collect_activations(model, negative_prompts, layer_idx)
    print(f"  Collected {len(negative_acts)} activations")

    # Compute steering vector
    steering_vector = compute_steering_vector(positive_acts, negative_acts)
    if steering_vector is None:
        print("Error: Could not compute steering vector")
        return None

    norm = float(mx.sqrt(mx.sum(steering_vector ** 2)))
    print(f"\nSteering vector computed (norm: {norm:.4f})")

    # Test steering
    test_prompt = "This product is"
    intervention_key = f"layers.{layer_idx}"

    print(f"\nTest prompt: '{test_prompt}'")
    print("-" * 50)

    # Normal output
    with model.trace(test_prompt) as trace:
        normal_output = model.output.save()

    mx.eval(normal_output)
    normal_pred = get_top_from_logits(model, normal_output, top_k=5)
    print("\nNormal (no steering):")
    for token_id, score in normal_pred[:3]:
        print(f"  '{model.token_to_str(token_id)}': {score:.2f}")

    # Positive steering
    strength = 2.0
    with model.trace(test_prompt,
                     interventions={intervention_key: iv.add_vector(steering_vector * strength)}):
        pos_output = model.output.save()

    mx.eval(pos_output)
    pos_pred = get_top_from_logits(model, pos_output, top_k=5)
    print(f"\nPositive steering (strength={strength}):")
    for token_id, score in pos_pred[:3]:
        print(f"  '{model.token_to_str(token_id)}': {score:.2f}")

    # Negative steering
    with model.trace(test_prompt,
                     interventions={intervention_key: iv.add_vector(-steering_vector * strength)}):
        neg_output = model.output.save()

    mx.eval(neg_output)
    neg_pred = get_top_from_logits(model, neg_output, top_k=5)
    print(f"\nNegative steering (strength={-strength}):")
    for token_id, score in neg_pred[:3]:
        print(f"  '{model.token_to_str(token_id)}': {score:.2f}")

    return steering_vector, layer_idx


def experiment_2_strength_analysis(model, steering_vector, layer_idx):
    """
    Experiment 2: Steering Strength Analysis

    Shows how different strengths affect model predictions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Steering Strength Analysis")
    print("=" * 70)

    test_prompt = "The movie was"
    intervention_key = f"layers.{layer_idx}"
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    print(f"\nTest prompt: '{test_prompt}'")
    print("-" * 50)
    print(f"{'Strength':>8} | Top Prediction")
    print("-" * 50)

    for strength in strengths:
        if strength == 0.0:
            with model.trace(test_prompt) as trace:
                output = model.output.save()
        else:
            with model.trace(test_prompt,
                             interventions={intervention_key: iv.add_vector(steering_vector * strength)}):
                output = model.output.save()

        mx.eval(output)
        pred = get_top_from_logits(model, output, top_k=1)
        token_str = model.token_to_str(pred[0][0])
        print(f"{strength:8.1f} | '{token_str}'")

    print("\n[Observation]")
    print("Low strength: Minimal effect")
    print("Medium strength: Clear behavioral shift")
    print("High strength: May cause degradation")


def experiment_3_multi_layer_steering(model):
    """
    Experiment 3: Multi-Layer Steering

    The paper found steering multiple layers can be more effective.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Multi-Layer Steering")
    print("=" * 70)

    # Contrastive prompts
    positive_prompts = [
        "I think this is great because",
        "I love the way this works since",
        "This makes me happy because",
    ]

    negative_prompts = [
        "I think this is terrible because",
        "I hate the way this works since",
        "This makes me sad because",
    ]

    # Compute steering vectors for multiple layers
    n_layers = len(model.layers)
    layers_to_steer = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

    print(f"\nSteering layers: {layers_to_steer}")

    steering_vectors = {}
    for layer_idx in layers_to_steer:
        pos_acts = collect_activations(model, positive_prompts, layer_idx)
        neg_acts = collect_activations(model, negative_prompts, layer_idx)

        sv = compute_steering_vector(pos_acts, neg_acts)
        if sv is not None:
            steering_vectors[layer_idx] = sv
            print(f"  Layer {layer_idx}: computed")

    # Compare single vs multi-layer steering
    test_prompt = "This experience was"
    strength = 1.5

    print(f"\nTest prompt: '{test_prompt}'")
    print("-" * 50)

    # Normal
    with model.trace(test_prompt) as trace:
        normal_output = model.output.save()

    mx.eval(normal_output)
    normal_pred = get_top_from_logits(model, normal_output, top_k=3)
    print("\nNormal:")
    for token_id, score in normal_pred:
        print(f"  '{model.token_to_str(token_id)}'")

    # Single layer (middle)
    middle_layer = layers_to_steer[1]
    with model.trace(test_prompt,
                     interventions={f"layers.{middle_layer}": iv.add_vector(steering_vectors[middle_layer] * strength)}):
        single_output = model.output.save()

    mx.eval(single_output)
    single_pred = get_top_from_logits(model, single_output, top_k=3)
    print(f"\nSingle layer ({middle_layer}):")
    for token_id, score in single_pred:
        print(f"  '{model.token_to_str(token_id)}'")

    # Multi-layer
    interventions = {
        f"layers.{layer_idx}": iv.add_vector(vec * strength)
        for layer_idx, vec in steering_vectors.items()
    }

    with model.trace(test_prompt, interventions=interventions):
        multi_output = model.output.save()

    mx.eval(multi_output)
    multi_pred = get_top_from_logits(model, multi_output, top_k=3)
    print(f"\nMulti-layer ({list(steering_vectors.keys())}):")
    for token_id, score in multi_pred:
        print(f"  '{model.token_to_str(token_id)}'")


def experiment_4_different_behaviors(model):
    """
    Experiment 4: Different Behavioral Dimensions

    Test steering for various behavioral traits.
    NOTE: The paper tested sycophancy, corrigibility, etc.
    These examples (formal/casual, confident/uncertain) are additional
    demonstrations not from the paper.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Different Behavioral Dimensions")
    print("=" * 70)
    print("(Additional examples, not from the paper)")

    behaviors = {
        "formal_casual": {
            "positive": [
                "I would like to formally address",
                "It is my professional opinion that",
                "Upon careful consideration,",
            ],
            "negative": [
                "Hey so basically like",
                "Yo check this out",
                "Dude you gotta see",
            ],
            "test": "Dear Sir or Madam,",
        },
        "confident_uncertain": {
            "positive": [
                "I am absolutely certain that",
                "There is no doubt that",
                "It is definitely true that",
            ],
            "negative": [
                "I'm not entirely sure but",
                "It might be possible that",
                "Perhaps this could be",
            ],
            "test": "The answer to your question is",
        },
    }

    layer_idx = len(model.layers) // 2

    for behavior_name, prompts in behaviors.items():
        print(f"\n[{behavior_name.upper()}]")
        print("-" * 40)

        # Compute steering vector
        pos_acts = collect_activations(model, prompts["positive"], layer_idx)
        neg_acts = collect_activations(model, prompts["negative"], layer_idx)
        sv = compute_steering_vector(pos_acts, neg_acts)

        if sv is None:
            print("  Could not compute steering vector")
            continue

        test_prompt = prompts["test"]
        print(f"Test: '{test_prompt}'")

        # Normal
        with model.trace(test_prompt) as trace:
            normal_output = model.output.save()

        mx.eval(normal_output)
        normal_pred = get_top_from_logits(model, normal_output, top_k=1)
        print(f"  Normal: '{model.token_to_str(normal_pred[0][0])}'")

        # Positive steering
        strength = 2.0
        with model.trace(test_prompt,
                         interventions={f"layers.{layer_idx}": iv.add_vector(sv * strength)}):
            pos_output = model.output.save()

        mx.eval(pos_output)
        pos_pred = get_top_from_logits(model, pos_output, top_k=1)
        print(f"  +Steering: '{model.token_to_str(pos_pred[0][0])}'")

        # Negative steering
        with model.trace(test_prompt,
                         interventions={f"layers.{layer_idx}": iv.add_vector(-sv * strength)}):
            neg_output = model.output.save()

        mx.eval(neg_output)
        neg_pred = get_top_from_logits(model, neg_output, top_k=1)
        print(f"  -Steering: '{model.token_to_str(neg_pred[0][0])}'")


def experiment_5_layer_comparison(model):
    """
    Experiment 5: Layer Effectiveness Comparison

    Find which layers are most effective for steering.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Layer Effectiveness Comparison")
    print("=" * 70)

    positive_prompts = [
        "I think this is great because",
        "I love the way this works since",
        "This makes me happy because",
    ]

    negative_prompts = [
        "I think this is terrible because",
        "I hate the way this works since",
        "This makes me sad because",
    ]

    test_prompt = "The food at this restaurant is"
    strength = 2.0

    # Test every 4th layer
    layers_to_test = list(range(0, len(model.layers), 4))

    print(f"\nTest prompt: '{test_prompt}'")
    print(f"Testing layers: {layers_to_test}")
    print("-" * 50)
    print(f"{'Layer':>6} | Normal     | +Steering  | -Steering")
    print("-" * 50)

    # Get normal prediction once
    with model.trace(test_prompt) as trace:
        normal_output = model.output.save()

    mx.eval(normal_output)
    normal_pred = get_top_from_logits(model, normal_output, top_k=1)
    normal_token = model.token_to_str(normal_pred[0][0])[:10]

    for layer_idx in layers_to_test:
        pos_acts = collect_activations(model, positive_prompts, layer_idx)
        neg_acts = collect_activations(model, negative_prompts, layer_idx)
        sv = compute_steering_vector(pos_acts, neg_acts)

        if sv is None:
            continue

        # Positive steering
        with model.trace(test_prompt,
                         interventions={f"layers.{layer_idx}": iv.add_vector(sv * strength)}):
            pos_output = model.output.save()

        mx.eval(pos_output)
        pos_pred = get_top_from_logits(model, pos_output, top_k=1)
        pos_token = model.token_to_str(pos_pred[0][0])[:10]

        # Negative steering
        with model.trace(test_prompt,
                         interventions={f"layers.{layer_idx}": iv.add_vector(-sv * strength)}):
            neg_output = model.output.save()

        mx.eval(neg_output)
        neg_pred = get_top_from_logits(model, neg_output, top_k=1)
        neg_token = model.token_to_str(neg_pred[0][0])[:10]

        changed = "*" if pos_token != normal_token or neg_token != normal_token else " "
        print(f"{layer_idx:6d} | {normal_token:10s} | {pos_token:10s} | {neg_token:10s} {changed}")

    print("\n(* = steering changed prediction)")


def main():
    """Run all experiments."""
    print("=" * 70)
    print("STEERING VECTORS TUTORIAL")
    print("Demonstrating: 'Steering Llama 2 via Contrastive Activation Addition'")
    print("Rimsky et al., ACL 2024")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
    print(f"Model loaded: {len(model.layers)} layers")

    # Run experiments
    result = experiment_1_basic_steering(model)
    if result:
        steering_vector, layer_idx = result
        experiment_2_strength_analysis(model, steering_vector, layer_idx)

    experiment_3_multi_layer_steering(model)
    experiment_4_different_behaviors(model)
    experiment_5_layer_comparison(model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key concepts from the paper (Rimsky et al., ACL 2024):

1. Contrastive Activation Addition (CAA) computes steering vectors
   as the difference between positive and negative example activations.

2. Adding steering vectors to intermediate layers shifts model behavior
   along the corresponding behavioral dimension.

3. Steering strength controls the magnitude of the effect:
   - Too low: No visible change
   - Typical range (1-3, heuristic): Clear behavioral shift
   - Too high: May cause degradation or incoherence

4. Multi-layer steering can be more effective than single-layer.

5. Different behavioral dimensions (sentiment, formality, confidence)
   can be independently steered.

Applications:
- Bias reduction
- Tone/style control
- Safety alignment
- Controllable generation

Limitations:
- Effectiveness varies by prompt and model
- Complex behaviors may not reduce to single directions
- Requires careful tuning of strength and layers
    """)


if __name__ == "__main__":
    main()
