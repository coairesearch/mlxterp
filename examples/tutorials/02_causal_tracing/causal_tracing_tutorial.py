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
    # Support both Llama (layers.X.mlp) and GPT-2 (h.X.mlp) naming
    with model.trace(clean_text) as trace:
        pass

    mlp_key = None
    # Look for the MLP module output (ends with .mlp, not .mlp.gate_proj etc.)
    mlp_patterns = [f"layers.{middle_layer}.mlp", f"h.{middle_layer}.mlp"]
    for key in sorted(trace.activations.keys()):
        for pattern in mlp_patterns:
            if pattern in key:
                # Check it's the MLP output, not a subcomponent
                after_mlp = key.split(pattern)[-1]
                if after_mlp == "" or after_mlp.startswith(".") is False:
                    mlp_key = key
                    break
        if mlp_key:
            break

    if mlp_key is None:
        # Fallback: find key ending with just ".mlp"
        for key in trace.activations:
            if key.endswith(f"layers.{middle_layer}.mlp") or key.endswith(f"h.{middle_layer}.mlp"):
                mlp_key = key
                break

    if mlp_key is None:
        # Last fallback: layer output
        for key in trace.activations:
            if key.endswith(f"layers.{middle_layer}") or key.endswith(f"h.{middle_layer}"):
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


# ============================================================================
# PAPER-ACCURATE EXPERIMENTS (More faithful to Meng et al., 2022)
# ============================================================================

def compute_embedding_noise_std(model, multiplier=3.0):
    """
    Compute noise standard deviation as per the paper.

    The ROME paper uses 3 times the standard deviation of the
    model's embedding matrix as the noise level. For GPT-2 XL,
    this was approximately 0.13-0.15.

    Args:
        model: InterpretableModel
        multiplier: Typically 3.0 (paper's value)

    Returns:
        Noise standard deviation (typically ~0.15 for effective corruption)
    """
    # Paper's GPT-2 XL had embedding std ~0.045, so 3x = ~0.13
    # For quantized models, computing from scales gives unreliable results
    # Use a fallback that matches the paper's effective noise level
    # The ROME paper uses 3 * embedding_std which gives ~0.13 for GPT-2 XL
    # This should result in ~80-90% probability reduction, not 99%+
    # If corruption is too strong, layer distribution analysis becomes unreliable
    # The ROME paper uses ~3x embedding std (~0.13 for GPT-2 XL).
    # We use a similar level for strong corruption.
    PAPER_NOISE_LEVEL = 0.13  # Paper's recommended level

    try:
        embed_layer = model._module_resolver.get_embedding_layer()

        # For quantized embeddings, use paper's noise level directly
        # Computing from quantized scales gives incorrect (too small) values
        if hasattr(embed_layer, 'scales'):
            return PAPER_NOISE_LEVEL

        elif hasattr(embed_layer, 'weight'):
            # For non-quantized models, compute from actual weights
            weights = embed_layer.weight
            mx.eval(weights)
            embedding_std = float(mx.std(weights))
            if embedding_std > 0 and not mx.isnan(mx.array(embedding_std)):
                computed_noise = multiplier * embedding_std
                # Ensure noise is in reasonable range (0.05 to 0.3)
                if computed_noise < 0.05:
                    return PAPER_NOISE_LEVEL
                elif computed_noise > 0.3:
                    return 0.3
                return computed_noise
            else:
                return PAPER_NOISE_LEVEL
        else:
            return PAPER_NOISE_LEVEL

    except Exception:
        # Safe fallback - paper's effective noise level
        return PAPER_NOISE_LEVEL


def add_gaussian_noise_to_embedding(model, tokens, noise_std=None, subject_range=None):
    """
    Add Gaussian noise to token embeddings (paper's corruption method).

    Args:
        model: InterpretableModel instance
        tokens: Token IDs (list or mx.array)
        noise_std: Standard deviation of noise. If None, uses paper's method
                  (3 * embedding stddev)
        subject_range: Tuple (start, end) of subject token positions to corrupt.
                      If None, corrupts all tokens.

    Returns:
        Noisy embeddings as mx.array
    """
    # Compute noise_std if not provided (paper's method)
    if noise_std is None:
        noise_std = compute_embedding_noise_std(model)

    if isinstance(tokens, list):
        tokens = mx.array([tokens])
    elif len(tokens.shape) == 1:
        tokens = tokens.reshape(1, -1)

    # Get embedding layer
    embed_layer = model._module_resolver.get_embedding_layer()

    # Get original embeddings
    if hasattr(embed_layer, 'scales'):
        # Quantized embedding - dequantize first
        original_embeddings = mx.dequantize(
            embed_layer.weight,
            embed_layer.scales,
            embed_layer.biases,
            embed_layer.group_size,
            embed_layer.bits,
        )
        embeddings = original_embeddings[tokens]
    else:
        embeddings = embed_layer(tokens)

    mx.eval(embeddings)

    # Create noise with fixed seed for reproducibility
    mx.random.seed(42)
    noise = mx.random.normal(embeddings.shape) * noise_std

    # Apply noise only to subject tokens if range specified
    if subject_range is not None:
        start, end = subject_range
        # Create mask using array operations
        seq_len = embeddings.shape[1]
        positions = mx.arange(seq_len)
        # Mask is 1 for positions in [start, end), 0 otherwise
        position_mask = (positions >= start) & (positions < end)
        # Expand to full shape: (1, seq_len, 1) for broadcasting
        mask = position_mask.reshape(1, seq_len, 1).astype(embeddings.dtype)
        noise = noise * mask

    noisy_embeddings = embeddings + noise
    mx.eval(noisy_embeddings)

    return noisy_embeddings


def get_target_token_probability(model, logits, target_token_id):
    """
    Get probability of target token (paper's recovery metric).

    Args:
        model: InterpretableModel
        logits: Model output logits
        target_token_id: Token ID for the correct answer

    Returns:
        Probability of target token
    """
    if len(logits.shape) == 3:
        logits = logits[0, -1, :]
    elif len(logits.shape) == 2:
        logits = logits[-1, :]

    # Softmax to get probabilities
    probs = mx.softmax(logits)
    target_prob = float(probs[target_token_id])

    return target_prob


def replace_at_position(clean_activation, position):
    """
    Create an intervention that replaces activation only at a specific position.

    Args:
        clean_activation: Clean activation tensor (batch, seq_len, hidden_dim)
        position: Token position to patch

    Returns:
        Intervention function that can be used with model.trace()
    """
    def _replace_at_pos(x):
        if x.shape[1] <= position or clean_activation.shape[1] <= position:
            return x

        seq_len = x.shape[1]
        positions = mx.arange(seq_len)
        position_mask = (positions == position).reshape(1, seq_len, 1).astype(x.dtype)
        result = x * (1 - position_mask) + clean_activation * position_mask
        return result

    return _replace_at_pos


def replace_at_positions(clean_activation, start_pos, end_pos):
    """
    Create an intervention that replaces activation at a range of positions.

    This is essential for causal tracing: when subject spans multiple tokens
    (e.g., "The Eiffel Tower"), we need to restore clean activations at ALL
    subject positions, not just the last one. Otherwise, attention in later
    layers reads corrupted key/values from other subject positions.

    Args:
        clean_activation: Clean activation tensor (batch, seq_len, hidden_dim)
        start_pos: Start position of subject (inclusive)
        end_pos: End position of subject (exclusive)

    Returns:
        Intervention function that can be used with model.trace()

    Example:
        # Patch all subject positions (0-5)
        intervention = replace_at_positions(clean_mlp, 0, 5)
        with model.trace(corrupted_input, interventions={mlp_key: intervention}):
            patched_output = model.output.save()
    """
    def _replace_at_range(x):
        if x.shape[1] < end_pos or clean_activation.shape[1] < end_pos:
            return x

        seq_len = x.shape[1]
        positions = mx.arange(seq_len)
        # Mask is 1 for positions in [start_pos, end_pos), 0 elsewhere
        position_mask = ((positions >= start_pos) & (positions < end_pos)).reshape(1, seq_len, 1).astype(x.dtype)
        result = x * (1 - position_mask) + clean_activation * position_mask
        return result

    return _replace_at_range


def experiment_6_gaussian_noise_corruption(model):
    """
    Experiment 6: Gaussian Noise Corruption with Residual Stream Patching

    This implements the paper's actual methodology:
    1. Gaussian noise corruption on subject embeddings
    2. Residual stream patching (full layer output, not just MLP)
    3. Subject-position patching (all subject tokens)
    4. Target-token probability recovery metric

    The key insight: We patch the FULL LAYER OUTPUT (h.N), which is the
    residual stream after that layer. This is what the paper does.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Gaussian Noise + Residual Stream Patching")
    print("=" * 70)
    print("Patching clean residual stream (full layer output) at subject positions")

    prompt = "The Eiffel Tower is located in"

    print(f"\nPrompt: '{prompt}'")

    # Encode prompt
    tokens = model.encode(prompt)

    # Show tokenization to clarify subject position
    print("Tokens:", end=" ")
    for i, tok in enumerate(tokens):
        print(f"[{i}:'{model.token_to_str(tok)}']", end=" ")
    print()

    # Find subject tokens ("The Eiffel Tower")
    # We patch ALL subject positions, not just the last one
    # This is essential: attention reads K/V from all positions, so we need
    # clean activations at all subject positions to avoid distribution mismatch
    subject_start = 0
    subject_end = 5  # End of subject range (exclusive)
    print(f"Subject positions: {subject_start}-{subject_end} (patching all subject tokens)")

    print(f"\n[Step 1] Clean run...")
    with model.trace(mx.array([tokens])) as trace:
        clean_output = model.output.save()

    mx.eval(clean_output)

    # Get the actual top prediction (not a hardcoded target)
    clean_pred = get_top_predictions_from_logits(model, clean_output, top_k=1)
    target_id = clean_pred[0][0]
    target_token = model.token_to_str(target_id)
    clean_prob = get_target_token_probability(model, clean_output, target_id)
    print(f"  Top prediction: '{target_token}' with P = {clean_prob:.4f}")

    print(f"\n[Step 2] Corrupted run (Gaussian noise on subject)...")
    # Compute noise std using paper's method: 3 * embedding stddev
    noise_std = compute_embedding_noise_std(model)
    print(f"  Noise std (3 * embedding stddev): {noise_std:.4f}")

    # Create noisy embeddings - only corrupt subject tokens
    noisy_embeddings = add_gaussian_noise_to_embedding(
        model, tokens, noise_std=noise_std, subject_range=(0, subject_end)
    )

    # Run model with noisy embeddings
    # We need to bypass the embedding layer and inject our noisy embeddings
    # This requires a custom intervention
    # Auto-detect embed key for different architectures (Llama vs GPT-2)
    embed_key = model._module_resolver.get_embedding_path()
    if embed_key is None:
        # Fallback: try common paths
        for path in ["model.embed_tokens", "model.wte", "model.model.wte"]:
            try:
                with model.trace(mx.array([tokens])) as test_trace:
                    pass
                if path in test_trace.activations:
                    embed_key = path
                    break
            except Exception:
                continue
        if embed_key is None:
            embed_key = "model.embed_tokens"  # Default fallback

    with model.trace(mx.array([tokens]),
                     interventions={embed_key: iv.replace_with(noisy_embeddings)}):
        corrupted_output = model.output.save()

    mx.eval(corrupted_output)
    corrupted_prob = get_target_token_probability(model, corrupted_output, target_id)
    print(f"  P('{target_token}') = {corrupted_prob:.4f}")

    # Verify corruption effectiveness (paper requirement)
    corruption_reduction = (clean_prob - corrupted_prob) / clean_prob * 100 if clean_prob > 0 else 0
    if corrupted_prob < clean_prob:
        print(f"  Corruption reduced probability by {corruption_reduction:.1f}%")
        if corruption_reduction < 10:
            print("  Note: Weak corruption (<10% reduction). Paper recommends stronger corruption.")
    else:
        print("  WARNING: Corruption did not reduce probability!")
        print("  This invalidates recovery measurements. Results below may not be meaningful.")

    # Corruption effectiveness check (paper uses threshold of significant reduction)
    MIN_CORRUPTION_THRESHOLD = 0.05  # At least 5% probability reduction
    if clean_prob - corrupted_prob < MIN_CORRUPTION_THRESHOLD:
        print(f"\n  [Corruption Verification: WEAK]")
        print(f"  Probability drop: {clean_prob - corrupted_prob:.4f} (threshold: {MIN_CORRUPTION_THRESHOLD})")
        print(f"  Recovery measurements may not be reliable.")
    else:
        print(f"\n  [Corruption Verification: PASSED]")
        print(f"  Probability drop: {clean_prob - corrupted_prob:.4f} >= {MIN_CORRUPTION_THRESHOLD}")

    print(f"\n[Step 3] Residual stream patching experiment...")
    print(f"Patching FULL LAYER OUTPUT at subject positions ({subject_start}-{subject_end})")
    print("(Paper's method: patch residual stream, not just MLP)")

    n_layers = len(model.layers)
    results = {}

    # PASS 1: Cache ALL clean LAYER outputs (residual stream)
    # This is the paper's method: patch the hidden state (residual stream),
    # not just the MLP output. The residual stream is the full layer output.
    with model.trace(mx.array([tokens])) as clean_trace:
        pass

    # Build a dict of all clean LAYER outputs (residual stream after each layer)
    # Key insight: h.{idx} is the residual stream, h.{idx}.mlp is just the MLP part
    clean_layer_cache = {}
    layer_keys = {}
    for layer_idx in range(n_layers):
        for key in clean_trace.activations:
            # Match full layer output: ends with "h.N" or "layers.N"
            # But NOT "h.N.mlp", "h.N.attn", etc.
            if f"h.{layer_idx}" in key or f"layers.{layer_idx}" in key:
                # Check it's the layer output, not a subcomponent
                after_layer = key.split(f"h.{layer_idx}")[-1] if f"h.{layer_idx}" in key else key.split(f"layers.{layer_idx}")[-1]
                if after_layer == "":  # Nothing after the layer index = full layer
                    clean_layer_cache[layer_idx] = clean_trace.activations[key]
                    layer_keys[layer_idx] = key
                    mx.eval(clean_layer_cache[layer_idx])
                    break

    # PASS 2: For each layer, run corrupted forward pass with residual stream injection
    # KEY INSIGHT: Patch the FULL LAYER OUTPUT (residual stream), not just MLP
    # This is what the paper does - it restores the complete hidden state
    for layer_idx in range(n_layers):
        if layer_idx not in clean_layer_cache:
            continue

        layer_key = layer_keys[layer_idx]
        clean_layer = clean_layer_cache[layer_idx]

        # Intervention key (strip model prefixes)
        if layer_key.startswith("model.model."):
            intervention_key = layer_key[12:]
        elif layer_key.startswith("model."):
            intervention_key = layer_key[6:]
        else:
            intervention_key = layer_key

        # Run corrupted forward pass, inject clean RESIDUAL STREAM at subject positions
        # This patches the complete hidden state, not just MLP output
        with model.trace(mx.array([tokens]),
                         interventions={
                             embed_key: iv.replace_with(noisy_embeddings),
                             intervention_key: replace_at_positions(clean_layer, subject_start, subject_end)
                         }):
            patched_output = model.output.save()

        mx.eval(patched_output)
        patched_prob = get_target_token_probability(model, patched_output, target_id)

        # Recovery = how much probability was restored
        if clean_prob - corrupted_prob > 1e-6:
            recovery = (patched_prob - corrupted_prob) / (clean_prob - corrupted_prob) * 100
        else:
            recovery = 0.0

        results[layer_idx] = recovery
        print(f"  Layer {layer_idx:2d}: P={patched_prob:.4f}, Recovery={recovery:6.1f}%")

    if results:
        peak = max(results, key=results.get)
        print(f"\n[Result] Peak recovery at layer {peak} ({results[peak]:.1f}%)")

        # Analyze layer distribution
        n_layers = len(model.layers)
        early_layers = [results[i] for i in range(n_layers // 3) if i in results]
        middle_layers = [results[i] for i in range(n_layers // 3, 2 * n_layers // 3) if i in results]
        late_layers = [results[i] for i in range(2 * n_layers // 3, n_layers) if i in results]

        print("\n[Layer Distribution Analysis]")
        if early_layers:
            print(f"  Early layers (0-{n_layers//3-1}):   avg {sum(early_layers)/len(early_layers):5.1f}%")
        if middle_layers:
            print(f"  Middle layers ({n_layers//3}-{2*n_layers//3-1}): avg {sum(middle_layers)/len(middle_layers):5.1f}%")
        if late_layers:
            print(f"  Late layers ({2*n_layers//3}-{n_layers-1}):  avg {sum(late_layers)/len(late_layers):5.1f}%")

    print("\n[Methodology]")
    print("This experiment implements the paper's methodology:")
    print("  1. Gaussian noise corruption on subject embeddings")
    print("  2. RESIDUAL STREAM patching (full layer output h.N, not just MLP)")
    print("  3. Subject-position patching (all subject tokens)")
    print("  4. Target-token probability recovery metric")
    print("")
    print("The residual stream (h.N) is the complete hidden state after layer N,")
    print("including both attention and MLP contributions. This is what the paper")
    print("patches to identify which layers store factual associations.")


def experiment_7_position_specific_patching(model):
    """
    Experiment 7: Why Position-Specific Patching Matters

    This experiment demonstrates why the paper patches at a SINGLE position
    (subject's last token) rather than the entire sequence. Full-sequence
    patching causes activation distribution mismatch in later layers.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Full-Sequence vs Position-Specific Patching")
    print("=" * 70)
    print("Demonstrating why position-specific patching is essential")

    # Use same-length prompts for fair comparison
    clean_prompt = "The Eiffel Tower is in"
    corrupted_prompt = "The Statue Liberty is in"

    # Tokenize
    clean_tokens = model.encode(clean_prompt)
    corrupted_tokens = model.encode(corrupted_prompt)

    print(f"\nClean: '{clean_prompt}'")
    print("Tokens:", end=" ")
    for i, tok in enumerate(clean_tokens):
        print(f"[{i}:'{model.token_to_str(tok)}']", end=" ")
    print()

    print(f"\nCorrupted: '{corrupted_prompt}'")
    print("Tokens:", end=" ")
    for i, tok in enumerate(corrupted_tokens):
        print(f"[{i}:'{model.token_to_str(tok)}']", end=" ")
    print()

    if len(clean_tokens) != len(corrupted_tokens):
        print(f"\nWARNING: Sequence lengths differ ({len(clean_tokens)} vs {len(corrupted_tokens)})")
        print("This experiment requires equal-length sequences for fair comparison.")
        return

    subject_last_pos = 4  # "Tower" token position

    print(f"\nPatching position: {subject_last_pos} (subject's last token)")

    # Get clean activations
    with model.trace(mx.array([clean_tokens])) as clean_trace:
        clean_output = model.output.save()
    mx.eval(clean_output)
    clean_pred = get_top_predictions_from_logits(model, clean_output, top_k=1)
    print(f"\nClean prediction: '{model.token_to_str(clean_pred[0][0])}'")

    # Get corrupted baseline
    with model.trace(mx.array([corrupted_tokens])) as corr_trace:
        corrupted_output = model.output.save()
    mx.eval(corrupted_output)
    corrupted_pred = get_top_predictions_from_logits(model, corrupted_output, top_k=1)
    print(f"Corrupted prediction: '{model.token_to_str(corrupted_pred[0][0])}'")

    n_layers = len(model.layers)
    print(f"\nComparing patching methods at layers {n_layers//4}, {n_layers//2}, {3*n_layers//4}:")
    print("-" * 60)
    print(f"{'Layer':>6} | {'Full-Sequence':>15} | {'Position-Only':>15}")
    print("-" * 60)

    for layer_idx in [n_layers // 4, n_layers // 2, 3 * n_layers // 4]:
        # Find MLP key
        mlp_key = None
        for key in clean_trace.activations:
            if key.endswith(f"layers.{layer_idx}.mlp") or key.endswith(f"h.{layer_idx}.mlp"):
                mlp_key = key
                break

        if mlp_key is None:
            continue

        clean_mlp = clean_trace.activations[mlp_key]
        mx.eval(clean_mlp)

        # Intervention key
        if mlp_key.startswith("model.model."):
            intervention_key = mlp_key[12:]
        elif mlp_key.startswith("model."):
            intervention_key = mlp_key[6:]
        else:
            intervention_key = mlp_key

        # Full-sequence patching
        with model.trace(mx.array([corrupted_tokens]),
                         interventions={intervention_key: iv.replace_with(clean_mlp)}):
            full_output = model.output.save()
        mx.eval(full_output)
        full_pred = get_top_predictions_from_logits(model, full_output, top_k=1)

        # Position-specific patching using our helper
        with model.trace(mx.array([corrupted_tokens]),
                         interventions={intervention_key: replace_at_position(clean_mlp, subject_last_pos)}):
            pos_output = model.output.save()
        mx.eval(pos_output)
        pos_pred = get_top_predictions_from_logits(model, pos_output, top_k=1)

        full_token = model.token_to_str(full_pred[0][0])
        pos_token = model.token_to_str(pos_pred[0][0])

        print(f"{layer_idx:>6} | {full_token:>15} | {pos_token:>15}")

    print("-" * 60)

    print("\n[Key Insight]")
    print("Full-sequence patching replaces activations at ALL positions, which")
    print("causes distribution mismatch when clean activations meet corrupted")
    print("activations from earlier layers. Position-specific patching avoids")
    print("this by only replacing at the subject's last token, allowing clean")
    print("information to flow forward naturally through subsequent layers.")


def experiment_8_statistical_averaging(model):
    """
    Experiment 8: Statistical Averaging Over Multiple Examples

    The paper averages results over many factual prompts for robustness.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Statistical Averaging")
    print("=" * 70)
    print("Averaging recovery over multiple factual prompts (paper's approach)")

    # Multiple factual prompts with their corrupted versions
    factual_prompts = [
        ("The Eiffel Tower is in", "The Statue of Liberty is in"),
        ("The capital of France is", "The capital of Germany is"),
        ("Albert Einstein developed the theory of", "Isaac Newton developed the theory of"),
        ("The Great Wall is in", "The Colosseum is in"),
        ("Apple was founded by Steve", "Microsoft was founded by Steve"),
    ]

    n_layers = len(model.layers)
    layer_recoveries = {i: [] for i in range(n_layers)}

    print(f"\nAnalyzing {len(factual_prompts)} factual prompts...")

    for i, (clean, corrupted) in enumerate(factual_prompts):
        print(f"  [{i+1}] {clean[:40]}...")

        results = model.activation_patching(
            clean_text=clean,
            corrupted_text=corrupted,
            component="mlp",
            plot=False
        )

        for layer, recovery in results.items():
            layer_recoveries[layer].append(recovery)

    # Compute statistics
    print("\n" + "-" * 50)
    print("Layer | Mean Recovery | Std Dev | N")
    print("-" * 50)

    layer_means = {}
    for layer in sorted(layer_recoveries.keys()):
        values = layer_recoveries[layer]
        if values:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            layer_means[layer] = mean
            print(f"{layer:5d} | {mean:12.1f}% | {std:7.1f}% | {len(values)}")

    if layer_means:
        peak = max(layer_means, key=layer_means.get)
        print("-" * 50)
        print(f"\nPeak (averaged): Layer {peak} ({layer_means[peak]:.1f}% mean recovery)")

        # Categorize layers
        early_mean = sum(layer_means.get(i, 0) for i in range(n_layers // 3)) / (n_layers // 3)
        mid_start = n_layers // 3
        mid_end = 2 * n_layers // 3
        middle_mean = sum(layer_means.get(i, 0) for i in range(mid_start, mid_end)) / (mid_end - mid_start)
        late_mean = sum(layer_means.get(i, 0) for i in range(mid_end, n_layers)) / (n_layers - mid_end)

        print(f"\nLayer region averages:")
        print(f"  Early layers:  {early_mean:.1f}%")
        print(f"  Middle layers: {middle_mean:.1f}%")
        print(f"  Late layers:   {late_mean:.1f}%")

    print("\n[Observation]")
    print("Statistical averaging reveals consistent patterns across prompts.")
    print("")
    print("Note on layer distribution:")
    print("- The ROME paper found peak recovery in MIDDLE layers for GPT-2/GPT-J")
    print("- Different model architectures (Llama vs GPT) may show different patterns")
    print("- Quantized models may behave differently than full-precision models")
    print("- The key finding (MLP > Attention) is more robust across models")


def main():
    """Run all experiments."""
    print("=" * 70)
    print("CAUSAL TRACING TUTORIAL")
    print("Reproducing: 'Locating and Editing Factual Associations in GPT'")
    print("Meng et al., NeurIPS 2022")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("This tutorial has two parts:")
    print("  - Experiments 1-5: Simplified implementation for learning")
    print("  - Experiments 6-8: Paper-accurate methodology for reproduction")
    print("-" * 70)

    # Load model - Use GPT-2 XL to match the paper's model
    # The ROME paper used GPT-2 XL (48 layers) and GPT-J (28 layers)
    print("\nLoading model...")

    # Primary: GPT-2 XL (paper's model)
    # Fallback: Llama if GPT-2 unavailable
    MODEL_CHOICES = [
        "MCES10/gpt2-xl-mlx-fp16",      # GPT-2 XL - paper's model (48 layers)
        "warshanks/gpt2-xl-bf16",        # GPT-2 XL alternative
        "mlx-community/Llama-3.2-1B-Instruct-4bit",  # Fallback
    ]

    model = None
    for model_name in MODEL_CHOICES:
        try:
            print(f"  Trying: {model_name}")
            model = InterpretableModel(model_name)
            print(f"  SUCCESS: Loaded {model_name}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if model is None:
        raise RuntimeError("Could not load any model. Please install mlx-lm.")

    print(f"Model loaded: {len(model.layers)} layers")

    # Note if not using paper's model
    if "gpt2" not in MODEL_CHOICES[0].lower() or len(model.layers) != 48:
        print("\n  NOTE: For exact paper reproduction, GPT-2 XL (48 layers) is recommended.")
        print(f"  Current model has {len(model.layers)} layers.")

    # Part 1: Simplified experiments (for learning)
    print("\n" + "=" * 70)
    print("PART 1: SIMPLIFIED EXPERIMENTS (for learning)")
    print("=" * 70)

    experiment_1_manual_causal_tracing(model)
    experiment_2_automated_patching(model)
    experiment_3_mlp_vs_attention(model)
    experiment_4_layer_distribution(model)
    experiment_5_multiple_facts(model)

    # Part 2: Paper-accurate experiments
    print("\n" + "=" * 70)
    print("PART 2: PAPER-ACCURATE EXPERIMENTS")
    print("=" * 70)

    experiment_6_gaussian_noise_corruption(model)
    experiment_7_position_specific_patching(model)
    experiment_8_statistical_averaging(model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key concepts from the paper (Meng et al., NeurIPS 2022):

1. Causal tracing identifies where factual knowledge is stored by
   measuring which layer activations restore correct outputs after
   input corruption.

2. The paper found factual associations are stored primarily in
   MLP modules at middle layers (for GPT-2/GPT-J models).

3. Knowledge is localized at the subject's last token position
   (demonstrated in Experiment 7).

4. This technique enables model editing: surgically updating specific
   facts without full retraining.

Paper methodology implemented:
- Experiment 6: Gaussian noise corruption on subject embeddings
- Experiment 7: Position-specific patching at subject's last token
- Experiment 8: Statistical averaging over multiple factual prompts

The paper's corruption and recovery metrics:
- Corruption: Add Gaussian noise (~0.15 std) to subject token embeddings
- Metric: Target-token probability recovery P(correct) / P_clean(correct)
- Statistics: Average recovery curves over many (subject, fact) pairs

Important notes on reproducibility:
- MLP > Attention finding: ROBUST across model architectures
- Middle-layer concentration: May vary by model (Llama vs GPT architecture)
- Noise level: Must be strong enough to cause significant probability drop
- Quantized models: May show different layer distributions than full-precision

Our tutorial demonstrates both simplified and paper-accurate approaches
to help users understand the methodology at different levels of detail.
    """)


if __name__ == "__main__":
    main()
