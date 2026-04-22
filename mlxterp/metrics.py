"""
Standard metrics for causal interpretability experiments.

All metrics follow a uniform signature for use with activation patching,
attribution patching, and causal trace APIs.
"""

import mlx.core as mx
from typing import Optional


def logit_diff(
    patched_logits: mx.array,
    clean_logits: mx.array,
    corrupted_logits: mx.array,
    correct_token: int,
    incorrect_token: int,
    **kwargs,
) -> float:
    """
    Logit difference metric: difference in logit between correct and incorrect tokens.

    Measures how much patching recovers the clean logit difference.
    Standard metric for IOI and factual recall tasks.

    Args:
        patched_logits: Logits after patching intervention (batch, vocab)
        clean_logits: Logits from clean forward pass (batch, vocab)
        corrupted_logits: Logits from corrupted forward pass (batch, vocab)
        correct_token: Token ID for the correct/expected completion
        incorrect_token: Token ID for the incorrect/counterfactual completion

    Returns:
        Normalized recovery: 1.0 = fully recovered clean behavior,
        0.0 = no change from corrupted, <0 = made worse
    """
    clean_diff = float(clean_logits[0, correct_token] - clean_logits[0, incorrect_token])
    corrupted_diff = float(corrupted_logits[0, correct_token] - corrupted_logits[0, incorrect_token])
    patched_diff = float(patched_logits[0, correct_token] - patched_logits[0, incorrect_token])

    baseline = clean_diff - corrupted_diff
    if abs(baseline) < 1e-10:
        return 0.0
    return (patched_diff - corrupted_diff) / baseline


def kl_divergence(
    patched_logits: mx.array,
    clean_logits: mx.array,
    corrupted_logits: mx.array,
    **kwargs,
) -> float:
    """
    KL divergence between patched and clean output distributions.

    Lower values mean the patched output is more similar to clean.
    Returns negative KL so that higher = better recovery (consistent with other metrics).

    Args:
        patched_logits: Logits after patching (batch, vocab) or (vocab,)
        clean_logits: Logits from clean pass (batch, vocab) or (vocab,)
        corrupted_logits: Not directly used, but included for uniform signature.

    Returns:
        Negative KL(clean || patched). Higher (closer to 0) = better recovery.
    """
    # Handle batched or unbatched input
    p_logits = clean_logits.astype(mx.float32)
    q_logits = patched_logits.astype(mx.float32)

    if p_logits.ndim == 2:
        p_logits = p_logits[0]
        q_logits = q_logits[0]

    # Compute log-softmax for numerical stability
    p_log = p_logits - mx.logsumexp(p_logits, keepdims=True)
    q_log = q_logits - mx.logsumexp(q_logits, keepdims=True)

    p = mx.softmax(p_logits)
    kl = float(mx.sum(p * (p_log - q_log)))
    return -kl


def cross_entropy_diff(
    patched_logits: mx.array,
    clean_logits: mx.array,
    corrupted_logits: mx.array,
    target_token: Optional[int] = None,
    **kwargs,
) -> float:
    """
    Difference in cross-entropy loss between corrupted and patched outputs.

    Positive values mean patching reduced the loss (recovered clean behavior).

    Args:
        patched_logits: Logits after patching (batch, vocab) or (vocab,)
        clean_logits: Logits from clean pass (used to determine target if not provided)
        corrupted_logits: Logits from corrupted pass (batch, vocab) or (vocab,)
        target_token: Token to compute CE against. If None, uses argmax of clean.

    Returns:
        CE(corrupted) - CE(patched). Positive = patching helped.
    """
    p_logits = patched_logits.astype(mx.float32)
    c_logits = corrupted_logits.astype(mx.float32)
    cl_logits = clean_logits.astype(mx.float32)

    if p_logits.ndim == 2:
        p_logits = p_logits[0]
        c_logits = c_logits[0]
        cl_logits = cl_logits[0]

    if target_token is None:
        target_token = int(mx.argmax(cl_logits))

    # Log-softmax
    p_log = p_logits - mx.logsumexp(p_logits, keepdims=True)
    c_log = c_logits - mx.logsumexp(c_logits, keepdims=True)

    ce_patched = -float(p_log[target_token])
    ce_corrupted = -float(c_log[target_token])

    return ce_corrupted - ce_patched


def l2_distance(
    patched_logits: mx.array,
    clean_logits: mx.array,
    corrupted_logits: mx.array,
    **kwargs,
) -> float:
    """
    Normalized L2 recovery: how much of the clean-corrupted distance was recovered.

    Args:
        patched_logits: Logits after patching
        clean_logits: Logits from clean pass
        corrupted_logits: Logits from corrupted pass

    Returns:
        Recovery fraction: 1.0 = fully recovered, 0.0 = no change
    """
    p = patched_logits.astype(mx.float32)
    cl = clean_logits.astype(mx.float32)
    co = corrupted_logits.astype(mx.float32)

    if p.ndim == 2:
        p, cl, co = p[0], cl[0], co[0]

    baseline_dist = float(mx.sqrt(mx.sum((cl - co) ** 2)))
    if baseline_dist < 1e-10:
        return 0.0

    patched_dist = float(mx.sqrt(mx.sum((cl - p) ** 2)))
    return 1.0 - patched_dist / baseline_dist


def cosine_distance(
    patched_logits: mx.array,
    clean_logits: mx.array,
    corrupted_logits: mx.array,
    **kwargs,
) -> float:
    """
    Cosine similarity recovery between patched and clean outputs.

    Args:
        patched_logits: Logits after patching
        clean_logits: Logits from clean pass
        corrupted_logits: Logits from corrupted pass

    Returns:
        Recovery: 1.0 = patched matches clean, 0.0 = no change from corrupted
    """
    p = patched_logits.astype(mx.float32)
    cl = clean_logits.astype(mx.float32)
    co = corrupted_logits.astype(mx.float32)

    if p.ndim == 2:
        p, cl, co = p[0], cl[0], co[0]

    def _cosine_sim(a, b):
        dot = float(mx.sum(a * b))
        norm_a = float(mx.sqrt(mx.sum(a * a)))
        norm_b = float(mx.sqrt(mx.sum(b * b)))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    sim_clean_corrupted = _cosine_sim(cl, co)
    sim_clean_patched = _cosine_sim(cl, p)

    baseline = 1.0 - sim_clean_corrupted
    if abs(baseline) < 1e-10:
        return 0.0

    return (sim_clean_patched - sim_clean_corrupted) / baseline


# Registry for string-based metric lookup
METRICS = {
    "logit_diff": logit_diff,
    "kl_divergence": kl_divergence,
    "kl": kl_divergence,
    "cross_entropy_diff": cross_entropy_diff,
    "ce_diff": cross_entropy_diff,
    "l2": l2_distance,
    "l2_distance": l2_distance,
    "cosine": cosine_distance,
    "cosine_distance": cosine_distance,
}


def get_metric(name):
    """Get a metric function by name. Returns the function if already callable."""
    if callable(name):
        return name
    if name not in METRICS:
        raise ValueError(
            f"Unknown metric: {name}. Available: {list(METRICS.keys())}"
        )
    return METRICS[name]
