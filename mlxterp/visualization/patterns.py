"""
Attention Pattern Detection Module.

Provides utilities for detecting and classifying attention head types,
including induction heads, previous token heads, copying heads, etc.

Based on research from:
- "In-context Learning and Induction Heads" (Olsson et al., 2022)
- "Interpretability in the Wild" (Wang et al., 2022)

Example:
    >>> from mlxterp import InterpretableModel
    >>> from mlxterp.visualization import detect_induction_heads, detect_head_types
    >>>
    >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
    >>>
    >>> # Detect induction heads
    >>> induction_heads = detect_induction_heads(model, threshold=0.4)
    >>> print(f"Found {len(induction_heads)} induction heads")
    >>>
    >>> # Classify all head types
    >>> head_types = detect_head_types(model, "The quick brown fox")
"""

import mlx.core as mx
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass


@dataclass
class HeadScore:
    """Score for an attention head."""
    layer: int
    head: int
    score: float
    head_type: Optional[str] = None


def induction_score(
    attention_pattern: np.ndarray,
    seq_len: int,
) -> float:
    """
    Compute induction head score for an attention pattern.

    Induction heads attend from position i to position (i - seq_len + 1)
    in repeated sequences, implementing the pattern:
    [A][B]...[A] -> predicts [B]

    Args:
        attention_pattern: Attention weights of shape (seq_q, seq_k)
                          Should be from a repeated sequence
        seq_len: Length of the repeated subsequence

    Returns:
        Induction score (higher = more induction-like)

    Note:
        For accurate detection, use repeated random tokens:
        tokens = random_tokens + random_tokens (repeated twice)
    """
    total_len = attention_pattern.shape[0]

    if total_len < seq_len:
        return 0.0

    # Induction pattern: position i attends to position (i - seq_len + 1)
    # This is the diagonal with offset = 1 - seq_len
    scores = []
    for i in range(seq_len, total_len):
        src_pos = i - seq_len + 1
        if 0 <= src_pos < total_len:
            scores.append(attention_pattern[i, src_pos])

    if not scores:
        return 0.0

    return float(np.mean(scores))


def previous_token_score(attention_pattern: np.ndarray) -> float:
    """
    Compute previous token head score.

    Previous token heads attend strongly to position (i - 1),
    implementing the pattern: attend to the token immediately before.

    Args:
        attention_pattern: Attention weights of shape (seq_q, seq_k)

    Returns:
        Previous token score (higher = attends more to previous position)
    """
    seq_len = attention_pattern.shape[0]

    if seq_len < 2:
        return 0.0

    # Extract the -1 diagonal (position i attending to i-1)
    scores = []
    for i in range(1, seq_len):
        scores.append(attention_pattern[i, i - 1])

    return float(np.mean(scores))


def first_token_score(attention_pattern: np.ndarray) -> float:
    """
    Compute first token head score.

    First token heads attend strongly to position 0 (BOS token or first token).

    Args:
        attention_pattern: Attention weights of shape (seq_q, seq_k)

    Returns:
        First token score (higher = attends more to position 0)
    """
    # Mean attention to position 0 across all query positions
    return float(np.mean(attention_pattern[:, 0]))


def current_token_score(attention_pattern: np.ndarray) -> float:
    """
    Compute current token head score.

    Current token heads attend strongly to themselves (main diagonal).

    Args:
        attention_pattern: Attention weights of shape (seq_q, seq_k)

    Returns:
        Current token score (higher = attends more to self)
    """
    seq_len = attention_pattern.shape[0]

    # Extract main diagonal
    scores = [attention_pattern[i, i] for i in range(seq_len)]

    return float(np.mean(scores))


def copying_score(
    ov_circuit: np.ndarray,
    unembedding: Optional[np.ndarray] = None,
) -> float:
    """
    Compute copying head score from OV circuit.

    Copying heads increase the logit of the attended-to token.
    High copying score if OV circuit has strong diagonal.

    Args:
        ov_circuit: The OV circuit matrix W_V @ W_O of shape (d_model, d_model)
        unembedding: Optional unembedding matrix W_U of shape (d_model, vocab_size)

    Returns:
        Copying score (higher = more copying behavior)

    Note:
        Full analysis requires: W_U.T @ W_V @ W_O @ W_U
        Simplified version just looks at OV circuit diagonal dominance.
    """
    if unembedding is not None:
        # Full copying analysis
        full_circuit = unembedding.T @ ov_circuit @ unembedding
        diagonal = np.diag(full_circuit)
        return float(np.mean(diagonal) / (np.mean(np.abs(full_circuit)) + 1e-8))
    else:
        # Simplified: diagonal dominance of OV circuit
        diagonal = np.diag(ov_circuit)
        return float(np.mean(diagonal) / (np.mean(np.abs(ov_circuit)) + 1e-8))


class AttentionPatternDetector:
    """
    Detector for classifying attention head types.

    Computes various pattern scores for attention heads and classifies them
    into types like induction, previous token, first token, etc.

    Example:
        >>> detector = AttentionPatternDetector()
        >>>
        >>> with model.trace("Hello world") as trace:
        >>>     pass
        >>>
        >>> patterns = get_attention_patterns(trace)
        >>> scores = detector.analyze_head(patterns[5][0, 3])  # Layer 5, Head 3
        >>> print(scores)
    """

    def __init__(
        self,
        induction_threshold: float = 0.4,
        previous_token_threshold: float = 0.5,
        first_token_threshold: float = 0.3,
        current_token_threshold: float = 0.3,
    ):
        """
        Initialize detector with score thresholds.

        Args:
            induction_threshold: Threshold for classifying as induction head
            previous_token_threshold: Threshold for previous token head
            first_token_threshold: Threshold for first token head
            current_token_threshold: Threshold for current token head
        """
        self.induction_threshold = induction_threshold
        self.previous_token_threshold = previous_token_threshold
        self.first_token_threshold = first_token_threshold
        self.current_token_threshold = current_token_threshold

    def analyze_head(
        self,
        attention_pattern: np.ndarray,
        seq_len_for_induction: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute all pattern scores for a single attention head.

        Args:
            attention_pattern: Attention weights of shape (seq_q, seq_k)
            seq_len_for_induction: Subsequence length for induction scoring
                                   (required if checking induction)

        Returns:
            Dict of pattern type -> score
        """
        scores = {
            "previous_token": previous_token_score(attention_pattern),
            "first_token": first_token_score(attention_pattern),
            "current_token": current_token_score(attention_pattern),
        }

        if seq_len_for_induction is not None:
            scores["induction"] = induction_score(
                attention_pattern, seq_len_for_induction
            )

        return scores

    def classify_head(
        self,
        attention_pattern: np.ndarray,
        seq_len_for_induction: Optional[int] = None,
    ) -> List[str]:
        """
        Classify an attention head into one or more types.

        Args:
            attention_pattern: Attention weights of shape (seq_q, seq_k)
            seq_len_for_induction: Subsequence length for induction scoring

        Returns:
            List of head type labels that exceed thresholds
        """
        scores = self.analyze_head(attention_pattern, seq_len_for_induction)
        types = []

        if scores["previous_token"] > self.previous_token_threshold:
            types.append("previous_token")
        if scores["first_token"] > self.first_token_threshold:
            types.append("first_token")
        if scores["current_token"] > self.current_token_threshold:
            types.append("current_token")
        if "induction" in scores and scores["induction"] > self.induction_threshold:
            types.append("induction")

        return types if types else ["unknown"]


def detect_head_types(
    model,
    text: str,
    threshold: float = 0.4,
    layers: Optional[List[int]] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Detect attention head types across a model.

    Args:
        model: InterpretableModel instance
        text: Input text for analysis
        threshold: Score threshold for classification
        layers: Specific layers to analyze (None for all)

    Returns:
        Dict mapping head type to list of (layer, head) tuples

    Example:
        >>> head_types = detect_head_types(model, "The quick brown fox")
        >>> print(f"Previous token heads: {head_types['previous_token']}")
    """
    from .attention import get_attention_patterns

    with model.trace(text) as trace:
        pass

    patterns = get_attention_patterns(trace, layers)

    detector = AttentionPatternDetector(
        previous_token_threshold=threshold,
        first_token_threshold=threshold,
        current_token_threshold=threshold,
    )

    results = {
        "previous_token": [],
        "first_token": [],
        "current_token": [],
        "unknown": [],
    }

    for layer_idx, attn in patterns.items():
        num_heads = attn.shape[1]
        for head_idx in range(num_heads):
            head_pattern = attn[0, head_idx]  # First batch
            types = detector.classify_head(head_pattern)

            for head_type in types:
                if head_type not in results:
                    results[head_type] = []
                results[head_type].append((layer_idx, head_idx))

    return results


def detect_induction_heads(
    model,
    n_random_tokens: int = 50,
    n_repeats: int = 2,
    threshold: float = 0.4,
    layers: Optional[List[int]] = None,
    seed: int = 42,
) -> List[HeadScore]:
    """
    Detect induction heads using repeated random token sequences.

    Induction heads implement the pattern: [A][B]...[A] -> predict [B]
    They are detected by measuring attention to the token after the
    previous occurrence of the current token.

    Args:
        model: InterpretableModel instance
        n_random_tokens: Number of random tokens in subsequence
        n_repeats: Number of times to repeat the subsequence
        threshold: Score threshold for detection
        layers: Specific layers to analyze (None for all)
        seed: Random seed for reproducibility

    Returns:
        List of HeadScore objects for heads above threshold,
        sorted by score descending

    Example:
        >>> induction_heads = detect_induction_heads(model, threshold=0.3)
        >>> for head in induction_heads[:5]:
        >>>     print(f"L{head.layer}H{head.head}: {head.score:.3f}")
    """
    from .attention import get_attention_patterns

    # Generate random tokens (avoiding special tokens)
    np.random.seed(seed)

    vocab_size = model.tokenizer.vocab_size if hasattr(model.tokenizer, 'vocab_size') else 32000
    # Use middle range of vocabulary to avoid special tokens
    min_token = int(vocab_size * 0.1)
    max_token = int(vocab_size * 0.9)

    random_tokens = np.random.randint(min_token, max_token, size=n_random_tokens)
    repeated_tokens = np.tile(random_tokens, n_repeats)

    # Convert to token IDs for the model
    token_ids = mx.array([repeated_tokens.tolist()])

    # Run trace with token IDs directly
    with model.trace(token_ids) as trace:
        pass

    patterns = get_attention_patterns(trace, layers)

    results = []

    for layer_idx, attn in patterns.items():
        num_heads = attn.shape[1]
        for head_idx in range(num_heads):
            head_pattern = attn[0, head_idx]
            score = induction_score(head_pattern, n_random_tokens)

            if score > threshold:
                results.append(HeadScore(
                    layer=layer_idx,
                    head=head_idx,
                    score=score,
                    head_type="induction",
                ))

    # Sort by score descending
    results.sort(key=lambda x: -x.score)

    return results


def find_attention_pattern(
    model,
    text: str,
    pattern_fn: Callable[[np.ndarray], float],
    threshold: float = 0.5,
    layers: Optional[List[int]] = None,
) -> List[HeadScore]:
    """
    Find attention heads matching a custom pattern function.

    This is a flexible utility for defining custom head detection criteria.

    Args:
        model: InterpretableModel instance
        text: Input text for analysis
        pattern_fn: Function that takes attention pattern (seq_q, seq_k)
                   and returns a score
        threshold: Score threshold for detection
        layers: Specific layers to analyze

    Returns:
        List of HeadScore objects for heads above threshold

    Example:
        >>> # Find heads that attend heavily to position 2
        >>> def attend_pos_2(pattern):
        >>>     return np.mean(pattern[:, 2])
        >>>
        >>> heads = find_attention_pattern(model, text, attend_pos_2, threshold=0.3)
    """
    from .attention import get_attention_patterns

    with model.trace(text) as trace:
        pass

    patterns = get_attention_patterns(trace, layers)

    results = []

    for layer_idx, attn in patterns.items():
        num_heads = attn.shape[1]
        for head_idx in range(num_heads):
            head_pattern = attn[0, head_idx]
            score = pattern_fn(head_pattern)

            if score > threshold:
                results.append(HeadScore(
                    layer=layer_idx,
                    head=head_idx,
                    score=score,
                    head_type="custom",
                ))

    results.sort(key=lambda x: -x.score)
    return results
