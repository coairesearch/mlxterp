"""
Test Visualization Module - Verify attention visualization and pattern detection.

Tests the visualization module including:
- Attention pattern extraction
- Attention heatmap generation
- Pattern detection (induction, previous token, etc.)
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlxterp import InterpretableModel
from mlxterp.visualization import (
    get_attention_patterns,
    attention_heatmap,
    attention_from_trace,
    AttentionVisualizationConfig,
    AttentionPatternDetector,
    detect_head_types,
    induction_score,
    previous_token_score,
    first_token_score,
)


def test_get_attention_patterns():
    """Test extracting attention patterns from trace."""
    print("\n" + "=" * 60)
    print("TEST: Get attention patterns")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Hello world"

    with model.trace(text) as trace:
        pass

    # Get patterns for all layers
    patterns = get_attention_patterns(trace)

    print(f"  Found patterns for {len(patterns)} layers")

    assert len(patterns) > 0, "No attention patterns extracted"

    # Check shape
    first_layer = min(patterns.keys())
    pattern = patterns[first_layer]
    print(f"  Layer {first_layer} shape: {pattern.shape}")

    # Should be (batch, heads, seq_len, seq_len)
    assert len(pattern.shape) == 4, f"Expected 4D, got {len(pattern.shape)}D"

    print("PASSED: Attention patterns extracted correctly!")
    return True


def test_get_attention_patterns_specific_layers():
    """Test extracting patterns for specific layers."""
    print("\n" + "=" * 60)
    print("TEST: Get attention patterns for specific layers")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Test"

    with model.trace(text) as trace:
        pass

    # Get patterns for specific layers
    patterns = get_attention_patterns(trace, layers=[0, 5, 10])

    print(f"  Requested layers [0, 5, 10], got: {list(patterns.keys())}")

    assert set(patterns.keys()) == {0, 5, 10}, "Wrong layers returned"

    print("PASSED: Specific layer extraction works!")
    return True


def test_attention_heatmap_matplotlib():
    """Test attention heatmap with matplotlib backend."""
    print("\n" + "=" * 60)
    print("TEST: Attention heatmap (matplotlib)")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The cat"

    with model.trace(text) as trace:
        pass

    patterns = get_attention_patterns(trace, layers=[5])
    tokens = model.to_str_tokens(text)

    print(f"  Tokens: {tokens}")
    print(f"  Pattern shape: {patterns[5].shape}")

    # Create heatmap
    fig = attention_heatmap(
        patterns[5],
        tokens,
        head_idx=0,
        title="Layer 5, Head 0",
        backend="matplotlib"
    )

    assert fig is not None, "No figure returned"
    print(f"  Figure type: {type(fig)}")

    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)

    print("PASSED: Matplotlib heatmap works!")
    return True


def test_attention_from_trace_single():
    """Test attention_from_trace with single mode."""
    print("\n" + "=" * 60)
    print("TEST: attention_from_trace (single mode)")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Hello"

    with model.trace(text) as trace:
        pass

    tokens = model.to_str_tokens(text)

    # Create visualization
    config = AttentionVisualizationConfig(backend="matplotlib")
    result = attention_from_trace(
        trace, tokens,
        layers=[0],
        mode="single",
        config=config
    )

    assert result is not None, "No visualization returned"

    # Clean up
    import matplotlib.pyplot as plt
    plt.close('all')

    print("PASSED: attention_from_trace single mode works!")
    return True


def test_attention_from_trace_grid():
    """Test attention_from_trace with grid mode."""
    print("\n" + "=" * 60)
    print("TEST: attention_from_trace (grid mode)")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Hi"

    with model.trace(text) as trace:
        pass

    tokens = model.to_str_tokens(text)

    # Create grid visualization
    config = AttentionVisualizationConfig(backend="matplotlib")
    result = attention_from_trace(
        trace, tokens,
        layers=[0],
        mode="grid",
        config=config
    )

    assert result is not None, "No visualization returned"

    # Clean up
    import matplotlib.pyplot as plt
    plt.close('all')

    print("PASSED: attention_from_trace grid mode works!")
    return True


def test_to_str_tokens():
    """Test to_str_tokens method."""
    print("\n" + "=" * 60)
    print("TEST: to_str_tokens")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Hello world"
    tokens = model.to_str_tokens(text)

    print(f"  Input: '{text}'")
    print(f"  Tokens: {tokens}")

    assert len(tokens) > 0, "No tokens returned"
    assert isinstance(tokens[0], str), "Tokens should be strings"

    # Test with token IDs
    token_ids = model.encode(text)
    tokens_from_ids = model.to_str_tokens(token_ids)

    print(f"  Token IDs: {token_ids}")
    print(f"  Tokens from IDs: {tokens_from_ids}")

    assert tokens == tokens_from_ids, "Token conversion mismatch"

    print("PASSED: to_str_tokens works!")
    return True


def test_induction_score():
    """Test induction score calculation."""
    print("\n" + "=" * 60)
    print("TEST: Induction score")
    print("=" * 60)

    # Create a synthetic attention pattern with induction behavior
    # In a repeated sequence, induction heads attend to i - seq_len + 1
    seq_len = 5
    total_len = 10  # Two repetitions

    # Create pattern where positions 5-9 attend to positions 1-5
    pattern = np.zeros((total_len, total_len))
    for i in range(seq_len, total_len):
        # Induction pattern: attend to token after previous occurrence
        pattern[i, i - seq_len + 1] = 1.0

    score = induction_score(pattern, seq_len)
    print(f"  Synthetic induction pattern score: {score:.3f}")

    assert score > 0.9, f"Expected high induction score, got {score}"

    # Test with random pattern (should have low score)
    random_pattern = np.random.rand(total_len, total_len)
    random_pattern = random_pattern / random_pattern.sum(axis=-1, keepdims=True)
    random_score = induction_score(random_pattern, seq_len)

    print(f"  Random pattern score: {random_score:.3f}")

    assert random_score < 0.3, f"Random pattern should have low score, got {random_score}"

    print("PASSED: Induction score works!")
    return True


def test_previous_token_score():
    """Test previous token score calculation."""
    print("\n" + "=" * 60)
    print("TEST: Previous token score")
    print("=" * 60)

    seq_len = 5

    # Create pattern with previous token attention
    pattern = np.zeros((seq_len, seq_len))
    for i in range(1, seq_len):
        pattern[i, i - 1] = 1.0

    score = previous_token_score(pattern)
    print(f"  Synthetic previous token pattern score: {score:.3f}")

    assert score > 0.9, f"Expected high score, got {score}"

    # Test with random pattern
    random_pattern = np.random.rand(seq_len, seq_len)
    random_pattern = random_pattern / random_pattern.sum(axis=-1, keepdims=True)
    random_score = previous_token_score(random_pattern)

    print(f"  Random pattern score: {random_score:.3f}")

    print("PASSED: Previous token score works!")
    return True


def test_first_token_score():
    """Test first token score calculation."""
    print("\n" + "=" * 60)
    print("TEST: First token score")
    print("=" * 60)

    seq_len = 5

    # Create pattern with first token attention
    pattern = np.zeros((seq_len, seq_len))
    pattern[:, 0] = 1.0  # All positions attend to first

    score = first_token_score(pattern)
    print(f"  Synthetic first token pattern score: {score:.3f}")

    assert score > 0.9, f"Expected high score, got {score}"

    # Test with random pattern
    random_pattern = np.random.rand(seq_len, seq_len)
    random_pattern = random_pattern / random_pattern.sum(axis=-1, keepdims=True)
    random_score = first_token_score(random_pattern)

    print(f"  Random pattern score: {random_score:.3f}")

    print("PASSED: First token score works!")
    return True


def test_attention_pattern_detector():
    """Test AttentionPatternDetector class."""
    print("\n" + "=" * 60)
    print("TEST: AttentionPatternDetector")
    print("=" * 60)

    detector = AttentionPatternDetector(
        previous_token_threshold=0.5,
        first_token_threshold=0.3,
    )

    seq_len = 5

    # Test previous token pattern
    prev_pattern = np.zeros((seq_len, seq_len))
    for i in range(1, seq_len):
        prev_pattern[i, i - 1] = 1.0

    scores = detector.analyze_head(prev_pattern)
    print(f"  Previous token pattern scores: {scores}")

    types = detector.classify_head(prev_pattern)
    print(f"  Classification: {types}")

    assert "previous_token" in types, "Should classify as previous_token"

    # Test first token pattern
    first_pattern = np.zeros((seq_len, seq_len))
    first_pattern[:, 0] = 1.0

    types = detector.classify_head(first_pattern)
    print(f"  First token pattern classification: {types}")

    assert "first_token" in types, "Should classify as first_token"

    print("PASSED: AttentionPatternDetector works!")
    return True


def test_detect_head_types():
    """Test detect_head_types function with real model."""
    print("\n" + "=" * 60)
    print("TEST: Detect head types (real model)")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The quick brown fox jumps over"

    head_types = detect_head_types(model, text, threshold=0.3, layers=[0, 5, 10])

    print(f"  Head types found:")
    for head_type, heads in head_types.items():
        if heads:
            print(f"    {head_type}: {len(heads)} heads")
            if len(heads) <= 3:
                print(f"      {heads}")

    # Should find at least some heads
    total_heads = sum(len(h) for h in head_types.values())
    print(f"  Total classified heads: {total_heads}")

    assert total_heads > 0, "Should classify at least some heads"

    print("PASSED: detect_head_types works!")
    return True


def test_visualization_config():
    """Test AttentionVisualizationConfig."""
    print("\n" + "=" * 60)
    print("TEST: AttentionVisualizationConfig")
    print("=" * 60)

    # Default config
    config = AttentionVisualizationConfig()
    print(f"  Default colorscale: {config.colorscale}")
    print(f"  Default mask_upper_tri: {config.mask_upper_tri}")
    print(f"  Default backend: {config.backend}")

    assert config.colorscale == "Blues"
    assert config.mask_upper_tri is True
    assert config.backend == "auto"

    # Custom config
    custom_config = AttentionVisualizationConfig(
        colorscale="Viridis",
        mask_upper_tri=False,
        backend="matplotlib"
    )

    assert custom_config.colorscale == "Viridis"
    assert custom_config.mask_upper_tri is False
    assert custom_config.backend == "matplotlib"

    print("PASSED: AttentionVisualizationConfig works!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VISUALIZATION MODULE TEST SUITE")
    print("Testing attention visualization and pattern detection")
    print("=" * 70)

    tests = [
        test_get_attention_patterns,
        test_get_attention_patterns_specific_layers,
        test_to_str_tokens,
        test_visualization_config,
        test_induction_score,
        test_previous_token_score,
        test_first_token_score,
        test_attention_pattern_detector,
        test_attention_heatmap_matplotlib,
        test_attention_from_trace_single,
        test_attention_from_trace_grid,
        test_detect_head_types,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("ALL VISUALIZATION TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - please check output above")
        exit(1)
