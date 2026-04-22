"""Tests for mlxterp.metrics module."""

import pytest
import mlx.core as mx
from mlxterp.metrics import (
    logit_diff,
    kl_divergence,
    cross_entropy_diff,
    l2_distance,
    cosine_distance,
    get_metric,
    METRICS,
)


class TestLogitDiff:
    """Tests for logit_diff metric."""

    def test_perfect_recovery(self):
        """Patched == clean should give recovery = 1.0."""
        clean = mx.array([[0.0, 5.0, 1.0]])
        corrupted = mx.array([[0.0, 1.0, 5.0]])
        patched = mx.array([[0.0, 5.0, 1.0]])  # same as clean
        result = logit_diff(patched, clean, corrupted, correct_token=1, incorrect_token=2)
        assert abs(result - 1.0) < 1e-5

    def test_no_recovery(self):
        """Patched == corrupted should give recovery = 0.0."""
        clean = mx.array([[0.0, 5.0, 1.0]])
        corrupted = mx.array([[0.0, 1.0, 5.0]])
        patched = mx.array([[0.0, 1.0, 5.0]])  # same as corrupted
        result = logit_diff(patched, clean, corrupted, correct_token=1, incorrect_token=2)
        assert abs(result - 0.0) < 1e-5

    def test_partial_recovery(self):
        """Patched halfway between clean and corrupted."""
        clean = mx.array([[0.0, 6.0, 2.0]])     # diff = 4
        corrupted = mx.array([[0.0, 2.0, 6.0]])  # diff = -4
        patched = mx.array([[0.0, 4.0, 4.0]])    # diff = 0, halfway
        result = logit_diff(patched, clean, corrupted, correct_token=1, incorrect_token=2)
        assert abs(result - 0.5) < 1e-5

    def test_identical_clean_corrupted(self):
        """When clean == corrupted, baseline is 0, should return 0."""
        same = mx.array([[1.0, 2.0, 3.0]])
        result = logit_diff(same, same, same, correct_token=1, incorrect_token=2)
        assert result == 0.0


class TestKLDivergence:
    """Tests for kl_divergence metric."""

    def test_identical_distributions(self):
        """KL(p||p) = 0, so result should be 0."""
        logits = mx.array([[1.0, 2.0, 3.0]])
        result = kl_divergence(logits, logits, logits)
        assert abs(result) < 1e-4

    def test_different_distributions(self):
        """Different distributions should give negative KL."""
        clean = mx.array([[10.0, 0.0, 0.0]])  # peaked at 0
        patched = mx.array([[0.0, 10.0, 0.0]])  # peaked at 1
        corrupted = mx.array([[0.0, 0.0, 10.0]])
        result = kl_divergence(patched, clean, corrupted)
        assert result < 0  # KL > 0 means negative return

    def test_unbatched_input(self):
        """Should handle 1D input."""
        logits = mx.array([1.0, 2.0, 3.0])
        result = kl_divergence(logits, logits, logits)
        assert abs(result) < 1e-4


class TestCrossEntropyDiff:
    """Tests for cross_entropy_diff metric."""

    def test_recovery_positive(self):
        """Patching that moves toward correct token should be positive."""
        clean = mx.array([[10.0, 0.0, 0.0]])  # correct = token 0
        corrupted = mx.array([[0.0, 10.0, 0.0]])  # wrong
        patched = mx.array([[8.0, 1.0, 1.0]])  # mostly recovered
        result = cross_entropy_diff(patched, clean, corrupted, target_token=0)
        assert result > 0

    def test_no_recovery(self):
        """Patched == corrupted: CE diff should be ~0."""
        clean = mx.array([[10.0, 0.0]])
        corrupted = mx.array([[0.0, 10.0]])
        result = cross_entropy_diff(corrupted, clean, corrupted, target_token=0)
        assert abs(result) < 1e-5

    def test_auto_target_token(self):
        """Should auto-detect target from clean argmax."""
        clean = mx.array([[0.0, 10.0, 0.0]])  # argmax = 1
        corrupted = mx.array([[10.0, 0.0, 0.0]])
        patched = mx.array([[0.0, 10.0, 0.0]])
        result = cross_entropy_diff(patched, clean, corrupted)
        assert result > 0


class TestL2Distance:
    """Tests for l2_distance metric."""

    def test_perfect_recovery(self):
        """Patched == clean should give 1.0."""
        clean = mx.array([[1.0, 0.0, 0.0]])
        corrupted = mx.array([[0.0, 1.0, 0.0]])
        result = l2_distance(clean, clean, corrupted)
        assert abs(result - 1.0) < 1e-5

    def test_no_recovery(self):
        """Patched == corrupted should give 0.0."""
        clean = mx.array([[1.0, 0.0, 0.0]])
        corrupted = mx.array([[0.0, 1.0, 0.0]])
        result = l2_distance(corrupted, clean, corrupted)
        assert abs(result - 0.0) < 1e-5

    def test_identical_inputs(self):
        """All same: baseline = 0, return 0."""
        same = mx.array([[1.0, 2.0, 3.0]])
        assert l2_distance(same, same, same) == 0.0


class TestCosineDistance:
    """Tests for cosine_distance metric."""

    def test_perfect_recovery(self):
        """Patched == clean should give 1.0."""
        clean = mx.array([[1.0, 0.0]])
        corrupted = mx.array([[0.0, 1.0]])
        result = cosine_distance(clean, clean, corrupted)
        assert abs(result - 1.0) < 1e-4

    def test_no_recovery(self):
        """Patched == corrupted should give 0.0."""
        clean = mx.array([[1.0, 0.0]])
        corrupted = mx.array([[0.0, 1.0]])
        result = cosine_distance(corrupted, clean, corrupted)
        assert abs(result - 0.0) < 1e-4

    def test_zero_vector(self):
        """Zero vectors should return 0.0 gracefully."""
        zero = mx.array([[0.0, 0.0]])
        nonzero = mx.array([[1.0, 0.0]])
        result = cosine_distance(zero, nonzero, zero)
        assert result == 0.0


class TestGetMetric:
    """Tests for metric registry lookup."""

    def test_string_lookup(self):
        """Should resolve string names to functions."""
        for name in ["logit_diff", "kl", "l2", "cosine", "ce_diff"]:
            fn = get_metric(name)
            assert callable(fn)

    def test_callable_passthrough(self):
        """Callable input should be returned as-is."""
        def my_metric(*args, **kwargs):
            return 0.0
        assert get_metric(my_metric) is my_metric

    def test_unknown_metric(self):
        """Unknown string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("nonexistent")

    def test_aliases(self):
        """Aliases should resolve to same function."""
        assert get_metric("kl") is get_metric("kl_divergence")
        assert get_metric("l2") is get_metric("l2_distance")
        assert get_metric("cosine") is get_metric("cosine_distance")
        assert get_metric("ce_diff") is get_metric("cross_entropy_diff")
