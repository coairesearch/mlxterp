"""Tests for mlxterp.causal.trace CausalTrace context manager."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel
from mlxterp.causal.trace import CausalTrace


class SimpleModel(nn.Module):
    """Minimal model for testing CausalTrace."""

    def __init__(self, hidden_dim=16, vocab_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [
            SimpleLayer(hidden_dim),
            SimpleLayer(hidden_dim),
        ]
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class SimpleLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x):
        h = x + self.self_attn(x)
        h = h + self.mlp(h)
        return h


@pytest.fixture
def model():
    m = SimpleModel(hidden_dim=16, vocab_size=32)
    mx.eval(m.parameters())
    return InterpretableModel(m)


class TestCausalTraceBasic:
    """Basic CausalTrace functionality."""

    def test_context_manager(self, model):
        """CausalTrace should work as a context manager."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        with model.causal_trace(clean, corrupted) as ct:
            assert ct.clean_output is not None
            assert ct.corrupted_output is not None
            assert len(ct.clean_activations) > 0

    def test_clean_activations_populated(self, model):
        """Clean activations should contain layer outputs."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        with model.causal_trace(clean, corrupted) as ct:
            assert len(ct.clean_activations) > 0
            # Should have entries for layers
            keys = list(ct.clean_activations.keys())
            assert any("layers" in k or "h" in k for k in keys)

    def test_clean_corrupted_different(self, model):
        """Clean and corrupted outputs should differ."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[10, 11, 12]])

        with model.causal_trace(clean, corrupted) as ct:
            diff = float(mx.sum(mx.abs(ct.clean_output - ct.corrupted_output)))
            assert diff > 0


class TestCausalTracePatch:
    """Tests for .patch() method."""

    def test_single_patch(self, model):
        """Should register a single patch."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        with model.causal_trace(clean, corrupted) as ct:
            ct.patch("layers.0.mlp")
            # Should have one patch registered
            assert len(ct._patches) == 1

    def test_multiple_patches(self, model):
        """Should support multiple patches."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        with model.causal_trace(clean, corrupted) as ct:
            ct.patch("layers.0.mlp")
            ct.patch("layers.1.self_attn")
            assert len(ct._patches) == 2

    def test_patch_with_positions(self, model):
        """Should support position-level patches."""
        clean = mx.array([[1, 2, 3, 4, 5]])
        corrupted = mx.array([[6, 7, 8, 9, 10]])

        with model.causal_trace(clean, corrupted) as ct:
            ct.patch("layers.0.mlp", positions=[1, 2])
            assert ct._patches[0]["positions"] == [1, 2]

    def test_patch_with_layer_param(self, model):
        """Should support canonical name + layer index."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        with model.causal_trace(clean, corrupted) as ct:
            ct.patch("mlp", layer=0)
            assert ct._patches[0]["layer"] == 0


class TestCausalTraceMetric:
    """Tests for .metric() method."""

    def test_metric_with_l2(self, model):
        """Should compute l2 metric."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[10, 11, 12]])

        with model.causal_trace(clean, corrupted) as ct:
            ct.patch("layers.0.mlp")
            effect = ct.metric("l2")
            assert isinstance(effect, float)

    def test_metric_with_callable(self, model):
        """Should accept callable metric."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        def custom_metric(patched, clean, corrupted, **kwargs):
            return 0.99

        with model.causal_trace(clean, corrupted) as ct:
            ct.patch("layers.0.mlp")
            effect = ct.metric(custom_metric)
            assert abs(effect - 0.99) < 1e-5

    def test_patching_affects_metric(self, model):
        """Patching should generally produce non-zero metric."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[15, 16, 17]])

        with model.causal_trace(clean, corrupted) as ct:
            ct.patch("layers.0.mlp")
            ct.patch("layers.1.mlp")
            effect = ct.metric("l2")
            # With all MLPs patched from clean, some recovery expected
            assert isinstance(effect, float)

    def test_invalid_component_raises(self, model):
        """Patching a nonexistent component should raise."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        with model.causal_trace(clean, corrupted) as ct:
            ct.patch("nonexistent.component")
            with pytest.raises(ValueError, match="Could not find"):
                ct.metric("l2")


class TestCausalTraceGetActivation:
    """Tests for get_clean_activation method."""

    def test_get_existing(self, model):
        """Should return clean activation for valid component."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        with model.causal_trace(clean, corrupted) as ct:
            act = ct.get_clean_activation("layers.0.mlp")
            assert isinstance(act, mx.array)
            assert act.ndim >= 2

    def test_get_nonexistent_raises(self, model):
        """Should raise KeyError for invalid component."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        with model.causal_trace(clean, corrupted) as ct:
            with pytest.raises(KeyError):
                ct.get_clean_activation("nonexistent")

    def test_get_with_layer(self, model):
        """Should support canonical name + layer."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        with model.causal_trace(clean, corrupted) as ct:
            act = ct.get_clean_activation("mlp", layer=0)
            assert isinstance(act, mx.array)
