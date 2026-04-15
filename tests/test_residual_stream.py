"""Tests for mlxterp.causal.residual ResidualStreamAccessor."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel
from mlxterp.causal.residual import ResidualStreamAccessor


class ResidualModel(nn.Module):
    """Model with explicit residual structure for testing."""

    def __init__(self, hidden_dim=16, vocab_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [
            ResidualLayer(hidden_dim),
            ResidualLayer(hidden_dim),
            ResidualLayer(hidden_dim),
        ]
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class ResidualLayer(nn.Module):
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
    m = ResidualModel(hidden_dim=16, vocab_size=32)
    mx.eval(m.parameters())
    return InterpretableModel(m)


class TestResidualStreamAccessor:
    """Tests for ResidualStreamAccessor."""

    def test_resid_pre_exists(self, model):
        """resid_pre should be captured during tracing."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        pre = rs.resid_pre(0)
        assert pre is not None
        assert isinstance(pre, mx.array)

    def test_resid_post_exists(self, model):
        """resid_post should be captured during tracing."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        post = rs.resid_post(0)
        assert post is not None
        assert isinstance(post, mx.array)

    def test_resid_pre_post_different(self, model):
        """resid_pre and resid_post should differ (layer transforms input)."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        pre = rs.resid_pre(0)
        post = rs.resid_post(0)

        if pre is not None and post is not None:
            diff = float(mx.sum(mx.abs(pre - post)))
            assert diff > 0, "resid_pre and resid_post should differ"

    def test_resid_pre_chain(self, model):
        """resid_pre[i] should approximately equal resid_post[i-1] for i > 0."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        post_0 = rs.resid_post(0)
        pre_1 = rs.resid_pre(1)

        if post_0 is not None and pre_1 is not None:
            diff = float(mx.max(mx.abs(post_0 - pre_1)))
            assert diff < 1e-5, f"resid_post[0] should equal resid_pre[1], diff={diff}"

    def test_attn_contribution(self, model):
        """attn_contribution should return the attention output."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        attn = rs.attn_contribution(0)
        assert attn is not None

    def test_mlp_contribution(self, model):
        """mlp_contribution should return the MLP output."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        mlp = rs.mlp_contribution(0)
        assert mlp is not None

    def test_resid_mid(self, model):
        """resid_mid should equal resid_pre + attn_contribution."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        pre = rs.resid_pre(0)
        attn = rs.attn_contribution(0)
        mid = rs.resid_mid(0)

        if pre is not None and attn is not None and mid is not None:
            expected = pre + attn
            diff = float(mx.max(mx.abs(mid - expected)))
            assert diff < 1e-5

    def test_layer_contribution(self, model):
        """layer_contribution should equal resid_post - resid_pre."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        pre = rs.resid_pre(0)
        post = rs.resid_post(0)
        contrib = rs.layer_contribution(0)

        if pre is not None and post is not None and contrib is not None:
            expected = post - pre
            diff = float(mx.max(mx.abs(contrib - expected)))
            assert diff < 1e-5

    def test_available_layers(self, model):
        """available_layers should return all layer indices."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        layers = rs.available_layers()
        assert len(layers) >= 3  # 3-layer model
        assert 0 in layers
        assert 1 in layers
        assert 2 in layers

    def test_nonexistent_layer(self, model):
        """Accessing a nonexistent layer should return None."""
        with model.trace(mx.array([[1, 2, 3]])) as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        assert rs.resid_pre(99) is None
        assert rs.resid_post(99) is None
        assert rs.attn_contribution(99) is None
        assert rs.mlp_contribution(99) is None

    def test_empty_activations(self):
        """Should handle empty activations gracefully."""
        rs = ResidualStreamAccessor({})
        assert rs.resid_pre(0) is None
        assert rs.available_layers() == []
