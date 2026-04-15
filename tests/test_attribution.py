"""Tests for mlxterp.causal.attribution module."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel
from mlxterp.causal.attribution import attribution_patching
from mlxterp.results import AttributionResult


class AttrModel(nn.Module):
    def __init__(self, hidden_dim=16, vocab_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [AttrLayer(hidden_dim), AttrLayer(hidden_dim)]
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class AttrLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x):
        return x + self.self_attn(x) + self.mlp(x)


@pytest.fixture
def model():
    m = AttrModel()
    mx.eval(m.parameters())
    return InterpretableModel(m)


class TestAttributionPatching:
    def test_basic(self, model):
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])
        result = attribution_patching(model, clean, corrupted)
        assert isinstance(result, AttributionResult)
        assert result.method == "finite_diff"

    def test_has_scores(self, model):
        result = attribution_patching(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]])
        )
        assert result.attribution_scores is not None
        assert result.attribution_scores.shape == (2,)  # 2 layers

    def test_specific_layers(self, model):
        result = attribution_patching(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            layers=[0],
        )
        assert result.attribution_scores.shape == (1,)

    def test_different_components(self, model):
        for comp in ["resid_post", "mlp", "attn"]:
            result = attribution_patching(
                model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
                component=comp,
            )
            assert isinstance(result, AttributionResult)

    def test_result_data(self, model):
        result = attribution_patching(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]])
        )
        assert "scores" in result.data
        assert "method" in result.data
        assert result.data["method"] == "finite_diff"

    def test_result_json(self, model):
        import json
        result = attribution_patching(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]])
        )
        parsed = json.loads(result.to_json())
        assert parsed["result_type"] == "attribution"

    def test_summary(self, model):
        result = attribution_patching(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]])
        )
        assert "finite_diff" in result.summary()
