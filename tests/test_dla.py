"""Tests for mlxterp.causal.dla Direct Logit Attribution."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel
from mlxterp.causal.dla import direct_logit_attribution
from mlxterp.results import DLAResult


class DLAModel(nn.Module):
    """Model with lm_head for testing DLA."""

    def __init__(self, hidden_dim=16, vocab_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [
            DLALayer(hidden_dim),
            DLALayer(hidden_dim),
        ]
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class DLALayer(nn.Module):
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
    m = DLAModel(hidden_dim=16, vocab_size=32)
    mx.eval(m.parameters())
    return InterpretableModel(m)


class TestDirectLogitAttribution:
    """Tests for direct_logit_attribution function."""

    def test_basic_dla(self, model):
        """DLA should return a DLAResult."""
        tokens = mx.array([[1, 2, 3, 4]])
        result = direct_logit_attribution(model, tokens)

        assert isinstance(result, DLAResult)
        assert result.result_type == "dla"

    def test_has_contributions(self, model):
        """Result should have head and MLP contributions."""
        tokens = mx.array([[1, 2, 3]])
        result = direct_logit_attribution(model, tokens)

        assert result.head_contributions is not None
        assert result.mlp_contributions is not None

    def test_contribution_shapes(self, model):
        """Contributions should have shape (n_layers,)."""
        tokens = mx.array([[1, 2, 3]])
        result = direct_logit_attribution(model, tokens)

        # Should have one entry per layer
        assert result.head_contributions.shape == (2,)  # 2 layers
        assert result.mlp_contributions.shape == (2,)

    def test_specific_target_token(self, model):
        """Should accept specific target token."""
        tokens = mx.array([[1, 2, 3]])
        result = direct_logit_attribution(model, tokens, target_token=5)

        assert result.target_token == 5

    def test_auto_target_token(self, model):
        """Should auto-detect target from argmax when not specified."""
        tokens = mx.array([[1, 2, 3]])
        result = direct_logit_attribution(model, tokens)

        assert result.target_token is not None
        assert isinstance(result.target_token, int)

    def test_specific_layers(self, model):
        """Should work with specific layer subset."""
        tokens = mx.array([[1, 2, 3]])
        result = direct_logit_attribution(model, tokens, layers=[0])

        assert result.head_contributions.shape == (1,)
        assert result.mlp_contributions.shape == (1,)

    def test_specific_position(self, model):
        """Should analyze a specific token position."""
        tokens = mx.array([[1, 2, 3, 4, 5]])
        result = direct_logit_attribution(model, tokens, position=2)

        assert result.data["position"] == 2

    def test_result_data(self, model):
        """Result data should contain expected fields."""
        tokens = mx.array([[1, 2, 3]])
        result = direct_logit_attribution(model, tokens)

        assert "layers" in result.data
        assert "attn_contributions" in result.data
        assert "mlp_contributions" in result.data
        assert "target_token" in result.data

    def test_result_summary(self, model):
        """Summary should contain useful info."""
        tokens = mx.array([[1, 2, 3]])
        result = direct_logit_attribution(model, tokens)

        summary = result.summary()
        assert "DLA" in summary

    def test_result_json(self, model):
        """Should serialize to valid JSON."""
        import json
        tokens = mx.array([[1, 2, 3]])
        result = direct_logit_attribution(model, tokens)

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["result_type"] == "dla"

    def test_contributions_nonzero(self, model):
        """At least some contributions should be non-zero."""
        tokens = mx.array([[1, 2, 3, 4]])
        result = direct_logit_attribution(model, tokens)

        total_attn = float(mx.sum(mx.abs(result.head_contributions)))
        total_mlp = float(mx.sum(mx.abs(result.mlp_contributions)))

        assert total_attn > 0 or total_mlp > 0, "All contributions are zero"
