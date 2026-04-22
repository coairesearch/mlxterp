"""Tests for mlxterp.causal.patching module."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel
from mlxterp.causal.patching import activation_patching
from mlxterp.results import PatchingResult


class SimpleModel(nn.Module):
    """Minimal 2-layer model for testing patching."""

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
    """Minimal transformer layer with self_attn and mlp."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x):
        h = x + self.self_attn(x)
        h = h + self.mlp(h)
        return h


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = SimpleModel(hidden_dim=16, vocab_size=32)
    mx.eval(model.parameters())
    return InterpretableModel(model)


class TestActivationPatching:
    """Tests for the activation_patching function."""

    def test_basic_patching(self, simple_model):
        """Basic layer-level patching should return PatchingResult."""
        clean = mx.array([[1, 2, 3, 4]])
        corrupted = mx.array([[5, 6, 7, 8]])

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="mlp",
            metric="l2",
        )

        assert isinstance(result, PatchingResult)
        assert result.component == "mlp"
        assert result.metric_name == "l2"
        assert len(result.layers) == 2  # 2-layer model
        assert result.effect_matrix is not None

    def test_specific_layers(self, simple_model):
        """Should only patch specified layers."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            layers=[0],
            component="mlp",
            metric="l2",
        )

        assert len(result.layers) == 1
        assert result.layers == [0]

    def test_resid_post(self, simple_model):
        """Patching resid_post (full layer output) should work."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="resid_post",
            metric="l2",
        )

        assert isinstance(result, PatchingResult)
        assert result.component == "resid_post"

    def test_attn_component(self, simple_model):
        """Patching attention component should work."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="attn",
            metric="l2",
        )

        assert isinstance(result, PatchingResult)

    def test_different_metrics(self, simple_model):
        """Should work with different metric names."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        for metric_name in ["l2", "cosine", "kl"]:
            result = activation_patching(
                simple_model,
                clean=clean,
                corrupted=corrupted,
                component="mlp",
                metric=metric_name,
            )
            assert result.metric_name == metric_name

    def test_custom_metric(self, simple_model):
        """Should work with custom callable metric."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        def my_metric(patched, clean, corrupted, **kwargs):
            return 0.42

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="mlp",
            metric=my_metric,
        )

        # All effects should be 0.42
        effects = result.effect_matrix.tolist()
        for e in effects:
            assert abs(e - 0.42) < 1e-5

    def test_result_has_data(self, simple_model):
        """Result should contain structured data."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="mlp",
            metric="l2",
        )

        assert "effects" in result.data
        assert "component" in result.data
        assert "metric" in result.data
        assert result.data["component"] == "mlp"

    def test_result_summary(self, simple_model):
        """Result summary should be informative."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="mlp",
            metric="l2",
        )

        summary = result.summary()
        assert "mlp" in summary
        assert "l2" in summary

    def test_result_json(self, simple_model):
        """Result should serialize to valid JSON."""
        import json
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="mlp",
            metric="l2",
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["result_type"] == "patching"

    def test_top_components(self, simple_model):
        """top_components should return sorted list."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="mlp",
            metric="l2",
        )

        top = result.top_components(k=2)
        assert len(top) == 2
        # Should be sorted by abs value descending
        assert abs(top[0][1]) >= abs(top[1][1])

    def test_position_level_patching(self, simple_model):
        """Position-level patching should only affect specified positions."""
        clean = mx.array([[1, 2, 3, 4, 5]])
        corrupted = mx.array([[6, 7, 8, 9, 10]])

        result = activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="mlp",
            metric="l2",
            positions=[2, 3],
        )

        assert isinstance(result, PatchingResult)
        assert result.metadata["positions"] == [2, 3]

    def test_verbose_mode(self, simple_model, capsys):
        """Verbose mode should print progress."""
        clean = mx.array([[1, 2, 3]])
        corrupted = mx.array([[4, 5, 6]])

        activation_patching(
            simple_model,
            clean=clean,
            corrupted=corrupted,
            component="mlp",
            metric="l2",
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "clean" in captured.out.lower() or "corrupt" in captured.out.lower()
