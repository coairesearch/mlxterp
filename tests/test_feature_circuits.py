"""Tests for mlxterp.causal.feature_circuits and visualization.dashboards."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel
from mlxterp.causal.feature_circuits import feature_patching, feature_circuit
from mlxterp.visualization.dashboards import (
    max_activating_examples,
    feature_activation_histogram,
    generate_feature_dashboard_html,
    _html_escape,
)
from mlxterp.results import CircuitResult


class SimpleModelFC(nn.Module):
    def __init__(self, hidden_dim=16, vocab_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [FCLayer(hidden_dim)]
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class FCLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class MockSAE:
    """Mock SAE for testing feature circuits."""

    def __init__(self, hidden_dim=16, n_features=64):
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.encoder = nn.Linear(hidden_dim, n_features, bias=False)
        self.decoder = nn.Linear(n_features, hidden_dim, bias=False)
        mx.eval(self.encoder.parameters())
        mx.eval(self.decoder.parameters())

    def encode(self, x):
        return mx.maximum(self.encoder(x), 0)

    def decode(self, x):
        return self.decoder(x)


@pytest.fixture
def model():
    m = SimpleModelFC()
    mx.eval(m.parameters())
    return InterpretableModel(m)


@pytest.fixture
def sae():
    return MockSAE(hidden_dim=16, n_features=64)


class TestFeaturePatching:
    def test_basic(self, model, sae):
        effects = feature_patching(
            model, sae, mx.array([[1, 2, 3]]),
            layer=0, component="mlp",
            top_k=5,
        )
        assert isinstance(effects, dict)
        assert len(effects) == 5

    def test_specific_features(self, model, sae):
        effects = feature_patching(
            model, sae, mx.array([[1, 2, 3]]),
            layer=0, component="mlp",
            feature_ids=[0, 5, 10],
        )
        assert len(effects) == 3
        assert 0 in effects
        assert 5 in effects
        assert 10 in effects

    def test_effects_are_floats(self, model, sae):
        effects = feature_patching(
            model, sae, mx.array([[1, 2, 3]]),
            layer=0, top_k=3,
        )
        for fid, eff in effects.items():
            assert isinstance(fid, int)
            assert isinstance(eff, float)


class TestFeatureCircuit:
    def test_basic(self, model, sae):
        result = feature_circuit(
            model, sae, mx.array([[1, 2, 3]]),
            layer=0, component="mlp",
            top_k=5, threshold=0.001,
        )
        assert isinstance(result, CircuitResult)

    def test_has_nodes(self, model, sae):
        result = feature_circuit(
            model, sae, mx.array([[1, 2, 3]]),
            layer=0, top_k=5, threshold=0.0001,
        )
        assert isinstance(result.nodes, list)

    def test_high_threshold_prunes(self, model, sae):
        result = feature_circuit(
            model, sae, mx.array([[1, 2, 3]]),
            layer=0, top_k=5, threshold=100.0,
        )
        assert len(result.nodes) == 0

    def test_result_data(self, model, sae):
        result = feature_circuit(
            model, sae, mx.array([[1, 2, 3]]),
            layer=0, top_k=3,
        )
        assert "feature_effects" in result.data
        assert "layer" in result.data


class TestDashboards:
    def test_html_escape(self):
        assert _html_escape("<b>test</b>") == "&lt;b&gt;test&lt;/b&gt;"
        assert _html_escape('a "b" c') == 'a &quot;b&quot; c'
        assert _html_escape("a & b") == "a &amp; b"

    def test_generate_dashboard_html(self):
        examples = [
            {"text": "Hello world", "activation_value": 0.95, "token_position": 1},
            {"text": "Test text", "activation_value": 0.80, "token_position": 0},
        ]
        histogram = {"mean": 0.5, "std": 0.2, "sparsity": 0.8}

        html = generate_feature_dashboard_html(42, examples, histogram)
        assert "Feature 42" in html
        assert "Hello world" in html
        assert "0.95" in html
        assert "0.5" in html  # mean
        assert "80.0%" in html  # sparsity

    def test_dashboard_custom_title(self):
        html = generate_feature_dashboard_html(
            0, [], {"mean": 0, "std": 0, "sparsity": 0},
            title="Custom Title",
        )
        assert "Custom Title" in html

    def test_histogram_empty(self):
        result = feature_activation_histogram.__wrapped__ if hasattr(
            feature_activation_histogram, '__wrapped__'
        ) else None

        # Test the histogram computation with mock data
        histogram = {
            "bin_edges": [],
            "counts": [],
            "mean": 0.0,
            "std": 0.0,
            "sparsity": 1.0,
        }
        html = generate_feature_dashboard_html(0, [], histogram)
        assert "mlxterp" in html

    def test_max_activating_examples(self, model, sae):
        """Integration test with mock model and SAE."""
        results = max_activating_examples(
            model, sae,
            feature_id=0,
            texts=[mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]])],
            layer=0,
        )
        assert isinstance(results, list)
        assert len(results) <= 2
        for r in results:
            assert "activation_value" in r
            assert "token_position" in r
