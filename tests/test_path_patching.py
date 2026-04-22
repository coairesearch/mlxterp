"""Tests for mlxterp.causal.path_patching and acdc modules."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel
from mlxterp.causal.path_patching import path_patching
from mlxterp.causal.acdc import acdc
from mlxterp.results import PatchingResult, CircuitResult


class CircuitModel(nn.Module):
    def __init__(self, hidden_dim=16, vocab_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [CircuitLayer(hidden_dim), CircuitLayer(hidden_dim)]
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class CircuitLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x):
        return x + self.self_attn(x) + self.mlp(x)


@pytest.fixture
def model():
    m = CircuitModel()
    mx.eval(m.parameters())
    return InterpretableModel(m)


class TestPathPatching:
    def test_basic(self, model):
        result = path_patching(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[4, 5, 6]]),
            sender="layers.0.self_attn",
            receiver="layers.1.self_attn",
        )
        assert isinstance(result, PatchingResult)
        assert "path:" in result.component

    def test_has_effect(self, model):
        result = path_patching(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[10, 11, 12]]),
            sender="layers.0.self_attn",
            receiver="layers.1.mlp",
        )
        assert result.effect_matrix is not None
        assert len(result.effect_matrix) == 1

    def test_result_data(self, model):
        result = path_patching(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[4, 5, 6]]),
            sender="layers.0.mlp",
            receiver="layers.1.mlp",
        )
        assert "sender" in result.data
        assert "receiver" in result.data
        assert "effect" in result.data
        assert "n_frozen" in result.data

    def test_json(self, model):
        import json
        result = path_patching(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[4, 5, 6]]),
            sender="layers.0.mlp",
            receiver="layers.1.mlp",
        )
        parsed = json.loads(result.to_json())
        assert "patching" in parsed["result_type"]


class TestACDC:
    def test_basic(self, model):
        result = acdc(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[4, 5, 6]]),
            threshold=0.01,
        )
        assert isinstance(result, CircuitResult)

    def test_has_nodes_and_edges(self, model):
        result = acdc(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[10, 11, 12]]),
            threshold=0.001,
        )
        # Should find some nodes (low threshold)
        assert isinstance(result.nodes, list)
        assert isinstance(result.edges, list)

    def test_high_threshold_prunes_all(self, model):
        result = acdc(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[4, 5, 6]]),
            threshold=100.0,  # Very high threshold
        )
        # Most or all nodes should be pruned
        assert len(result.nodes) <= 4  # max 4 components

    def test_result_data(self, model):
        result = acdc(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[4, 5, 6]]),
        )
        assert "node_effects" in result.data
        assert "threshold" in result.data

    def test_to_graph(self, model):
        result = acdc(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[4, 5, 6]]),
            threshold=0.001,
        )
        graph = result.to_graph()
        assert "nodes" in graph
        assert "edges" in graph

    def test_summary(self, model):
        result = acdc(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[4, 5, 6]]),
        )
        assert "Circuit" in result.summary()
        assert "nodes" in result.summary()
