"""Tests for mlxterp.workflows module."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel
from mlxterp.workflows import (
    behavior_localization,
    circuit_discovery,
    feature_investigation,
    WorkflowResult,
)


class WFModel(nn.Module):
    def __init__(self, hidden_dim=16, vocab_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [WFLayer(hidden_dim), WFLayer(hidden_dim)]
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class WFLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class MockSAE:
    def __init__(self, hidden_dim=16, n_features=64):
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
    m = WFModel()
    mx.eval(m.parameters())
    return InterpretableModel(m)


@pytest.fixture
def sae():
    return MockSAE()


class TestBehaviorLocalization:
    def test_basic(self, model):
        """Test with steps that work on simple models (no attn_head)."""
        result = behavior_localization(
            model,
            clean=mx.array([[1, 2, 3]]),
            corrupted=mx.array([[4, 5, 6]]),
            steps=["dla", "patch_mlp", "patch_attn"],
            verbose=False,
        )
        assert isinstance(result, WorkflowResult)
        assert len(result.steps) > 0

    def test_has_narrative(self, model):
        result = behavior_localization(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            steps=["patch_mlp"],
            verbose=False,
        )
        assert result.narrative != ""

    def test_specific_steps(self, model):
        result = behavior_localization(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            steps=["patch_mlp"],
            verbose=False,
        )
        assert len(result.steps) == 1
        assert result.steps[0][0] == "patch_mlp"

    def test_get_step(self, model):
        result = behavior_localization(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            steps=["patch_mlp"],
            verbose=False,
        )
        step = result.get_step("patch_mlp")
        assert step is not None
        assert result.get_step("nonexistent") is None

    def test_summary(self, model):
        result = behavior_localization(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            steps=["patch_mlp"],
            verbose=False,
        )
        assert "Workflow" in result.summary()

    def test_to_markdown(self, model):
        result = behavior_localization(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            steps=["patch_mlp", "patch_attn"],
            verbose=False,
        )
        md = result.to_markdown()
        assert "Behavior Localization" in md

    def test_to_json(self, model):
        import json
        result = behavior_localization(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            steps=["patch_mlp"],
            verbose=False,
        )
        parsed = json.loads(result.to_json())
        assert parsed["result_type"] == "workflow"


class TestCircuitDiscovery:
    def test_basic(self, model):
        result = circuit_discovery(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            verbose=False,
        )
        assert isinstance(result, WorkflowResult)
        assert len(result.steps) > 0

    def test_specific_steps(self, model):
        result = circuit_discovery(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            steps=["attribution"],
            verbose=False,
        )
        assert len(result.steps) == 1

    def test_with_acdc(self, model):
        result = circuit_discovery(
            model, mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]]),
            steps=["acdc"],
            threshold=0.01,
            verbose=False,
        )
        acdc_result = result.get_step("acdc")
        assert acdc_result is not None


class TestFeatureInvestigation:
    def test_basic(self, model, sae):
        result = feature_investigation(
            model, sae,
            text=mx.array([[1, 2, 3]]),
            layer=0,
            top_k=3,
            verbose=False,
        )
        assert isinstance(result, WorkflowResult)
        assert len(result.steps) >= 2  # active_features + ablation

    def test_has_active_features(self, model, sae):
        result = feature_investigation(
            model, sae, mx.array([[1, 2, 3]]),
            layer=0, top_k=3, verbose=False,
        )
        step = result.get_step("active_features")
        assert step is not None
        assert isinstance(step, dict)

    def test_has_ablation(self, model, sae):
        result = feature_investigation(
            model, sae, mx.array([[1, 2, 3]]),
            layer=0, top_k=3, verbose=False,
        )
        step = result.get_step("ablation")
        assert step is not None
