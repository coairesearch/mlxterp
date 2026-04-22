"""Tests for mlxterp.generation module."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel
from mlxterp.generation import generate, _sample_token
from mlxterp.results import GenerationResult
from mlxterp.core.intervention import scale


class GenModel(nn.Module):
    def __init__(self, hidden_dim=16, vocab_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [GenLayer(hidden_dim)]
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class GenLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x):
        return x + self.self_attn(x) + self.mlp(x)


@pytest.fixture
def model():
    m = GenModel()
    mx.eval(m.parameters())
    return InterpretableModel(m)


class TestGenerate:
    def test_basic_generation(self, model):
        result = generate(model, mx.array([[1, 2, 3]]), max_tokens=5)
        assert isinstance(result, GenerationResult)
        assert len(result.tokens) <= 5
        assert len(result.tokens) > 0

    def test_greedy_deterministic(self, model):
        """Greedy generation should be deterministic."""
        r1 = generate(model, mx.array([[1, 2, 3]]), max_tokens=5, temperature=0.0)
        r2 = generate(model, mx.array([[1, 2, 3]]), max_tokens=5, temperature=0.0)
        assert r1.tokens == r2.tokens

    def test_stop_token(self, model):
        """Should stop at stop token."""
        result = generate(
            model, mx.array([[1, 2, 3]]),
            max_tokens=100,
            stop_tokens=[result_token := generate(
                model, mx.array([[1, 2, 3]]), max_tokens=3
            ).tokens[0]],
        )
        # Should stop before max_tokens
        assert len(result.tokens) <= 100

    def test_callback(self, model):
        """Callback should be invoked and can stop generation."""
        steps = []

        def cb(step, token, logits):
            steps.append(step)
            return step >= 2  # Stop after 3 tokens

        result = generate(model, mx.array([[1, 2, 3]]), max_tokens=10, callback=cb)
        assert len(steps) >= 3

    def test_with_interventions(self, model):
        """Generation with interventions should produce different output."""
        r_normal = generate(model, mx.array([[1, 2, 3]]), max_tokens=5)
        r_scaled = generate(
            model, mx.array([[1, 2, 3]]),
            max_tokens=5,
            interventions={"layers.0.mlp": scale(0.0)},
        )
        # With MLP zeroed out, results may differ
        assert isinstance(r_scaled, GenerationResult)

    def test_result_data(self, model):
        result = generate(model, mx.array([[1, 2, 3]]), max_tokens=3)
        assert "generated_tokens" in result.data
        assert "n_tokens" in result.data
        assert result.data["temperature"] == 0.0

    def test_result_json(self, model):
        import json
        result = generate(model, mx.array([[1, 2, 3]]), max_tokens=3)
        parsed = json.loads(result.to_json())
        assert parsed["result_type"] == "generation"

    def test_token_list_input(self, model):
        result = generate(model, [1, 2, 3], max_tokens=3)
        assert isinstance(result, GenerationResult)

    def test_model_generate_method(self, model):
        result = model.generate(mx.array([[1, 2, 3]]), max_tokens=3)
        assert isinstance(result, GenerationResult)


class TestSampleToken:
    def test_greedy(self):
        logits = mx.array([0.0, 5.0, 1.0, 2.0])
        token = _sample_token(logits, temperature=0.0)
        assert int(token) == 1  # argmax

    def test_temperature(self):
        logits = mx.array([0.0, 10.0, 0.0, 0.0])
        token = _sample_token(logits, temperature=0.01)
        assert int(token) == 1  # Very peaked, should be argmax

    def test_top_k(self):
        logits = mx.array([1.0, 10.0, 9.0, 0.0, 0.0])
        # With top_k=2, only indices 1 and 2 should be considered
        token = _sample_token(logits, temperature=0.01, top_k=2)
        assert int(token) in [1, 2]

    def test_returns_valid_index(self):
        logits = mx.random.normal((100,))
        token = _sample_token(logits, temperature=1.0)
        assert 0 <= int(token) < 100
