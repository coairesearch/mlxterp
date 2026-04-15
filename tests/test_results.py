"""Tests for mlxterp.results module."""

import json
import pytest
import mlx.core as mx
from mlxterp.results import (
    AnalysisResult,
    PatchingResult,
    AttributionResult,
    DLAResult,
    GenerationResult,
    ConversationResult,
    CircuitResult,
    _array_to_list,
)


class TestArrayToList:
    """Tests for the serialization helper."""

    def test_mx_array(self):
        arr = mx.array([1.0, 2.0, 3.0])
        result = _array_to_list(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_nested_dict(self):
        data = {"a": mx.array([1, 2]), "b": {"c": mx.array([3])}}
        result = _array_to_list(data)
        assert result == {"a": [1, 2], "b": {"c": [3]}}

    def test_nested_list(self):
        data = [mx.array([1]), mx.array([2])]
        result = _array_to_list(data)
        assert result == [[1], [2]]

    def test_passthrough(self):
        assert _array_to_list(42) == 42
        assert _array_to_list("hello") == "hello"


class TestAnalysisResult:
    """Tests for the base AnalysisResult class."""

    def test_basic_creation(self):
        result = AnalysisResult(data={"key": "value"}, result_type="test")
        assert result.data == {"key": "value"}
        assert result.result_type == "test"

    def test_summary(self):
        result = AnalysisResult(data={"a": 1, "b": 2})
        assert "2 entries" in result.summary()

    def test_to_json(self):
        result = AnalysisResult(
            data={"scores": [1, 2, 3]},
            metadata={"model": "test"},
            result_type="test",
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["result_type"] == "test"
        assert parsed["data"]["scores"] == [1, 2, 3]
        assert "summary" in parsed

    def test_to_json_with_mx_array(self):
        result = AnalysisResult(data={"arr": mx.array([1.0, 2.0])})
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["data"]["arr"] == [1.0, 2.0]

    def test_to_markdown(self):
        result = AnalysisResult(
            data={"key": "value"},
            metadata={"model": "llama"},
            result_type="test_analysis",
        )
        md = result.to_markdown()
        assert "# Test Analysis" in md
        assert "model" in md
        assert "llama" in md

    def test_plot_raises(self):
        result = AnalysisResult(data={})
        with pytest.raises(NotImplementedError):
            result.plot()


class TestPatchingResult:
    """Tests for PatchingResult."""

    def test_summary_with_effects(self):
        result = PatchingResult(
            data={},
            effect_matrix=mx.array([0.1, 0.5, 0.3, 0.8, 0.2]),
            layers=[0, 1, 2, 3, 4],
            component="mlp",
            metric_name="logit_diff",
        )
        s = result.summary()
        assert "0.8" in s  # max effect
        assert "mlp" in s
        assert "logit_diff" in s

    def test_summary_2d(self):
        result = PatchingResult(
            data={},
            effect_matrix=mx.array([[0.1, 0.9], [0.2, 0.3]]),
            layers=[0, 1],
            component="attn",
            metric_name="l2",
        )
        s = result.summary()
        assert "0.9" in s

    def test_summary_no_results(self):
        result = PatchingResult(data={})
        assert "no results" in result.summary()

    def test_top_components_1d(self):
        result = PatchingResult(
            data={},
            effect_matrix=mx.array([0.1, 0.8, 0.3, 0.5, 0.2]),
            layers=[0, 1, 2, 3, 4],
        )
        top = result.top_components(k=3)
        assert len(top) == 3
        assert top[0][0] == 1  # layer 1 has highest effect
        assert abs(top[0][1] - 0.8) < 1e-5

    def test_top_components_2d(self):
        result = PatchingResult(
            data={},
            effect_matrix=mx.array([[0.1, 0.9], [0.8, 0.2]]),
            layers=[0, 1],
        )
        top = result.top_components(k=2)
        assert len(top) == 2
        # Layer 0 has max 0.9, layer 1 has max 0.8
        assert top[0][0] == 0

    def test_top_components_empty(self):
        result = PatchingResult(data={})
        assert result.top_components() == []


class TestAttributionResult:
    """Tests for AttributionResult."""

    def test_summary(self):
        result = AttributionResult(
            data={},
            attribution_scores=mx.array([[0.1, -0.5], [0.3, 0.8]]),
            layers=[0, 1],
            component="attn_head",
            method="finite_diff",
        )
        s = result.summary()
        assert "0.8" in s
        assert "finite_diff" in s
        assert "attn_head" in s

    def test_summary_no_results(self):
        result = AttributionResult(data={})
        assert "no results" in result.summary()


class TestDLAResult:
    """Tests for DLAResult."""

    def test_summary(self):
        result = DLAResult(
            data={},
            head_contributions=mx.array([[0.1, -0.3], [0.5, 0.2]]),
            target_token=42,
            target_token_str="Paris",
        )
        s = result.summary()
        assert "Paris" in s
        assert "0.5" in s

    def test_summary_no_results(self):
        result = DLAResult(data={})
        assert "no results" in result.summary()


class TestGenerationResult:
    """Tests for GenerationResult."""

    def test_summary(self):
        result = GenerationResult(
            data={},
            text="The capital of France is Paris.",
            tokens=[1, 2, 3, 4, 5, 6],
            prompt="The capital of",
        )
        s = result.summary()
        assert "6 tokens" in s
        assert "Paris" in s

    def test_summary_long_text(self):
        result = GenerationResult(
            data={},
            text="A" * 100,
            tokens=list(range(100)),
        )
        s = result.summary()
        assert "..." in s


class TestConversationResult:
    """Tests for ConversationResult."""

    def test_summary(self):
        result = ConversationResult(
            data={},
            turns=[
                {"role": "user", "index": 0},
                {"role": "assistant", "index": 1},
                {"role": "user", "index": 2},
            ],
        )
        s = result.summary()
        assert "3 turns" in s
        assert "user" in s
        assert "assistant" in s


class TestCircuitResult:
    """Tests for CircuitResult."""

    def test_summary(self):
        result = CircuitResult(
            data={},
            nodes=["layers.5.attn", "layers.7.mlp", "layers.9.attn"],
            edges=[
                ("layers.5.attn", "layers.7.mlp", 0.8),
                ("layers.7.mlp", "layers.9.attn", 0.6),
            ],
            threshold=0.5,
        )
        s = result.summary()
        assert "3 nodes" in s
        assert "2 edges" in s
        assert "0.5" in s

    def test_to_graph(self):
        result = CircuitResult(
            data={},
            nodes=["A", "B"],
            edges=[("A", "B", 0.9)],
        )
        graph = result.to_graph()
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["source"] == "A"
        assert graph["edges"][0]["weight"] == 0.9

    def test_json_roundtrip(self):
        result = CircuitResult(
            data={"info": "test"},
            nodes=["A", "B"],
            edges=[("A", "B", 0.5)],
            threshold=0.3,
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["result_type"] == "circuit"
