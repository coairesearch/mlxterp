"""Tests for mlxterp.auto_interp module."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from unittest.mock import MagicMock, patch
from mlxterp import InterpretableModel
from mlxterp.auto_interp import (
    auto_label_feature,
    auto_label_features,
    sensitivity_test,
    FeatureLabel,
    _build_labeling_prompt,
    _parse_label_response,
    _find_top_features,
)


class AutoInterpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.layers = [AILayer()]
        self.norm = nn.RMSNorm(16)
        self.lm_head = nn.Linear(16, 32, bias=False)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class AILayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(16, 16, bias=False)
        self.mlp = nn.Linear(16, 16, bias=False)

    def __call__(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class MockSAE:
    def __init__(self):
        self.encoder = nn.Linear(16, 64, bias=False)
        self.decoder = nn.Linear(64, 16, bias=False)
        mx.eval(self.encoder.parameters())
        mx.eval(self.decoder.parameters())

    def encode(self, x):
        return mx.maximum(self.encoder(x), 0)

    def decode(self, x):
        return self.decoder(x)


@pytest.fixture
def model():
    m = AutoInterpModel()
    mx.eval(m.parameters())
    return InterpretableModel(m)


@pytest.fixture
def sae():
    return MockSAE()


class TestFeatureLabel:
    def test_basic(self):
        label = FeatureLabel(feature_id=42, label="test_label", confidence=0.8)
        assert label.feature_id == 42
        assert label.label == "test_label"
        assert label.confidence == 0.8

    def test_to_dict(self):
        label = FeatureLabel(
            feature_id=1, label="numbers", description="Detects numbers",
            confidence=0.9, evidence=[{"text": "123"}],
        )
        d = label.to_dict()
        assert d["feature_id"] == 1
        assert d["label"] == "numbers"
        assert len(d["evidence"]) == 1

    def test_defaults(self):
        label = FeatureLabel(feature_id=0)
        assert label.label == ""
        assert label.confidence == 0.0
        assert label.sensitivity_passed is None


class TestBuildLabelingPrompt:
    def test_prompt_structure(self):
        examples = [
            {"text": "Hello world", "activation_value": 0.95, "token_position": 1},
            {"text": "Test text", "activation_value": 0.80, "token_position": 0},
        ]
        prompt = _build_labeling_prompt(42, examples)
        assert "feature #42" in prompt
        assert "Hello world" in prompt
        assert "0.95" in prompt
        assert "JSON" in prompt

    def test_empty_examples(self):
        prompt = _build_labeling_prompt(0, [])
        assert "feature #0" in prompt


class TestParseLabelResponse:
    def test_valid_json(self):
        response = '{"label": "numbers", "description": "Detects numbers", "confidence": 0.9}'
        result = _parse_label_response(response)
        assert result["label"] == "numbers"
        assert result["confidence"] == 0.9

    def test_json_with_text(self):
        response = 'Here is my analysis:\n{"label": "greetings", "description": "Greeting words", "confidence": 0.7}\nDone.'
        result = _parse_label_response(response)
        assert result["label"] == "greetings"

    def test_invalid_json(self):
        result = _parse_label_response("This is not JSON at all")
        assert result["label"] == "parse_error"


class TestAutoLabelFeature:
    def test_no_api_client(self, model, sae):
        """Without anthropic SDK, should return placeholder."""
        label = auto_label_feature(
            model, sae, feature_id=0,
            texts=[mx.array([[1, 2, 3]])],
            layer=0, llm_client=None,
        )
        assert isinstance(label, FeatureLabel)
        assert label.feature_id == 0
        # Should either be "unlabeled" (no SDK) or "api_error" or have evidence

    def test_with_mock_client(self, model, sae):
        """With a mock LLM client, should parse response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"label": "test_feature", "description": "A test", "confidence": 0.8}')
        ]
        mock_client.messages.create.return_value = mock_response

        label = auto_label_feature(
            model, sae, feature_id=5,
            texts=[mx.array([[1, 2, 3]])],
            layer=0,
            llm_client=mock_client,
        )
        assert label.label == "test_feature"
        assert label.confidence == 0.8

    def test_no_activating_examples(self, model, sae):
        """Empty text list should handle gracefully."""
        label = auto_label_feature(
            model, sae, feature_id=0,
            texts=[],
            layer=0,
        )
        assert label.label == "no_activations"


class TestAutoLabelFeatures:
    def test_batch_labeling(self, model, sae):
        """Batch labeling with mock client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"label": "batch_test", "description": "Batch", "confidence": 0.7}')
        ]
        mock_client.messages.create.return_value = mock_response

        labels = auto_label_features(
            model, sae,
            texts=[mx.array([[1, 2, 3]])],
            layer=0,
            feature_ids=[0, 1, 2],
            llm_client=mock_client,
            verbose=False,
        )
        assert len(labels) == 3
        assert all(l.label == "batch_test" for l in labels)


class TestSensitivityTest:
    def test_basic(self, model, sae):
        label = FeatureLabel(feature_id=0, label="test")
        result = sensitivity_test(
            model, sae, label,
            test_texts=[mx.array([[1, 2, 3]]), mx.array([[10, 11, 12]])],
            layer=0,
        )
        assert result.sensitivity_details != ""
        assert result.sensitivity_passed is not None

    def test_empty_texts(self, model, sae):
        label = FeatureLabel(feature_id=0, label="test")
        result = sensitivity_test(
            model, sae, label,
            test_texts=[],
            layer=0,
        )
        assert result.sensitivity_passed is None


class TestFindTopFeatures:
    def test_basic(self, model, sae):
        features = _find_top_features(
            model, sae,
            texts=[mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]])],
            layer=0, component="mlp", top_k=5,
        )
        assert len(features) == 5
        assert all(isinstance(f, int) for f in features)

    def test_empty_texts(self, model, sae):
        features = _find_top_features(
            model, sae, texts=[], layer=0, component="mlp", top_k=5,
        )
        assert len(features) == 5  # Falls back to range(5)
