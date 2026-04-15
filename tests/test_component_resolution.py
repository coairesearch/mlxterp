"""Tests for component resolution and position-level interventions."""

import pytest
import mlx.core as mx
from mlxterp.core.module_resolver import (
    resolve_component,
    resolve_intervention_key,
    COMPONENT_ALIASES,
    LAYER_CONTAINERS,
)
from mlxterp.core.intervention import replace_at_positions


class TestResolveComponent:
    """Tests for resolve_component function."""

    def _make_activations(self, keys):
        """Create a mock activations dict with given keys."""
        return {k: mx.zeros((1,)) for k in keys}

    def test_llama_style_mlp(self):
        acts = self._make_activations([
            "model.model.layers.0.mlp",
            "model.model.layers.0.self_attn",
            "model.model.layers.1.mlp",
        ])
        key = resolve_component("mlp", 0, acts)
        assert key == "model.model.layers.0.mlp"

    def test_llama_style_attn(self):
        acts = self._make_activations([
            "model.model.layers.5.self_attn",
            "model.model.layers.5.mlp",
        ])
        key = resolve_component("attn", 5, acts)
        assert key == "model.model.layers.5.self_attn"

    def test_gpt2_style(self):
        acts = self._make_activations([
            "model.h.0.attn",
            "model.h.0.mlp",
        ])
        key = resolve_component("attn", 0, acts)
        assert key == "model.h.0.attn"

    def test_direct_layers(self):
        acts = self._make_activations([
            "layers.3.mlp",
            "layers.3.self_attn",
        ])
        key = resolve_component("mlp", 3, acts)
        assert key == "layers.3.mlp"

    def test_resid_post(self):
        acts = self._make_activations([
            "model.model.layers.2",
            "model.model.layers.2.mlp",
        ])
        key = resolve_component("resid_post", 2, acts)
        assert key == "model.model.layers.2"

    def test_resid_post_with_suffix(self):
        """If resid_post is stored as a suffixed key."""
        acts = self._make_activations([
            "model.model.layers.2.resid_post",
        ])
        key = resolve_component("resid_post", 2, acts)
        assert key == "model.model.layers.2.resid_post"

    def test_not_found(self):
        acts = self._make_activations(["something.else"])
        key = resolve_component("mlp", 0, acts)
        assert key is None

    def test_attn_head_resolves_to_attn(self):
        """attn_head should resolve to the same key as attn."""
        acts = self._make_activations(["model.model.layers.0.self_attn"])
        key = resolve_component("attn_head", 0, acts)
        assert key == "model.model.layers.0.self_attn"

    def test_direct_component_name(self):
        """Non-canonical names should be tried directly."""
        acts = self._make_activations(["model.model.layers.0.feed_forward"])
        key = resolve_component("mlp", 0, acts)
        assert key == "model.model.layers.0.feed_forward"

    def test_custom_component_path(self):
        """Full path like 'self_attn.q_proj' should work."""
        acts = self._make_activations(["model.model.layers.0.self_attn.q_proj"])
        key = resolve_component("self_attn.q_proj", 0, acts)
        assert key == "model.model.layers.0.self_attn.q_proj"


class TestResolveInterventionKey:
    """Tests for resolve_intervention_key function."""

    def test_double_prefix(self):
        assert resolve_intervention_key("model.model.layers.5.mlp") == "layers.5.mlp"

    def test_single_prefix(self):
        assert resolve_intervention_key("model.layers.5.mlp") == "layers.5.mlp"

    def test_no_prefix(self):
        assert resolve_intervention_key("layers.5.mlp") == "layers.5.mlp"

    def test_gpt2_style(self):
        assert resolve_intervention_key("model.h.0.attn") == "h.0.attn"


class TestReplaceAtPositions:
    """Tests for replace_at_positions intervention."""

    def test_3d_replacement(self):
        """Replace specific positions in (batch, seq, hidden) tensor."""
        original = mx.zeros((1, 5, 4))
        replacement = mx.ones((1, 5, 4))
        fn = replace_at_positions(replacement, [1, 3])
        result = fn(original)
        mx.eval(result)

        # Positions 1 and 3 should be 1.0, others should be 0.0
        assert float(mx.sum(result[:, 0, :])) == 0.0
        assert float(mx.sum(result[:, 1, :])) == 4.0
        assert float(mx.sum(result[:, 2, :])) == 0.0
        assert float(mx.sum(result[:, 3, :])) == 4.0
        assert float(mx.sum(result[:, 4, :])) == 0.0

    def test_2d_replacement(self):
        """Replace specific positions in (seq, hidden) tensor."""
        original = mx.zeros((5, 4))
        replacement = mx.ones((5, 4))
        fn = replace_at_positions(replacement, [0, 4])
        result = fn(original)
        mx.eval(result)

        assert float(mx.sum(result[0, :])) == 4.0
        assert float(mx.sum(result[1, :])) == 0.0
        assert float(mx.sum(result[4, :])) == 4.0

    def test_out_of_bounds_ignored(self):
        """Positions beyond tensor bounds should be silently skipped."""
        original = mx.zeros((1, 3, 4))
        replacement = mx.ones((1, 3, 4))
        fn = replace_at_positions(replacement, [1, 10])  # 10 is out of bounds
        result = fn(original)
        mx.eval(result)

        # Only position 1 should be replaced
        assert float(mx.sum(result[:, 0, :])) == 0.0
        assert float(mx.sum(result[:, 1, :])) == 4.0
        assert float(mx.sum(result[:, 2, :])) == 0.0

    def test_empty_positions(self):
        """Empty position list should return original unchanged."""
        original = mx.ones((1, 3, 4))
        replacement = mx.zeros((1, 3, 4))
        fn = replace_at_positions(replacement, [])
        result = fn(original)
        mx.eval(result)
        assert float(mx.sum(result)) == 12.0  # all ones

    def test_different_seq_lengths(self):
        """Should handle source and target with different sequence lengths."""
        original = mx.zeros((1, 6, 4))  # 6 tokens
        replacement = mx.ones((1, 4, 4))  # 4 tokens (shorter)
        fn = replace_at_positions(replacement, [4, 5])  # patch last 2 positions
        result = fn(original)
        mx.eval(result)

        # With end-alignment, src positions map to: 4->2, 5->3
        assert float(mx.sum(result[:, 4, :])) == 4.0
        assert float(mx.sum(result[:, 5, :])) == 4.0
        assert float(mx.sum(result[:, 0, :])) == 0.0


class TestComponentAliases:
    """Tests for component alias definitions."""

    def test_all_canonical_names_present(self):
        expected = {"resid_post", "resid_pre", "attn", "mlp", "attn_head", "self_attn"}
        assert expected.issubset(set(COMPONENT_ALIASES.keys()))

    def test_attn_aliases_include_self_attn(self):
        assert "self_attn" in COMPONENT_ALIASES["attn"]

    def test_mlp_aliases_include_feed_forward(self):
        assert "feed_forward" in COMPONENT_ALIASES["mlp"]

    def test_layer_containers(self):
        assert "layers" in LAYER_CONTAINERS
        assert "h" in LAYER_CONTAINERS
