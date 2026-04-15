"""
Residual stream access utilities.

Provides decomposition of the residual stream into per-component contributions
(attention, MLP) at each layer, enabling Direct Logit Attribution and
circuit analysis.
"""

import mlx.core as mx
from typing import Dict, Optional

from ..core.module_resolver import resolve_component


class ResidualStreamAccessor:
    """
    Access residual stream components from a trace's activations.

    After running a trace, this class provides structured access to:
    - resid_pre: Input to each layer (= output of previous layer)
    - resid_post: Output of each layer
    - attn_contribution: Attention output (before residual add)
    - mlp_contribution: MLP output (before residual add)
    - resid_mid: State between attention and MLP (resid_pre + attn_output)

    Example:
        with model.trace("Hello world") as trace:
            pass

        rs = ResidualStreamAccessor(trace.activations)
        layer5_input = rs.resid_pre(5)
        attn_effect = rs.attn_contribution(5)
    """

    def __init__(self, activations: Dict[str, mx.array]):
        self._activations = activations

    def resid_pre(self, layer_idx: int) -> Optional[mx.array]:
        """Get the input to a layer (residual stream before the layer).

        For layer 0, this is the embedding output.
        For layer i > 0, this equals resid_post(i-1).
        """
        key = resolve_component("resid_pre", layer_idx, self._activations)
        if key is not None:
            return self._activations[key]
        return None

    def resid_post(self, layer_idx: int) -> Optional[mx.array]:
        """Get the output of a layer (residual stream after the layer)."""
        key = resolve_component("resid_post", layer_idx, self._activations)
        if key is not None:
            return self._activations[key]
        # Fall back to the layer output itself
        key = resolve_component("resid_post", layer_idx, self._activations)
        return self._activations.get(key) if key else None

    def attn_contribution(self, layer_idx: int) -> Optional[mx.array]:
        """Get the attention component's output (before residual addition).

        This is the raw output of the self_attn module, NOT resid_pre + attn.
        """
        key = resolve_component("attn", layer_idx, self._activations)
        if key is not None:
            return self._activations[key]
        return None

    def mlp_contribution(self, layer_idx: int) -> Optional[mx.array]:
        """Get the MLP component's output (before residual addition).

        This is the raw output of the MLP module, NOT the residual.
        """
        key = resolve_component("mlp", layer_idx, self._activations)
        if key is not None:
            return self._activations[key]
        return None

    def resid_mid(self, layer_idx: int) -> Optional[mx.array]:
        """Get the residual stream between attention and MLP.

        Computed as: resid_pre + attn_contribution.
        For pre-norm architectures (Llama, Mistral), the attention module
        receives norm(resid_pre) but the residual connection adds the raw
        attention output back to resid_pre.
        """
        pre = self.resid_pre(layer_idx)
        attn = self.attn_contribution(layer_idx)
        if pre is not None and attn is not None:
            # Handle sequence length mismatch
            if pre.shape == attn.shape:
                return pre + attn
            # Try to align by using the last tokens
            min_seq = min(pre.shape[1], attn.shape[1]) if pre.ndim >= 2 else None
            if min_seq is not None:
                return pre[:, -min_seq:, :] + attn[:, -min_seq:, :]
        return None

    def layer_contribution(self, layer_idx: int) -> Optional[mx.array]:
        """Get the total contribution of a layer to the residual stream.

        Computed as: resid_post - resid_pre = attn + mlp (approximately).
        """
        pre = self.resid_pre(layer_idx)
        post = self.resid_post(layer_idx)
        if pre is not None and post is not None:
            if pre.shape == post.shape:
                return post - pre
        return None

    def available_layers(self) -> list:
        """Return list of layer indices that have residual stream data."""
        import re
        layers = set()
        for key in self._activations:
            # Match patterns like "...layers.5..." or "...h.5..."
            for pattern in [r"layers\.(\d+)", r"h\.(\d+)"]:
                m = re.search(pattern, key)
                if m:
                    layers.add(int(m.group(1)))
        return sorted(layers)
