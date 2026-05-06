"""Named hook points on the residual stream.

A transformer's residual stream has three canonical hook points per
layer, named after the order in which a forward pass touches them:

  * ``resid_pre[i]``  — the residual entering layer ``i`` (i.e. the
    output of layer ``i-1``, or the embedding for layer 0).
  * ``resid_mid[i]``  — the residual after attention has added in but
    before the MLP runs (``resid_pre[i] + attn[i]``).
  * ``resid_post[i]`` — the residual leaving layer ``i`` (which equals
    the output of the layer module).

Standard TransformerLens / pyvene mechanism interpretability
analyses lean on these as the canonical points to read or write the
residual stream.

This module exposes a :class:`ResidualStream` view that derives all
three at every layer from the activations a normal ``model.trace``
already captures, without requiring any extra hook points or
intervention machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx


_LAYER_PATH_PREFIXES = ("model.model.layers", "model.layers", "layers")
_EMBED_PATHS = (
    "model.model.embed_tokens",
    "model.embed_tokens",
    "embed_tokens",
    "model.model.wte",
    "model.wte",
    "wte",
)


def _find_one(activations: Dict[str, mx.array], paths) -> Optional[mx.array]:
    for p in paths:
        if p in activations:
            return activations[p]
    return None


@dataclass
class ResidualStream:
    """View over the residual stream at every layer of a single trace.

    Construct via :py:meth:`InterpretableModel.residual_stream`.
    """

    n_layers: int
    layer_outputs: Dict[int, mx.array]   # resid_post[i]
    attn_outputs: Dict[int, mx.array]    # attn contribution at layer i
    mlp_outputs: Dict[int, mx.array]     # mlp contribution at layer i
    embedding: Optional[mx.array]        # embed output (resid_pre[0])

    def pre(self, layer_idx: int) -> mx.array:
        """Residual stream entering layer ``layer_idx``.

        ``resid_pre[0]`` requires the token-embedding output to have
        been captured. If it wasn't (some model wrappings don't
        surface it as a trace hook), a ``RuntimeError`` is raised
        with a hint.
        """
        if layer_idx == 0:
            if self.embedding is None:
                raise RuntimeError(
                    "resid_pre[0] needs the token-embedding output, which "
                    "was not captured in the trace. Either look at "
                    "resid_post[0] (post-layer-0) or pass an "
                    "embedding_path to InterpretableModel(...)."
                )
            return self.embedding
        prev = self.layer_outputs.get(layer_idx - 1)
        if prev is None:
            raise IndexError(
                f"resid_pre[{layer_idx}]: layer {layer_idx - 1} output "
                "was not captured in the trace."
            )
        return prev

    def post(self, layer_idx: int) -> mx.array:
        """Residual stream leaving layer ``layer_idx`` (= layer's output)."""
        out = self.layer_outputs.get(layer_idx)
        if out is None:
            raise IndexError(
                f"resid_post[{layer_idx}]: layer was not captured in the trace."
            )
        return out

    def mid(self, layer_idx: int) -> mx.array:
        """Residual stream after attention and before the MLP at layer
        ``layer_idx`` — the standard "resid_mid" hook point used by
        TransformerLens. Computed as ``resid_pre[i] + attn[i]``.
        """
        attn = self.attn_outputs.get(layer_idx)
        if attn is None:
            raise IndexError(
                f"resid_mid[{layer_idx}]: self_attn output was not "
                "captured in the trace."
            )
        return self.pre(layer_idx) + attn

    def attn_contribution(self, layer_idx: int) -> mx.array:
        """Layer ``layer_idx``'s attention contribution to the residual
        stream (``attn[i]``)."""
        out = self.attn_outputs.get(layer_idx)
        if out is None:
            raise IndexError(
                f"attn_contribution[{layer_idx}]: not captured."
            )
        return out

    def mlp_contribution(self, layer_idx: int) -> mx.array:
        """Layer ``layer_idx``'s MLP contribution to the residual
        stream (``mlp[i]``)."""
        out = self.mlp_outputs.get(layer_idx)
        if out is None:
            raise IndexError(
                f"mlp_contribution[{layer_idx}]: not captured."
            )
        return out

    def accumulated(self) -> List[mx.array]:
        """Per-layer running residual: a list ``[resid_post[0],
        resid_post[1], ...]``. The list is in layer order; element
        ``i`` is the residual stream after layer ``i`` finishes.

        Useful for "what does the residual stream encode at each
        depth" analyses (e.g. logit lens already does this; this gives
        the underlying activations).
        """
        return [self.layer_outputs[i] for i in range(self.n_layers)]

    def decompose(self, layer_idx: Optional[int] = None) -> Dict[str, mx.array]:
        """Decompose the residual stream at layer ``layer_idx`` into
        the embedding and per-component contributions that produced it.

        ``layer_idx`` defaults to ``n_layers - 1`` (the final residual
        stream — the input to the final norm).

        Returns a dict with keys:
          ``"embedding"`` (or absent if not captured),
          ``"attn.{j}"`` for ``j`` in ``0..layer_idx``,
          ``"mlp.{j}"`` for ``j`` in ``0..layer_idx``.

        The values sum to ``self.post(layer_idx)`` exactly (modulo
        float precision). This is the residual-stream identity that
        underlies DLA and path-patching: every component's output is
        a vector that gets *added* to the residual, so the residual
        is literally the sum of those vectors plus the embedding.
        """
        if layer_idx is None:
            layer_idx = self.n_layers - 1

        out: Dict[str, mx.array] = {}
        if self.embedding is not None:
            out["embedding"] = self.embedding
        for j in range(layer_idx + 1):
            attn = self.attn_outputs.get(j)
            if attn is not None:
                out[f"attn.{j}"] = attn
            mlp = self.mlp_outputs.get(j)
            if mlp is not None:
                out[f"mlp.{j}"] = mlp
        return out


def build_residual_stream_from_trace(
    trace_activations: Dict[str, mx.array],
    n_layers: int,
) -> ResidualStream:
    """Construct a :class:`ResidualStream` from the activations
    dictionary of a finished trace context.

    Tries the standard mlx-lm path templates and falls back through
    common alternatives. Any hook point that wasn't captured leaves a
    None in the corresponding accessor; the accessor methods raise
    with a useful message in that case.
    """
    layer_outputs: Dict[int, mx.array] = {}
    attn_outputs: Dict[int, mx.array] = {}
    mlp_outputs: Dict[int, mx.array] = {}

    for i in range(n_layers):
        layer_paths = [f"{p}.{i}" for p in _LAYER_PATH_PREFIXES]
        attn_paths = [f"{p}.{i}.self_attn" for p in _LAYER_PATH_PREFIXES]
        mlp_paths = [f"{p}.{i}.mlp" for p in _LAYER_PATH_PREFIXES]

        v = _find_one(trace_activations, layer_paths)
        if v is not None:
            layer_outputs[i] = v
        v = _find_one(trace_activations, attn_paths)
        if v is not None:
            attn_outputs[i] = v
        v = _find_one(trace_activations, mlp_paths)
        if v is not None:
            mlp_outputs[i] = v

    embedding = _find_one(trace_activations, _EMBED_PATHS)

    return ResidualStream(
        n_layers=n_layers,
        layer_outputs=layer_outputs,
        attn_outputs=attn_outputs,
        mlp_outputs=mlp_outputs,
        embedding=embedding,
    )
