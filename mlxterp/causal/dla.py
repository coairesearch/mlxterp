"""
Direct Logit Attribution (DLA).

Decomposes the final logit into per-component (per-head, per-MLP) contributions
by projecting each component's output through the unembedding matrix.
"""

import mlx.core as mx
from typing import Dict, List, Optional, Union

from ..core.module_resolver import resolve_component
from ..results import DLAResult
from .residual import ResidualStreamAccessor


def direct_logit_attribution(
    model,
    text: Union[str, mx.array],
    target_token: Optional[int] = None,
    position: int = -1,
    layers: Optional[List[int]] = None,
    verbose: bool = False,
) -> DLAResult:
    """
    Compute Direct Logit Attribution for each component.

    Projects each layer's attention and MLP contributions through the
    final norm and unembedding matrix to determine their effect on
    the target token's logit.

    Args:
        model: InterpretableModel instance
        text: Input text or token array
        target_token: Token ID to attribute. If None, uses the argmax prediction.
        position: Token position to analyze (-1 = last token)
        layers: Layer indices to analyze. None = all layers.
        verbose: Print progress

    Returns:
        DLAResult with per-head and per-MLP contributions
    """
    # Run forward pass
    with model.trace(text) as trace:
        output = model.output.save()

    mx.eval(output)

    # Determine target token
    if target_token is None:
        if output.ndim == 3:
            target_token = int(mx.argmax(output[0, position, :]))
        else:
            target_token = int(mx.argmax(output[0]))

    # Get target token string
    target_str = ""
    if model.tokenizer is not None:
        try:
            target_str = model.tokenizer.decode([target_token])
        except Exception:
            pass

    # Get unembedding direction
    resolver = model._module_resolver
    output_proj, _, is_tied = resolver.get_output_projection()

    if output_proj is None:
        raise ValueError("Could not find output projection (lm_head) in model")

    # Get the unembedding vector for the target token
    if is_tied:
        # Weight-tied: use embedding weights
        embed = resolver.get_embedding_layer()
        if hasattr(embed, 'weight'):
            w_u = embed.weight[target_token]  # (hidden_dim,)
        else:
            raise ValueError("Cannot extract embedding weights for DLA")
    else:
        if hasattr(output_proj, 'weight'):
            w_u = output_proj.weight[target_token]  # (hidden_dim,)
        else:
            raise ValueError("Cannot extract lm_head weights for DLA")

    # Get final norm
    final_norm = resolver.get_final_norm()

    # Setup
    if layers is None:
        layers = list(range(len(model.layers)))

    rs = ResidualStreamAccessor(trace.activations)
    mlp_contributions = []
    attn_contributions = []

    for layer_idx in layers:
        if verbose:
            print(f"  DLA layer {layer_idx}...", end="\r")

        # Get component contributions
        attn_out = rs.attn_contribution(layer_idx)
        mlp_out = rs.mlp_contribution(layer_idx)

        # Project through norm + unembedding
        attn_logit = _project_to_logit(attn_out, final_norm, w_u, position)
        mlp_logit = _project_to_logit(mlp_out, final_norm, w_u, position)

        attn_contributions.append(attn_logit)
        mlp_contributions.append(mlp_logit)

    if verbose:
        print()

    attn_array = mx.array(attn_contributions)
    mlp_array = mx.array(mlp_contributions)

    return DLAResult(
        data={
            "layers": layers,
            "attn_contributions": attn_contributions,
            "mlp_contributions": mlp_contributions,
            "target_token": target_token,
            "target_token_str": target_str,
            "position": position,
        },
        metadata={
            "text": text if isinstance(text, str) else f"array({text.shape})",
            "n_layers": len(layers),
        },
        head_contributions=attn_array,
        mlp_contributions=mlp_array,
        target_token=target_token,
        target_token_str=target_str,
    )


def _project_to_logit(
    component_output: Optional[mx.array],
    final_norm: Optional[object],
    w_u: mx.array,
    position: int,
) -> float:
    """Project a component's output through norm + unembedding to get logit contribution.

    This is an approximation: we apply the final norm to the component output
    in isolation, then dot with the unembedding vector. This is exact when the
    norm is linear (identity) and approximate for RMSNorm/LayerNorm.

    For more accurate DLA, the full residual stream context would be needed.
    """
    if component_output is None:
        return 0.0

    # Extract the token at the target position
    if component_output.ndim == 3:
        vec = component_output[0, position, :]  # (hidden_dim,)
    elif component_output.ndim == 2:
        vec = component_output[position, :]
    else:
        vec = component_output

    vec = vec.astype(mx.float32)
    w_u_f32 = w_u.astype(mx.float32)

    # Apply norm if available (approximate: norm of component alone)
    if final_norm is not None:
        try:
            # RMSNorm/LayerNorm expect (batch, seq, dim) or (seq, dim)
            normed = final_norm(vec.reshape(1, 1, -1))
            vec = normed.reshape(-1)
        except Exception:
            pass  # Skip norm if it fails

    # Dot product with unembedding
    logit = float(mx.sum(vec * w_u_f32))
    return logit
