"""
ConversationTrace: Multi-turn conversation tracing.

Traces a full multi-turn conversation as a single forward pass
and provides per-turn activation slicing.
"""

import mlx.core as mx
from typing import Any, Dict, List, Optional, Union

from ..results import ConversationResult
from .turns import Turn, TurnList, detect_turns


class ConversationTrace:
    """
    Context manager for conversation-level analysis.

    Applies the chat template, runs a single forward pass on the full
    token sequence, then provides per-turn activation slicing.

    Example:
        conversation = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you!"},
            {"role": "user", "content": "What's my name?"},
        ]
        with model.conversation_trace(conversation) as ct:
            turn1_act = ct.get_turn_activation(0, "layers.5")
            cross_attn = ct.cross_turn_attention(layer=5, head=0)

    Attributes:
        turns: TurnList of detected turns
        activations: Full activation dict from the trace
        output: Model output
    """

    def __init__(self, model, messages: List[Dict[str, str]]):
        self._model = model
        self._messages = messages
        self.turns = None
        self.activations = {}
        self.output = None
        self._token_ids = None

    def __enter__(self):
        if self._model.tokenizer is None:
            raise ValueError(
                "ConversationTrace requires a tokenizer. "
                "Load a model with a tokenizer."
            )

        # Apply chat template and tokenize
        tokenizer = self._model.tokenizer
        if hasattr(tokenizer, 'apply_chat_template'):
            self._token_ids = tokenizer.apply_chat_template(
                self._messages, tokenize=True, add_generation_prompt=False
            )
        else:
            # Fallback: concatenate encodings
            self._token_ids = []
            for msg in self._messages:
                encoded = tokenizer.encode(msg["content"])
                self._token_ids.extend(
                    encoded if isinstance(encoded, list) else encoded.tolist()
                )

        # Detect turns
        self.turns = detect_turns(tokenizer, self._messages, self._token_ids)

        # Run trace on full sequence
        input_ids = mx.array([self._token_ids])

        with self._model.trace(input_ids) as trace:
            self.output = self._model.output.save()

        mx.eval(self.output)
        self.activations = dict(trace.activations)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def get_turn_activation(
        self,
        turn_idx: int,
        component: str,
        content_only: bool = True,
    ) -> Optional[mx.array]:
        """
        Get activations sliced to a specific turn.

        Args:
            turn_idx: Turn index
            component: Activation key (e.g., "layers.5" or full key)
            content_only: If True, only content tokens; else full turn

        Returns:
            Activation tensor sliced to the turn's positions
        """
        turn = self.turns[turn_idx]
        pos = turn.content_positions if content_only else turn.full_positions

        # Find activation key
        act = self._find_activation(component)
        if act is None:
            return None

        # Slice to turn positions
        if act.ndim == 3:
            return act[:, pos, :]
        elif act.ndim == 2:
            return act[pos, :]
        return act

    def cross_turn_attention(
        self,
        layer: int,
        head: int = 0,
    ) -> Optional[mx.array]:
        """
        Compute cross-turn attention matrix.

        Aggregates token-level attention to turn-level by averaging
        attention weights within each turn pair.

        Args:
            layer: Layer index
            head: Attention head index

        Returns:
            (n_turns, n_turns) matrix of aggregated attention
        """
        if self.turns is None or len(self.turns) == 0:
            return None

        # Find attention weights
        attn_key = None
        for key in self.activations:
            if f"layers.{layer}" in key and "attention_weights" in key:
                attn_key = key
                break
            if f"h.{layer}" in key and "attention_weights" in key:
                attn_key = key
                break

        if attn_key is None:
            return None

        weights = self.activations[attn_key]  # (batch, n_heads, seq, seq)
        if weights.ndim != 4:
            return None

        head_weights = weights[0, head, :, :]  # (seq, seq)
        n_turns = len(self.turns)
        cross_attn = mx.zeros((n_turns, n_turns))

        for i, target_turn in enumerate(self.turns):
            for j, source_turn in enumerate(self.turns):
                t_pos = target_turn.content_positions
                s_pos = source_turn.content_positions

                # Average attention from target to source
                t_range = range(t_pos.start, min(t_pos.stop, head_weights.shape[0]))
                s_range = range(s_pos.start, min(s_pos.stop, head_weights.shape[1]))

                if len(t_range) == 0 or len(s_range) == 0:
                    continue

                total = 0.0
                count = 0
                for ti in t_range:
                    for si in s_range:
                        total += float(head_weights[ti, si])
                        count += 1

                if count > 0:
                    cross_attn[i, j] = total / count

        return cross_attn

    def to_result(self) -> ConversationResult:
        """Convert to a ConversationResult."""
        turn_data = []
        if self.turns:
            for t in self.turns:
                turn_data.append({
                    "index": t.index,
                    "role": t.role,
                    "content_start": t.content_start,
                    "content_end": t.content_end,
                    "full_start": t.full_start,
                    "full_end": t.full_end,
                })

        return ConversationResult(
            data={"n_turns": len(turn_data), "messages": self._messages},
            metadata={"n_tokens": len(self._token_ids) if self._token_ids else 0},
            turns=turn_data,
        )

    def _find_activation(self, component: str) -> Optional[mx.array]:
        """Find an activation by component name with prefix matching."""
        for prefix in ["model.model.", "model.", ""]:
            key = f"{prefix}{component}"
            if key in self.activations:
                return self.activations[key]
        # Try partial match
        for key in self.activations:
            if component in key:
                return self.activations[key]
        return None
