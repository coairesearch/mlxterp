"""
Turn detection and management for multi-turn conversations.

Identifies turn boundaries in tokenized conversations by detecting
chat template special tokens (Llama 3, ChatML, Gemma, etc.).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Turn:
    """A single turn in a conversation.

    Tracks both the full token range (including template tokens) and
    the content-only range (excluding role markers and end tokens).

    Attributes:
        index: Turn number (0-indexed)
        role: Speaker role ("user", "assistant", "system")
        full_start: First token position (including template tokens)
        full_end: Last token position (exclusive, including template tokens)
        content_start: First content token (excluding role markers)
        content_end: Last content token (exclusive, excluding eot markers)
    """

    index: int
    role: str
    full_start: int
    full_end: int
    content_start: int
    content_end: int

    @property
    def content_positions(self) -> slice:
        """Slice for content-only token positions."""
        return slice(self.content_start, self.content_end)

    @property
    def full_positions(self) -> slice:
        """Slice for all token positions (including template)."""
        return slice(self.full_start, self.full_end)

    @property
    def n_content_tokens(self) -> int:
        """Number of content tokens."""
        return self.content_end - self.content_start

    @property
    def n_total_tokens(self) -> int:
        """Total number of tokens including template."""
        return self.full_end - self.full_start


class TurnList:
    """Container for conversation turns with indexing and filtering.

    Supports integer indexing, slicing, role-based filtering,
    and iteration.

    Example:
        turns = TurnList([turn0, turn1, turn2])
        turns[0]                      # First turn
        turns[0:2]                    # First two turns
        turns.by_role("user")         # All user turns
        turns.by_role("assistant")    # All assistant turns
    """

    def __init__(self, turns: List[Turn]):
        self._turns = turns

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._turns[key]
        elif isinstance(key, slice):
            return TurnList(self._turns[key])
        raise TypeError(f"Invalid index type: {type(key)}")

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self):
        return iter(self._turns)

    def __repr__(self) -> str:
        entries = []
        for t in self._turns:
            entries.append(
                f"Turn({t.index}, role={t.role!r}, "
                f"content={t.content_start}:{t.content_end}, "
                f"full={t.full_start}:{t.full_end})"
            )
        return "[" + ",\n ".join(entries) + "]"

    def by_role(self, role: str) -> "TurnList":
        """Filter turns by speaker role.

        Args:
            role: "user", "assistant", or "system"

        Returns:
            TurnList containing only turns with matching role
        """
        return TurnList([t for t in self._turns if t.role == role])

    @property
    def roles(self) -> List[str]:
        """List of roles in conversation order."""
        return [t.role for t in self._turns]

    def content_positions(self) -> List[int]:
        """All content token positions across all turns."""
        positions = []
        for t in self._turns:
            positions.extend(range(t.content_start, t.content_end))
        return positions


def detect_turns(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    token_ids: Optional[List[int]] = None,
) -> TurnList:
    """
    Detect turn boundaries in a tokenized conversation.

    Uses the tokenizer's chat template to tokenize the full conversation,
    then locates each turn's content within the full sequence.

    Args:
        tokenizer: HuggingFace-compatible tokenizer with apply_chat_template
        messages: List of {"role": str, "content": str} dicts
        token_ids: Pre-tokenized IDs (if None, applies chat template)

    Returns:
        TurnList with detected turn boundaries
    """
    # Get full token sequence
    if token_ids is None:
        if hasattr(tokenizer, 'apply_chat_template'):
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        else:
            # Fallback: concatenate individual encodings
            token_ids = []
            for msg in messages:
                encoded = tokenizer.encode(msg["content"])
                token_ids.extend(encoded if isinstance(encoded, list) else encoded.tolist())

    # Tokenize each message individually to find boundaries
    turns = []
    search_start = 0

    for idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        # Encode just the content
        content_tokens = tokenizer.encode(content)
        if not isinstance(content_tokens, list):
            content_tokens = content_tokens.tolist()

        # Remove BOS token if present (tokenizers often prepend it)
        if content_tokens and hasattr(tokenizer, 'bos_token_id'):
            if content_tokens[0] == tokenizer.bos_token_id:
                content_tokens = content_tokens[1:]

        # Find content tokens in the full sequence
        content_start = _find_subsequence(token_ids, content_tokens, search_start)

        if content_start == -1:
            # Try without leading/trailing whitespace tokens
            if len(content_tokens) > 1:
                content_start = _find_subsequence(
                    token_ids, content_tokens[1:], search_start
                )
                if content_start != -1:
                    content_tokens = content_tokens[1:]

        if content_start == -1:
            # Couldn't find exact match — use heuristic
            content_start = search_start + 3  # skip template overhead

        content_end = content_start + len(content_tokens)

        # Full turn boundaries include template tokens
        full_start = search_start if idx == 0 else (
            turns[-1].full_end if turns else 0
        )
        full_end = content_end + 1 if content_end < len(token_ids) else content_end

        # Look for EOT token after content
        eot_candidates = [
            getattr(tokenizer, 'eos_token_id', None),
        ]
        for pos in range(content_end, min(content_end + 5, len(token_ids))):
            if token_ids[pos] in [t for t in eot_candidates if t is not None]:
                full_end = pos + 1
                break

        turns.append(Turn(
            index=idx,
            role=role,
            full_start=full_start,
            full_end=full_end,
            content_start=content_start,
            content_end=content_end,
        ))

        search_start = full_end

    return TurnList(turns)


def _find_subsequence(
    sequence: List[int], subsequence: List[int], start: int = 0
) -> int:
    """Find the start index of a subsequence within a sequence.

    Returns -1 if not found.
    """
    if not subsequence:
        return start

    n = len(sequence)
    m = len(subsequence)

    for i in range(start, n - m + 1):
        if sequence[i:i + m] == subsequence[:m]:
            return i

    return -1
