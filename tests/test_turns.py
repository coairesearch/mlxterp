"""Tests for mlxterp.conversation.turns module."""

import pytest
from mlxterp.conversation.turns import Turn, TurnList, detect_turns, _find_subsequence


class MockTokenizer:
    """Mock tokenizer for testing turn detection."""

    def __init__(self):
        self.bos_token_id = 1
        self.eos_token_id = 2
        # Simple vocab: each character gets an ID
        self._vocab = {}
        self._next_id = 10

    def encode(self, text):
        """Encode text to token IDs (one per word for simplicity)."""
        tokens = [self.bos_token_id]
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = self._next_id
                self._next_id += 1
            tokens.append(self._vocab[word])
        return tokens

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        """Simple chat template: [BOS] role: content [EOS] ..."""
        tokens = [self.bos_token_id]
        for msg in messages:
            # Add role marker (3=user, 4=assistant, 5=system)
            role_id = {"user": 3, "assistant": 4, "system": 5}.get(msg["role"], 6)
            tokens.append(role_id)
            # Add content tokens
            content = self.encode(msg["content"])
            tokens.extend(content[1:])  # skip BOS
            tokens.append(self.eos_token_id)
        return tokens


class TestTurn:
    def test_basic(self):
        turn = Turn(index=0, role="user", full_start=0, full_end=10,
                   content_start=2, content_end=8)
        assert turn.index == 0
        assert turn.role == "user"

    def test_content_positions(self):
        turn = Turn(index=0, role="user", full_start=0, full_end=10,
                   content_start=2, content_end=8)
        s = turn.content_positions
        assert s == slice(2, 8)

    def test_full_positions(self):
        turn = Turn(index=0, role="user", full_start=0, full_end=10,
                   content_start=2, content_end=8)
        s = turn.full_positions
        assert s == slice(0, 10)

    def test_n_tokens(self):
        turn = Turn(index=0, role="user", full_start=0, full_end=10,
                   content_start=2, content_end=8)
        assert turn.n_content_tokens == 6
        assert turn.n_total_tokens == 10


class TestTurnList:
    @pytest.fixture
    def turns(self):
        return TurnList([
            Turn(0, "user", 0, 10, 2, 8),
            Turn(1, "assistant", 10, 20, 12, 18),
            Turn(2, "user", 20, 30, 22, 28),
        ])

    def test_len(self, turns):
        assert len(turns) == 3

    def test_getitem(self, turns):
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    def test_slice(self, turns):
        sliced = turns[0:2]
        assert isinstance(sliced, TurnList)
        assert len(sliced) == 2

    def test_iter(self, turns):
        roles = [t.role for t in turns]
        assert roles == ["user", "assistant", "user"]

    def test_by_role(self, turns):
        users = turns.by_role("user")
        assert len(users) == 2
        assert all(t.role == "user" for t in users)

    def test_by_role_assistant(self, turns):
        assistants = turns.by_role("assistant")
        assert len(assistants) == 1
        assert assistants[0].index == 1

    def test_roles(self, turns):
        assert turns.roles == ["user", "assistant", "user"]

    def test_content_positions(self, turns):
        positions = turns.content_positions()
        assert 2 in positions
        assert 12 in positions
        assert 22 in positions

    def test_repr(self, turns):
        r = repr(turns)
        assert "Turn(0" in r
        assert "user" in r

    def test_invalid_index(self, turns):
        with pytest.raises(TypeError):
            turns["invalid"]


class TestDetectTurns:
    def test_basic_detection(self):
        tokenizer = MockTokenizer()
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        turns = detect_turns(tokenizer, messages)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    def test_three_turns(self):
        tokenizer = MockTokenizer()
        messages = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Nice to meet you"},
            {"role": "user", "content": "What is my name"},
        ]
        turns = detect_turns(tokenizer, messages)
        assert len(turns) == 3

    def test_content_start_positive(self):
        tokenizer = MockTokenizer()
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        turns = detect_turns(tokenizer, messages)
        assert turns[0].content_start >= 0

    def test_pre_tokenized(self):
        tokenizer = MockTokenizer()
        messages = [{"role": "user", "content": "Test message"}]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        turns = detect_turns(tokenizer, messages, token_ids=token_ids)
        assert len(turns) == 1


class TestFindSubsequence:
    def test_found(self):
        assert _find_subsequence([1, 2, 3, 4, 5], [3, 4]) == 2

    def test_not_found(self):
        assert _find_subsequence([1, 2, 3], [4, 5]) == -1

    def test_start_offset(self):
        assert _find_subsequence([1, 2, 3, 2, 3], [2, 3], start=2) == 3

    def test_empty_subsequence(self):
        assert _find_subsequence([1, 2, 3], []) == 0

    def test_at_beginning(self):
        assert _find_subsequence([1, 2, 3], [1, 2]) == 0

    def test_at_end(self):
        assert _find_subsequence([1, 2, 3], [2, 3]) == 1
