"""
Tokenization utilities for InterpretableModel.

This module provides tokenizer-related methods as a mixin class.
"""

import mlx.core as mx
from typing import Optional, List, Union, Any


class TokenizerMixin:
    """
    Mixin class providing tokenizer-related methods.

    This mixin assumes the class has a `tokenizer` attribute.
    """

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using the model's tokenizer.

        Args:
            text: Text string to encode

        Returns:
            List of token IDs

        Raises:
            ValueError: If no tokenizer is available

        Example:
            >>> tokens = model.encode("Hello world")
            >>> print(tokens)  # [15496, 1917]
        """
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available. Pass a tokenizer to InterpretableModel "
                "or load a model that includes one."
            )

        return self.tokenizer.encode(text)

    def decode(self, tokens: Union[List[int], mx.array]) -> str:
        """
        Decode token IDs to text using the model's tokenizer.

        Args:
            tokens: List of token IDs or mx.array of tokens

        Returns:
            Decoded text string

        Raises:
            ValueError: If no tokenizer is available

        Example:
            >>> text = model.decode([15496, 1917])
            >>> print(text)  # "Hello world"
        """
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available. Pass a tokenizer to InterpretableModel "
                "or load a model that includes one."
            )

        # Convert mx.array to list if needed
        if isinstance(tokens, mx.array):
            tokens = tokens.tolist()

        return self.tokenizer.decode(tokens)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts to token IDs.

        Args:
            texts: List of text strings to encode

        Returns:
            List of token ID lists

        Raises:
            ValueError: If no tokenizer is available

        Example:
            >>> token_lists = model.encode_batch(["Hello", "World"])
            >>> print(token_lists)  # [[15496], [10343]]
        """
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available. Pass a tokenizer to InterpretableModel "
                "or load a model that includes one."
            )

        return [self.tokenizer.encode(text) for text in texts]

    def token_to_str(self, token_id: int) -> str:
        """
        Convert a single token ID to its string representation.

        Args:
            token_id: Token ID to decode

        Returns:
            String representation of the token

        Raises:
            ValueError: If no tokenizer is available

        Example:
            >>> token_str = model.token_to_str(15496)
            >>> print(token_str)  # "Hello"
        """
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available. Pass a tokenizer to InterpretableModel "
                "or load a model that includes one."
            )

        return self.tokenizer.decode([token_id])

    def to_str_tokens(
        self,
        input: Union[str, List[int], mx.array],
        prepend_bos: bool = False,
    ) -> List[str]:
        """
        Convert text or token IDs to a list of token strings.

        This is useful for visualization and analysis where you need
        the string representation of each token.

        Args:
            input: Either a text string or list of token IDs
            prepend_bos: Whether to prepend BOS token (only for string input)

        Returns:
            List of token strings

        Raises:
            ValueError: If no tokenizer is available

        Example:
            >>> # From text
            >>> tokens = model.to_str_tokens("Hello world")
            >>> print(tokens)  # ['Hello', ' world']
            >>>
            >>> # From token IDs
            >>> tokens = model.to_str_tokens([15496, 1917])
            >>> print(tokens)  # ['Hello', ' world']
        """
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available. Pass a tokenizer to InterpretableModel "
                "or load a model that includes one."
            )

        # Convert to token IDs if string
        if isinstance(input, str):
            token_ids = self.tokenizer.encode(input)
            if prepend_bos and hasattr(self.tokenizer, 'bos_token_id'):
                bos_id = self.tokenizer.bos_token_id
                if bos_id is not None and (not token_ids or token_ids[0] != bos_id):
                    token_ids = [bos_id] + token_ids
        elif isinstance(input, mx.array):
            token_ids = input.tolist()
            if isinstance(token_ids[0], list):
                # Handle batched input - use first sequence
                token_ids = token_ids[0]
        else:
            token_ids = list(input)

        # Convert each token to string
        str_tokens = []
        for token_id in token_ids:
            str_tokens.append(self.tokenizer.decode([token_id]))

        return str_tokens

    @property
    def vocab_size(self) -> Optional[int]:
        """
        Get the vocabulary size of the tokenizer.

        Returns:
            Vocabulary size, or None if no tokenizer available

        Example:
            >>> print(model.vocab_size)  # 128256
        """
        if self.tokenizer is None:
            return None

        # Try different attributes depending on tokenizer type
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'n_words'):
            return self.tokenizer.n_words
        elif hasattr(self.tokenizer, '__len__'):
            return len(self.tokenizer)
        else:
            return None
