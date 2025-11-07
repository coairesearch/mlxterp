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
