"""
Text generation with intervention support.

Provides token-by-token generation with optional interventions applied
at each step, enabling study of intervention effects on generated text.
"""

import mlx.core as mx
from typing import Any, Callable, Dict, List, Optional, Union

from .results import GenerationResult


def generate(
    model,
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    interventions: Optional[Dict[str, Callable]] = None,
    stop_tokens: Optional[List[int]] = None,
    callback: Optional[Callable[[int, int, mx.array], bool]] = None,
) -> GenerationResult:
    """
    Generate text token-by-token with optional interventions.

    When interventions are provided, each forward pass goes through the
    tracing system with interventions applied. Without interventions,
    uses direct forward passes for efficiency.

    Args:
        model: InterpretableModel instance
        prompt: Input prompt (text string, token array, or token list)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        top_k: Top-k filtering (0 = no filtering)
        top_p: Nucleus sampling threshold (1.0 = no filtering)
        interventions: Dict mapping component names to intervention functions.
            Applied at every generation step.
        stop_tokens: Token IDs that stop generation. If None, uses
            tokenizer's eos_token_id if available.
        callback: Called after each token: callback(step, token_id, logits).
            Return True to stop generation.

    Returns:
        GenerationResult with generated text, tokens, and metadata
    """
    # Process prompt
    if isinstance(prompt, str):
        if model.tokenizer is None:
            raise ValueError("Tokenizer required for string prompts")
        tokens = model.tokenizer.encode(prompt)
        if isinstance(tokens, list):
            input_ids = mx.array([tokens])
        else:
            input_ids = mx.array([tokens]) if not isinstance(tokens, mx.array) else tokens
        prompt_text = prompt
    elif isinstance(prompt, list):
        input_ids = mx.array([prompt])
        prompt_text = ""
    else:
        input_ids = prompt if prompt.ndim == 2 else prompt.reshape(1, -1)
        prompt_text = ""

    if input_ids.ndim == 1:
        input_ids = input_ids.reshape(1, -1)

    # Set up stop tokens
    if stop_tokens is None:
        stop_tokens = []
        if model.tokenizer is not None:
            eos = getattr(model.tokenizer, 'eos_token_id', None)
            if eos is not None:
                stop_tokens = [eos] if isinstance(eos, int) else list(eos)

    generated_tokens = []
    all_logits = []
    current_ids = input_ids

    for step in range(max_tokens):
        # Forward pass
        if interventions:
            with model.trace(current_ids, interventions=interventions):
                logits = model.output.save()
        else:
            logits = model._forward(current_ids)

        mx.eval(logits)

        # Get logits for last token
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # Sample next token
        next_token = _sample_token(last_logits, temperature, top_k, top_p)
        next_token_id = int(next_token)

        # Callback
        if callback is not None:
            should_stop = callback(step, next_token_id, last_logits)
            if should_stop:
                break

        # Check stop tokens
        if next_token_id in stop_tokens:
            break

        generated_tokens.append(next_token_id)
        all_logits.append(last_logits)

        # Append to input for next step
        current_ids = mx.concatenate(
            [current_ids, mx.array([[next_token_id]])], axis=1
        )

    # Decode generated tokens
    generated_text = ""
    if model.tokenizer is not None and generated_tokens:
        try:
            generated_text = model.tokenizer.decode(generated_tokens)
        except Exception:
            generated_text = str(generated_tokens)

    return GenerationResult(
        data={
            "generated_tokens": generated_tokens,
            "n_tokens": len(generated_tokens),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "has_interventions": interventions is not None,
        },
        metadata={
            "prompt": prompt_text or f"array({input_ids.shape})",
            "max_tokens": max_tokens,
        },
        text=generated_text,
        tokens=generated_tokens,
        token_logits=mx.stack(all_logits) if all_logits else None,
        prompt=prompt_text,
    )


def _sample_token(
    logits: mx.array,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> mx.array:
    """Sample a token from logit distribution.

    Args:
        logits: Raw logits (vocab_size,)
        temperature: Sampling temperature (0 = greedy)
        top_k: Keep only top-k logits (0 = all)
        top_p: Nucleus sampling threshold (1.0 = all)

    Returns:
        Single token ID as mx.array
    """
    if temperature == 0.0:
        return mx.argmax(logits)

    logits = logits.astype(mx.float32)
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[0])
        # Get the k-th largest value
        top_values = mx.sort(logits)[-top_k]
        logits = mx.where(logits < top_values, mx.array(float('-inf')), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = mx.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = mx.softmax(sorted_logits)
        cumulative_probs = mx.cumsum(probs)

        # Find cutoff
        cutoff_mask = cumulative_probs > top_p
        # Keep at least one token
        if mx.any(cutoff_mask):
            cutoff_idx = int(mx.argmax(cutoff_mask.astype(mx.int32)))
            if cutoff_idx > 0:
                # Set everything after cutoff to -inf
                mask = mx.zeros_like(logits)
                for i in range(cutoff_idx):
                    mask = mask.at[sorted_indices[i]].add(1.0)
                logits = mx.where(mask > 0, logits, mx.array(float('-inf')))

    # Sample from distribution
    probs = mx.softmax(logits)
    return mx.random.categorical(mx.log(probs + 1e-10))
