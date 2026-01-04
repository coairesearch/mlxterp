#!/usr/bin/env python3
"""
ROME-style causal tracing runner (minimal, no library changes).

This script focuses on the paper's methodology:
- Gaussian noise corruption on subject embeddings
- Residual stream patching (full layer output)
- Position-specific patching at the subject's last token
- Target-token probability recovery metric
- Averaging across multiple factual prompts
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import mlx.core as mx
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


@dataclass
class FactExample:
    subject: str
    object: str
    prompt: str


DEFAULT_FACTS = [
    FactExample(subject="France", object="Paris", prompt="The capital of France is"),
    FactExample(subject="Germany", object="Berlin", prompt="The capital of Germany is"),
    FactExample(subject="Italy", object="Rome", prompt="The capital of Italy is"),
    FactExample(subject="Spain", object="Madrid", prompt="The capital of Spain is"),
    FactExample(subject="Japan", object="Tokyo", prompt="The capital of Japan is"),
    FactExample(subject="Canada", object="Ottawa", prompt="The capital of Canada is"),
    FactExample(subject="Brazil", object="Brasilia", prompt="The capital of Brazil is"),
    FactExample(subject="Australia", object="Canberra", prompt="The capital of Australia is"),
    FactExample(subject="Greece", object="Athens", prompt="The capital of Greece is"),
    FactExample(subject="Egypt", object="Cairo", prompt="The capital of Egypt is"),
]


def facts_from_record(data: Dict[str, str]) -> FactExample:
    prompt = data.get("prompt")
    if prompt is None:
        template = data.get("template")
        if template is None:
            raise ValueError("Each fact must have 'prompt' or 'template'.")
        prompt = template.format(subject=data["subject"])

    obj = data.get("object", data.get("attribute"))
    if obj is None:
        raise ValueError("Each fact must have 'object' or 'attribute'.")

    return FactExample(subject=data["subject"], object=obj, prompt=prompt)


def load_facts(path: Optional[str]) -> List[FactExample]:
    if path is None:
        return list(DEFAULT_FACTS)

    with open(path, "r", encoding="utf-8") as handle:
        contents = handle.read().strip()

    if not contents:
        return []

    facts: List[FactExample] = []
    if contents[0] in ("[", "{"):
        parsed: Union[List[Dict[str, str]], Dict[str, Dict[str, str]]] = json.loads(contents)
        if isinstance(parsed, list):
            for record in parsed:
                facts.append(facts_from_record(record))
        elif isinstance(parsed, dict):
            for record in parsed.values():
                facts.append(facts_from_record(record))
    else:
        for line in contents.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            facts.append(facts_from_record(record))

    return facts


def find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start:start + len(needle)] == needle:
            return start
    return None


def find_subject_span(model: InterpretableModel, prompt: str, subject: str) -> Optional[Tuple[int, int]]:
    prompt_tokens = model.encode(prompt)

    candidates = [subject]
    if not subject.startswith(" "):
        candidates.append(" " + subject)

    for candidate in candidates:
        subj_tokens = model.encode(candidate)
        start = find_subsequence(prompt_tokens, subj_tokens)
        if start is not None:
            end = start + len(subj_tokens)
            return start, end
    return None


def join_prompt_target(prompt: str, target: str) -> str:
    if prompt.endswith((" ", "\n", "\t")):
        return prompt + target
    return prompt + " " + target


def get_target_token_id(model: InterpretableModel, prompt: str, target: str) -> Optional[int]:
    prompt_tokens = model.encode(prompt)
    full_tokens = model.encode(join_prompt_target(prompt, target))
    if len(full_tokens) <= len(prompt_tokens):
        return None
    return full_tokens[len(prompt_tokens)]


def get_target_probability(logits: mx.array, target_id: int) -> float:
    if len(logits.shape) == 3:
        logits = logits[0, -1, :]
    elif len(logits.shape) == 2:
        logits = logits[-1, :]
    probs = mx.softmax(logits)
    return float(probs[target_id])


def compute_embedding_noise_std(model: InterpretableModel, multiplier: float = 3.0) -> float:
    embed_layer = model._module_resolver.get_embedding_layer()

    if hasattr(embed_layer, "scales"):
        return 0.13

    if hasattr(embed_layer, "weight"):
        weights = embed_layer.weight
        mx.eval(weights)
        embedding_std = float(mx.std(weights))
        if embedding_std > 0:
            return max(0.05, min(0.3, multiplier * embedding_std))

    return 0.13


def add_gaussian_noise_to_embedding(
    model: InterpretableModel,
    tokens: List[int],
    noise_std: float,
    subject_range: Tuple[int, int],
    seed: int,
) -> mx.array:
    embed_layer = model._module_resolver.get_embedding_layer()

    tokens_array = mx.array([tokens])
    if hasattr(embed_layer, "scales"):
        original_embeddings = mx.dequantize(
            embed_layer.weight,
            embed_layer.scales,
            embed_layer.biases,
            embed_layer.group_size,
            embed_layer.bits,
        )
        embeddings = original_embeddings[tokens_array]
    else:
        embeddings = embed_layer(tokens_array)

    mx.eval(embeddings)
    mx.random.seed(seed)
    noise = mx.random.normal(embeddings.shape) * noise_std

    start, end = subject_range
    seq_len = embeddings.shape[1]
    positions = mx.arange(seq_len)
    mask = ((positions >= start) & (positions < end)).reshape(1, seq_len, 1).astype(embeddings.dtype)
    noise = noise * mask

    noisy_embeddings = embeddings + noise
    mx.eval(noisy_embeddings)
    return noisy_embeddings


def replace_at_position(clean_activation: mx.array, position: int):
    def _replace(x: mx.array) -> mx.array:
        if x.shape[1] <= position or clean_activation.shape[1] <= position:
            return x
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)
        mask = (positions == position).reshape(1, seq_len, 1).astype(x.dtype)
        return x * (1 - mask) + clean_activation * mask

    return _replace


def detect_embedding_key(model: InterpretableModel, tokens: List[int]) -> str:
    embed_key = model._module_resolver.get_embedding_path()
    if embed_key is not None:
        return embed_key

    for path in ["model.embed_tokens", "model.wte", "model.model.wte"]:
        with model.trace(mx.array([tokens])) as trace:
            pass
        if path in trace.activations:
            return path

    return "model.embed_tokens"


def extract_layer_keys(trace, n_layers: int) -> Dict[int, str]:
    layer_keys: Dict[int, str] = {}
    for layer_idx in range(n_layers):
        for key in trace.activations:
            for prefix in (f"h.{layer_idx}", f"layers.{layer_idx}"):
                if prefix in key:
                    suffix = key.split(prefix)[-1]
                    if suffix == "":
                        layer_keys[layer_idx] = key
                        break
            if layer_idx in layer_keys:
                break
    return layer_keys


def strip_model_prefix(key: str) -> str:
    if key.startswith("model.model."):
        return key[12:]
    if key.startswith("model."):
        return key[6:]
    return key


def run_example(
    model: InterpretableModel,
    example: FactExample,
    noise_std: float,
    min_drop: float,
    seed: int,
) -> Optional[Dict[int, float]]:
    tokens = model.encode(example.prompt)
    subject_span = find_subject_span(model, example.prompt, example.subject)
    if subject_span is None:
        print(f"  Skip: subject span not found for '{example.subject}'")
        return None

    target_id = get_target_token_id(model, example.prompt, example.object)
    if target_id is None:
        print(f"  Skip: target token not found for '{example.object}'")
        return None

    subject_start, subject_end = subject_span
    subject_last_pos = subject_end - 1

    tokens_array = mx.array([tokens])

    with model.trace(tokens_array) as clean_trace:
        clean_output = model.output.save()

    mx.eval(clean_output)
    clean_prob = get_target_probability(clean_output, target_id)

    noisy_embeddings = add_gaussian_noise_to_embedding(
        model,
        tokens,
        noise_std=noise_std,
        subject_range=(subject_start, subject_end),
        seed=seed,
    )

    embed_key = detect_embedding_key(model, tokens)

    with model.trace(tokens_array, interventions={embed_key: iv.replace_with(noisy_embeddings)}):
        corrupted_output = model.output.save()
    mx.eval(corrupted_output)
    corrupted_prob = get_target_probability(corrupted_output, target_id)

    drop = clean_prob - corrupted_prob
    if drop < min_drop:
        print(f"  Skip: weak corruption (drop {drop:.4f}) for '{example.prompt}'")
        return None

    n_layers = len(model.layers)
    layer_keys = extract_layer_keys(clean_trace, n_layers)
    clean_layers = {idx: clean_trace.activations[key] for idx, key in layer_keys.items()}
    for value in clean_layers.values():
        mx.eval(value)

    results: Dict[int, float] = {}
    for layer_idx in range(n_layers):
        layer_key = layer_keys.get(layer_idx)
        if layer_key is None:
            continue
        intervention_key = strip_model_prefix(layer_key)
        clean_layer = clean_layers[layer_idx]

        with model.trace(
            tokens_array,
            interventions={
                embed_key: iv.replace_with(noisy_embeddings),
                intervention_key: replace_at_position(clean_layer, subject_last_pos),
            },
        ):
            patched_output = model.output.save()
        mx.eval(patched_output)
        patched_prob = get_target_probability(patched_output, target_id)

        if drop > 1e-6:
            recovery = (patched_prob - corrupted_prob) / drop * 100
        else:
            recovery = 0.0
        results[layer_idx] = recovery

    return results


def aggregate_results(all_results: Iterable[Dict[int, float]], n_layers: int) -> Dict[int, float]:
    bucket: Dict[int, List[float]] = {i: [] for i in range(n_layers)}
    for result in all_results:
        for layer, value in result.items():
            bucket[layer].append(value)
    return {
        layer: (sum(values) / len(values)) if values else 0.0
        for layer, values in bucket.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ROME-style causal tracing runner.")
    parser.add_argument("--model", default="MCES10/gpt2-xl-mlx-fp16")
    parser.add_argument("--facts", default=None, help="Path to JSON/JSONL facts file.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--noise-multiplier", type=float, default=3.0)
    parser.add_argument("--min-drop", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = InterpretableModel(args.model)
    print(f"Model loaded: {len(model.layers)} layers")

    facts = load_facts(args.facts)
    print(f"Loaded {len(facts)} facts")

    if args.noise_std is None:
        noise_std = compute_embedding_noise_std(model, multiplier=args.noise_multiplier)
    else:
        noise_std = args.noise_std
    print(f"Noise std: {noise_std:.4f}")

    results = []
    if args.limit is not None:
        facts = facts[: args.limit]

    for idx, fact in enumerate(facts, start=1):
        print(f"[{idx}/{len(facts)}] {fact.prompt}")
        run = run_example(
            model=model,
            example=fact,
            noise_std=noise_std,
            min_drop=args.min_drop,
            seed=args.seed,
        )
        if run is not None:
            results.append(run)

    if not results:
        print("No valid results (all examples skipped).")
        return

    mean_recovery = aggregate_results(results, n_layers=len(model.layers))
    peak_layer = max(mean_recovery, key=mean_recovery.get)

    print("\nMean recovery by layer (top 10):")
    sorted_layers = sorted(mean_recovery.items(), key=lambda x: x[1], reverse=True)
    for layer, recovery in sorted_layers[:10]:
        print(f"  Layer {layer:2d}: {recovery:6.2f}%")

    print(f"\nPeak layer: {peak_layer} ({mean_recovery[peak_layer]:.2f}%)")


if __name__ == "__main__":
    main()
