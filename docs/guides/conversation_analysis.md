# Conversation-Level Analysis Guide

## Overview

Most interpretability research analyzes single prompts. But LLMs are used in multi-turn conversations where meaning builds across turns, context accumulates, and interventions on early turns cascade through later ones.

mlxterp provides conversation-level analysis tools that let you:

- **Trace full conversations** as a single forward pass
- **Detect turn boundaries** automatically from chat templates
- **Slice activations by turn** to study per-turn representations
- **Measure cross-turn attention** to understand information flow between turns

## Key Concepts

### Turns and Chat Templates

Chat models insert special tokens between turns. For example, Llama 3:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
My name is Alice.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Nice to meet you!<|eot_id|>
```

mlxterp automatically detects these boundaries and separates **content tokens** (the actual message) from **template tokens** (role markers, end-of-turn markers).

### The Turn Dataclass

Each turn tracks its position in the full token sequence:

```python
from mlxterp.conversation import Turn

# Turn.content_start / content_end — just the message text
# Turn.full_start / full_end — including template overhead
# Turn.role — "user", "assistant", "system"
```

## Quick Start

```python
from mlxterp import InterpretableModel

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

conversation = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"},
]

with model.conversation_trace(conversation) as ct:
    # See detected turns
    print(ct.turns)

    # Get activations for the question turn
    question_act = ct.get_turn_activation(2, "layers.10")
    print(f"Question turn activations shape: {question_act.shape}")

    # Cross-turn attention: does the model look back at turn 0?
    cross_attn = ct.cross_turn_attention(layer=10, head=0)
    print(f"Turn 2 -> Turn 0 attention: {float(cross_attn[2, 0]):.4f}")
```

## Turn Detection

### Automatic Detection

`detect_turns()` uses the tokenizer's chat template to find boundaries:

```python
from mlxterp.conversation import detect_turns

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
]

turns = detect_turns(model.tokenizer, messages)

for turn in turns:
    print(f"Turn {turn.index} ({turn.role}): "
          f"content tokens {turn.content_start}:{turn.content_end}, "
          f"full tokens {turn.full_start}:{turn.full_end}")
```

### Supported Templates

| Model Family | Template | Support |
|-------------|----------|---------|
| Llama 3 | `<\|start_header_id\|>...<\|eot_id\|>` | Auto-detected |
| ChatML (Qwen, etc.) | `<\|im_start\|>...<\|im_end\|>` | Auto-detected |
| Gemma | `<start_of_turn>...<end_of_turn>` | Auto-detected |
| Custom | Any with apply_chat_template | Supported |

### Filtering Turns

```python
turns = detect_turns(tokenizer, messages)

# Get all user turns
user_turns = turns.by_role("user")
print(f"User turns: {len(user_turns)}")

# Get assistant turns
assistant_turns = turns.by_role("assistant")

# Slice turns
first_two = turns[0:2]

# List all roles
print(turns.roles)  # ["user", "assistant", "user"]

# Get all content token positions
positions = turns.content_positions()
```

## Conversation Tracing

### Per-Turn Activations

Extract activations for a specific turn only:

```python
with model.conversation_trace(conversation) as ct:
    # Content only (default) — excludes template tokens
    turn0_act = ct.get_turn_activation(0, "layers.5")

    # Full turn — includes role markers and end tokens
    turn0_full = ct.get_turn_activation(0, "layers.5", content_only=False)

    print(f"Content tokens: {turn0_act.shape}")
    print(f"Full turn tokens: {turn0_full.shape}")
```

### Cross-Turn Attention

Measure how much each turn attends to other turns:

```python
with model.conversation_trace(conversation) as ct:
    # (n_turns, n_turns) matrix
    cross_attn = ct.cross_turn_attention(layer=10, head=0)

    # Does turn 2 (the question) attend to turn 0 (the name)?
    print(f"Question -> Name turn: {float(cross_attn[2, 0]):.4f}")
    print(f"Question -> Response turn: {float(cross_attn[2, 1]):.4f}")
```

### Converting to Result

```python
with model.conversation_trace(conversation) as ct:
    result = ct.to_result()

print(result.summary())
print(result.to_json())
```

## Example: Name Recall Analysis

Investigate whether the model remembers a name from an earlier turn:

```python
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

conversation = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What did I say my name was?"},
]

with model.conversation_trace(conversation) as ct:
    print(f"Detected {len(ct.turns)} turns")

    # Which layers attend back to the name?
    for layer in range(16):
        cross = ct.cross_turn_attention(layer=layer, head=0)
        if cross is not None:
            score = float(cross[2, 0])  # Question -> Name turn
            if score > 0.05:
                print(f"Layer {layer}: turn 2 attends to turn 0 with score {score:.4f}")

    # Get the activation at the question turn
    question_act = ct.get_turn_activation(2, "layers.10")
    print(f"Question turn activation shape: {question_act.shape}")
```

## Example: Information Flow Across Turns

```python
conversation = [
    {"role": "user", "content": "The capital of France is Paris."},
    {"role": "assistant", "content": "That's correct!"},
    {"role": "user", "content": "What's the capital of France?"},
]

with model.conversation_trace(conversation) as ct:
    # Check attention flow at every layer
    for layer in [5, 10, 15]:
        for head in range(4):  # Check first 4 heads
            cross = ct.cross_turn_attention(layer=layer, head=head)
            if cross is not None:
                # How much does the question attend to the fact?
                fact_attn = float(cross[2, 0])
                if fact_attn > 0.1:
                    print(f"L{layer}H{head}: question->fact = {fact_attn:.3f}")
```

## API Reference Summary

| Class/Function | Purpose |
|---------------|---------|
| `Turn` | Dataclass for turn boundaries (positions, role) |
| `TurnList` | Container with indexing, slicing, `by_role()` |
| `detect_turns(tokenizer, messages)` | Auto-detect turn boundaries |
| `model.conversation_trace(messages)` | Context manager for multi-turn tracing |
| `ct.get_turn_activation(turn_idx, component)` | Slice activations to a turn |
| `ct.cross_turn_attention(layer, head)` | Turn x turn attention matrix |
| `ct.to_result()` | Convert to `ConversationResult` |
