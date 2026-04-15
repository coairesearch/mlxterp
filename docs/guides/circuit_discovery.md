# Circuit Discovery Guide

## Overview

Circuit discovery identifies the **minimal set of components** (attention heads, MLPs) that are sufficient for a model behavior. This guide covers three techniques with increasing sophistication:

1. **Direct Logit Attribution (DLA)** — Which components contribute to the predicted token?
2. **Path Patching** — Which connections between components matter?
3. **ACDC** — Automatically discover the complete circuit.

## 1. Direct Logit Attribution

DLA decomposes the final logit into per-component contributions. It answers: **how much does each attention layer and MLP contribute to predicting a specific token?**

```python
from mlxterp import InterpretableModel
from mlxterp.causal import direct_logit_attribution

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

result = direct_logit_attribution(
    model,
    text="The capital of France is",
    target_token=None,   # Auto-detect (argmax of output)
    position=-1,         # Analyze last token position
)

print(f"Target: '{result.target_token_str}' (id={result.target_token})")
print(f"Attention contributions per layer: {result.head_contributions.tolist()}")
print(f"MLP contributions per layer: {result.mlp_contributions.tolist()}")
```

### How DLA Works

For each component (attention layer, MLP), DLA:

1. Extracts the component's output (what it added to the residual stream)
2. Projects it through the final layer norm
3. Dots it with the unembedding vector for the target token

The result is how many "logits" each component contributed.

### When to Use DLA

- **Starting point** for any circuit analysis
- Identifying which layers are "writing" the final answer
- Quick: only requires one forward pass

### Limitations

DLA is approximate for pre-norm architectures (Llama, Mistral) because the layer norm is applied to the full residual stream, not to individual components. The contributions may not sum exactly to the final logit.

## 2. Residual Stream Analysis

Before diving into circuit-level analysis, understanding the residual stream is essential:

```python
from mlxterp.causal import ResidualStreamAccessor

with model.trace("The capital of France is") as trace:
    pass

rs = ResidualStreamAccessor(trace.activations)

# What does each layer add to the residual stream?
for layer_idx in range(16):
    contribution = rs.layer_contribution(layer_idx)
    if contribution is not None:
        magnitude = float(mx.sqrt(mx.sum(contribution[0, -1, :] ** 2)))
        print(f"Layer {layer_idx}: contribution magnitude = {magnitude:.4f}")

# Decompose a specific layer
pre = rs.resid_pre(5)           # Input to layer 5
attn = rs.attn_contribution(5)  # What attention added
mlp = rs.mlp_contribution(5)    # What MLP added
mid = rs.resid_mid(5)           # State after attention, before MLP
post = rs.resid_post(5)         # Final output of layer 5
```

## 3. Path Patching

Path patching measures the causal effect of a specific **connection** between two components. It answers: **does head 7.3 communicate to MLP 9?**

### How It Works

1. Run the corrupted input
2. Freeze ALL components to their clean values EXCEPT the sender
3. Measure the effect on the output

If the output changes, the sender's corrupted signal was transmitted to the output via some path.

```python
from mlxterp.causal import path_patching

result = path_patching(
    model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    sender="layers.7.self_attn",     # Source component
    receiver="layers.9.self_attn",   # Target component
    metric="l2",
)

print(f"Path effect: {result.data['effect']:.4f}")
print(f"Components frozen: {result.data['n_frozen']}")
```

### Interpreting Path Patching

- **Large effect**: The sender communicates important information through the network
- **Near-zero effect**: The sender's contribution is not transmitted (or is redundant)

### Scanning Multiple Paths

```python
# Test all attention-to-attention paths across layers
for sender_layer in range(16):
    for receiver_layer in range(sender_layer + 1, 16):
        result = path_patching(
            model, clean, corrupted,
            sender=f"layers.{sender_layer}.self_attn",
            receiver=f"layers.{receiver_layer}.self_attn",
        )
        if abs(result.data['effect']) > 0.05:
            print(f"L{sender_layer}.attn -> L{receiver_layer}.attn: {result.data['effect']:.4f}")
```

## 4. ACDC: Automated Circuit Discovery

ACDC automatically discovers the minimal circuit by testing every component and pruning those below a threshold.

```python
from mlxterp.causal import acdc

circuit = acdc(
    model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    threshold=0.01,               # Minimum effect to keep a node
    components=["attn", "mlp"],   # Component types to test
    layers=range(16),
    verbose=True,
)

print(circuit.summary())
# "Circuit: 8 nodes, 12 edges (threshold=0.01)"

# List the important components
for node in circuit.nodes:
    effect = circuit.data["node_effects"][node]
    print(f"  {node}: effect={effect:.4f}")

# Export for visualization
graph = circuit.to_graph()
```

### Choosing the Threshold

| Threshold | Result |
|-----------|--------|
| `0.001` | Large circuit, includes minor components |
| `0.01` | Balanced — typical starting point |
| `0.05` | Compact circuit, only major components |
| `0.1` | Very sparse, may miss important edges |

### Full Circuit Discovery Workflow

```python
# Step 1: DLA to identify key layers
dla = direct_logit_attribution(model, text)

# Step 2: Activation patching to confirm
patching = activation_patching(model, clean, corrupted, component="attn")

# Step 3: ACDC for automated circuit
circuit = acdc(model, clean, corrupted, threshold=0.01)

# Step 4: Path patching to verify specific connections
for sender in circuit.nodes:
    for receiver in circuit.nodes:
        if sender != receiver:
            pp = path_patching(model, clean, corrupted, sender=sender, receiver=receiver)
            if abs(pp.data['effect']) > 0.01:
                print(f"{sender} -> {receiver}: {pp.data['effect']:.4f}")
```

## 5. Attribution Patching (Fast Approximation)

For quick exploration, attribution patching approximates activation patching using finite differences — ~100x faster.

```python
from mlxterp.causal import attribution_patching

result = attribution_patching(
    model, clean, corrupted,
    component="resid_post",
    metric="l2",
)

# Scores approximate the brute-force patching effects
print(result.attribution_scores.tolist())
```

!!! info "When to Use Each Technique"
    - **DLA**: Quick overview, one forward pass
    - **Attribution patching**: Fast approximation when exploring
    - **Activation patching**: Ground truth for component importance
    - **Path patching**: Understanding connections between components
    - **ACDC**: Comprehensive automated circuit discovery
