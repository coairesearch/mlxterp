# mlxterp Causal Interpretability Roadmap

Gap analysis and implementation plan for making mlxterp a competitive mechanistic interpretability library, with focus on causal interpretation/intervention, conversation-level analysis, and agentic interpretability workflows on Apple Silicon.

**Date**: 2026-04-14
**Benchmark libraries**: TransformerLens, nnsight, pyvene, BauLab/Baukit, SAELens
**Vision**: The only library that combines native Apple Silicon interpretability with frontier-model-driven autonomous research — mlxterp as both the microscope *and* the researcher's assistant.

---

## Current State

### What mlxterp does well

| Capability | Status | Notes |
|---|---|---|
| Context-manager tracing (~196 activations) | Done | Composition-based wrapping, any MLX model |
| 7 intervention types | Done | zero_out, scale, add_vector, replace_with, clamp, noise, compose |
| Proxy-based attribute access | Done | `model.layers[3].attn.output.save()` |
| Model-agnostic wrapping | Done | No model-specific subclasses needed |
| Logit lens + tuned lens | Done | With visualization |
| SAE training (TopK, BatchTopK) | Done | SAELens-validated defaults, W&B integration |
| Attention visualization + pattern detection | Done | Matplotlib, Plotly, interactive HTML |
| Tokenization utilities | Done | Encode, decode, str_tokens, batch support |
| Activation caching | Done | Smart key normalization, collect_activations() |
| Comprehensive test suite | Done | 36 test files, ~5,300 lines |

### What's missing

Three core gaps:

1. **Single-pass thinking**: mlxterp operates on individual forward passes. Causal interpretability requires **comparative analysis** (clean vs corrupted inputs).
2. **Token-level only**: Analysis targets individual prompts. Real LLM usage involves **multi-turn conversations** where context accumulates across turns via KV-cache. No way to study how turn 3 causally depends on turn 1.
3. **Human-in-the-loop only**: Every analysis step requires manual orchestration. No way for a frontier model (Claude, etc.) to autonomously drive interpretability workflows using mlxterp as its toolkit.

---

## Tier 1: Critical for Causal Interpretability

### 1. Activation Patching API

**Why**: The single most-used technique in mechanistic interpretability. Every competing library has this as a first-class API. Currently requires manual two-trace boilerplate with `replace_with()`.

**What researchers expect**:

```python
results = model.activation_patching(
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    layers=range(16),
    component="resid_post",   # or "attn", "mlp", "attn_head"
    metric=logit_diff,        # user-defined metric function
    positions=None,            # None = all positions
)
# Returns: (n_layers, n_positions) matrix of causal effects
```

**Deliverables**:
- [ ] `model.activation_patching()` method on InterpretableModel
- [ ] Support for component types: `resid_post`, `resid_pre`, `attn`, `mlp`, `attn_head`
- [ ] Position-level and layer-level patching
- [ ] Built-in metrics: `logit_diff`, `kl_divergence`, `cross_entropy_diff`
- [ ] Patching result heatmap visualization (layer x position matrix)

**Reference**: TransformerLens `patching` module, pyvene interchange interventions

---

### 2. Multi-Input Trace (Clean/Corrupted Pairs)

**Why**: Most causal experiments need two inputs simultaneously. The current API works but is verbose and error-prone. A unified API makes experiments ergonomic and less bug-prone.

**Current (verbose)**:

```python
with model.trace(clean_input) as clean:
    clean_mlp_5 = model.layers[5].mlp.output.save()

with model.trace(corrupted_input, interventions={
    "layers.5.mlp": replace_with(clean_mlp_5)
}):
    patched_output = model.output.save()
```

**Proposed**:

```python
with model.causal_trace(clean_input, corrupted_input) as ct:
    ct.patch("layers.5.mlp")       # swap clean activation into corrupted run
    result = ct.output.save()
    effect = ct.metric(logit_diff)  # compute metric on patched vs corrupted
```

**Deliverables**:
- [ ] `model.causal_trace(clean, corrupted)` context manager
- [ ] `.patch(component)` method for declarative patching
- [ ] `.metric(fn)` for computing causal effects inline
- [ ] Automatic activation alignment when sequence lengths differ
- [ ] Batch support for patching multiple components in one pass

---

### 3. Text Generation with Interventions

**Why**: Can't study in-context learning, induction heads in practice, or intervention effects on generated text without autoregressive generation. pyvene supports per-token interventions during generation.

**Proposed API**:

```python
# Basic generation
output = model.generate("The capital of France is", max_tokens=20)

# Generation with persistent intervention
output = model.generate(
    "The capital of France is",
    max_tokens=20,
    interventions={"layers.5": add_vector(steering_vec)}
)

# Per-token intervention (advanced)
output = model.generate(
    prompt,
    max_tokens=20,
    interventions={"layers.5": add_vector(steering_vec)},
    intervention_tokens="all"  # or specific token positions
)
```

**Deliverables**:
- [ ] `model.generate()` with basic sampling (greedy, temperature, top-k, top-p)
- [ ] Intervention support during generation
- [ ] KV-cache integration (interventions must work with cached inference)
- [ ] Token-by-token callback for custom stopping / logging

**Reference**: pyvene per-token interventions, mlx-lm generate utilities

---

### 4. Attribution Patching (Gradient-Based)

**Why**: 100x faster than brute-force activation patching. Approximates patching effect using gradients (one forward + one backward pass instead of O(layers x positions) forward passes). Now the standard first-pass technique in circuit analysis. Identified IOI heads in <4.1 seconds vs 8 minutes for ACDC.

**Proposed API**:

```python
attr = model.attribution_patching(
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    metric=logit_diff,
    component="attn_head",  # per-head attribution
)
# Returns: (n_layers, n_heads) or (n_layers, n_positions) attribution scores
```

**Deliverables**:
- [ ] Gradient computation through MLX (mx.grad or vjp)
- [ ] Attribution patching for layers, heads, MLPs, positions
- [ ] Integrated visualization (heatmap of attributions)
- [ ] Validation against brute-force patching results

**Challenge**: MLX's lazy evaluation and functional design may require careful handling for backward passes.

**Reference**: Neel Nanda's attribution patching, Relevance Patching (RelP)

---

### 5. Path Patching (Edge-Level Circuits)

**Why**: Activation patching tells you *which* components matter. Path patching tells you *which connections between components* matter. Essential for circuit discovery (e.g., "head 9.1 reads from head 7.10's output").

**Proposed API**:

```python
edge_effects = model.path_patching(
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    sender="layers.7.attn",       # source component
    receiver="layers.9.attn",     # destination component
    metric=logit_diff,
)
```

**Deliverables**:
- [ ] Sender/receiver specification for path patching
- [ ] Freeze-all-except patching (isolate specific paths)
- [ ] Edge-level causal effect computation
- [ ] Circuit graph construction from path patching results

**Reference**: TransformerLens path patching, Wang et al. "Interpretability in the Wild" (2022)

---

## Tier 2: Important for Competitive Feature Set

### 6. Direct Logit Attribution (DLA)

**Why**: Decompose the final logit into per-head and per-MLP contributions. Starting point for most circuit analyses. Basic in TransformerLens.

**Deliverables**:
- [ ] Residual stream decomposition per component
- [ ] Per-head logit contribution computation
- [ ] Per-MLP logit contribution computation
- [ ] DLA visualization (bar chart of component contributions)

---

### 7. Residual Stream Access

**Why**: The residual stream is the central data structure in transformer interpretability. Explicit access to pre/post residual stream at each layer underpins DLA, path patching, and circuit analysis.

**Deliverables**:
- [ ] Named hook points: `resid_pre`, `resid_post`, `resid_mid` (between attn and MLP)
- [ ] `model.layers[i].resid_pre.save()` / `model.layers[i].resid_post.save()` proxy access
- [ ] Residual stream difference computation (contribution of each layer)

---

### 8. SAE Feature Circuits / Indirect Effects

**Why**: SAE training without causal analysis of features is incomplete. The BauLab "sparse feature circuits" method and Anthropic's attribution graphs represent the frontier of interpretable circuit analysis.

**Deliverables**:
- [ ] Compute indirect effect of each SAE feature on model output
- [ ] Prune feature graph by effect threshold
- [ ] Feature-level causal graph construction
- [ ] Feature activation patching (patch individual SAE features)
- [ ] Integration with existing SAE training pipeline

**Reference**: BauLab sparse feature circuits, Anthropic attribution graphs (May 2025)

---

### 9. Feature Dashboards

**Why**: Max-activating examples and feature statistics are essential for understanding what SAE features represent. SAELens + SAE-Vis set the standard.

**Deliverables**:
- [ ] Max-activating dataset examples per feature (with context highlighting)
- [ ] Feature activation distribution histograms
- [ ] Logit weight analysis (which output tokens does a feature promote/suppress)
- [ ] Feature co-occurrence analysis
- [ ] HTML dashboard generation (standalone, shareable)

**Reference**: SAE-Vis, Neuronpedia

---

### 10. Patching Result Visualization

**Why**: Standard mechinterp output is a (layer x position) heatmap of causal effects. Currently no way to visualize patching results.

**Deliverables**:
- [ ] Layer x position heatmap for activation patching results
- [ ] Layer x head heatmap for head-level patching
- [ ] Component contribution bar charts for DLA
- [ ] Interactive Plotly versions for exploration
- [ ] Matplotlib versions for publication

---

## Tier 3: Differentiators / Frontier

### 11. ACDC (Automated Circuit Discovery)

Automated pruning of computational graphs to find minimal sufficient circuits. Iteratively removes edges that don't affect the metric.

- [ ] ACDC algorithm implementation
- [ ] Circuit visualization (DAG of components)
- [ ] Comparison with attribution patching results

**Reference**: Conmy et al. "Towards Automated Circuit Discovery" (2023)

---

### 12. Trainable Interventions (DAS / Boundless DAS)

pyvene's unique capability: learn *which subspace* of an activation is causally relevant via gradient descent. Powerful for causal abstraction testing.

- [ ] Distributed Alignment Search (DAS) implementation
- [ ] Boundless DAS with learnable boundaries
- [ ] Low-rank rotated subspace interventions
- [ ] Integration with MLX's optimization utilities

**Reference**: pyvene, Geiger et al. "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations" (2023)

---

### 13. Cross-Layer Transcoders (CLTs)

Anthropic's latest approach: replace MLPs with interpretable feature dictionaries that model cross-layer information flow. Open-sourced for Gemma-2-2b and Llama-3.2-1b.

- [ ] CLT architecture implementation
- [ ] Attribution graph construction from CLT features
- [ ] Integration with existing SAE infrastructure
- [ ] Pre-trained CLT loading (Anthropic open-source weights)

**Reference**: Anthropic "Circuit Tracing" (May 2025)

---

### 14. Automated Interpretability

LLM-generated feature descriptions and sensitivity testing.

- [ ] Auto-labeling SAE features via LLM prompting
- [ ] Sensitivity testing (does toggling the feature change behavior as expected?)
- [ ] Feature description confidence scoring
- [ ] Batch processing for large feature dictionaries

**Reference**: EleutherAI auto-interp, Anthropic automated interpretability

---

## Tier 4: Conversation-Level Interpretability

Most interpretability research analyzes single prompts. But LLMs are used in multi-turn conversations where meaning builds across turns, context accumulates in KV-cache, and interventions on early turns cascade through later ones. mlxterp should support analysis at the conversation level, not just the token level.

### 15. Conversation Trace (Multi-Turn Analysis)

**Why**: A chat model's response to turn 5 depends on everything said in turns 1-4. Researchers need to trace how information flows across conversation turns — which earlier turns does the model "remember"? How does KV-cache carry information? Where does context get lost?

#### How Turn Identification Works

Chat models use **chat templates** that insert special tokens between turns. These tokens are the natural turn boundaries:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

My name is Alice.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Nice to meet you!<|eot_id|><|start_header_id|>user<|end_header_id|>

What's my name?<|eot_id|>
```

**Implementation approach**:

1. **Tokenize with chat template** — use the tokenizer's `apply_chat_template()` to get the full token sequence with all special tokens inserted
2. **Scan for boundary tokens** — detect template-specific delimiters:
   - Llama: `<|start_header_id|>` ... `<|end_header_id|>` ... `<|eot_id|>`
   - ChatML (Qwen, etc.): `<|im_start|>` ... `<|im_end|>`
   - Gemma: `<start_of_turn>` ... `<end_of_turn>`
   - Fallback: regex on known patterns, or user-provided boundary token IDs
3. **Build a turn index** — map each token position to its turn, role, and whether it's content or template markup
4. **Expose via Turn objects** — each turn knows its position range, role, and can slice activations

**Internal data structure**:

```python
@dataclass
class Turn:
    index: int              # turn number (0-indexed)
    role: str               # "user", "assistant", "system"
    full_start: int         # first token position (including template tokens)
    full_end: int           # last token position (including template tokens)
    content_start: int      # first content token (excluding role markers)
    content_end: int        # last content token (excluding eot markers)
    
    @property
    def content_positions(self) -> slice:
        return slice(self.content_start, self.content_end)
    
    @property
    def full_positions(self) -> slice:
        return slice(self.full_start, self.full_end)
```

**Proposed API**:

```python
conversation = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"},
]

# Trace the full conversation, get per-turn activations
with model.conversation_trace(conversation) as ct:
    # Access activations at specific turns
    turn1_residual = ct.turns[0].layers[5].output.save()
    turn3_residual = ct.turns[2].layers[5].output.save()
    final_output = ct.output.save()

# Turn metadata
print(ct.turns)
# [Turn(0, role="user",      content=5:12,  full=1:13),
#  Turn(1, role="assistant",  content=16:22, full=13:23),
#  Turn(2, role="user",      content=26:31, full=23:32)]

# Flexible turn selection
ct.turns[0]                          # all tokens in turn 0 (including template)
ct.turns[0].content                  # only content tokens (skip role/eot markers)
ct.turns[0:2]                        # turns 0 and 1
ct.turns.by_role("user")             # all user turns
ct.turns.by_role("assistant")        # all assistant turns
ct.turns[2].layers[5].output.save()  # activations sliced to turn 2's positions only

# Cross-turn attention
ct.cross_turn_attention(layer=5, head=0)
# Returns: (n_turns, n_turns) matrix of aggregated attention between turns
```

**Deliverables**:
- [ ] `model.conversation_trace(messages)` context manager
- [ ] `Turn` dataclass with content vs full position tracking
- [ ] `TurnList` with indexing, slicing, role filtering
- [ ] Chat template detection (auto-detect from tokenizer, support Llama/ChatML/Gemma/custom)
- [ ] Per-turn activation slicing via `ct.turns[i].layers[j].output.save()`
- [ ] Cross-turn attention aggregation
- [ ] KV-cache aware tracing (understand cached vs freshly computed activations)
- [ ] Visualization: token-level heatmap with turn boundaries annotated

---

### 16. Turn-Level Causal Patching

**Why**: "Does the model's answer to question 3 actually depend on information from turn 1, or is it relying on turn 2?" This requires patching activations at the turn level — replace all activations from turn 1 with a counterfactual and measure how it affects turn 3's output.

**Proposed API**:

```python
clean_conv = [
    {"role": "user", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What's the capital of France?"},
]

corrupted_conv = [
    {"role": "user", "content": "The capital of France is Berlin."},  # corrupted
    {"role": "user", "content": "What's the capital of France?"},
]

# Patch at the turn level
results = model.conversation_patching(
    clean=clean_conv,
    corrupted=corrupted_conv,
    patch_turns=[0],          # patch turn 0's activations
    measure_turn=1,           # measure effect on turn 1's output
    layers=range(16),
    metric=logit_diff,
)

# Or at specific positions within a turn
results = model.conversation_patching(
    clean=clean_conv,
    corrupted=corrupted_conv,
    patch_turns=[0],
    patch_positions="content_only",  # skip special/template tokens
    measure_turn=1,
    metric=logit_diff,
)
```

**Deliverables**:
- [ ] `model.conversation_patching()` with turn-level granularity
- [ ] Position filtering: `"all"`, `"content_only"` (skip template tokens), specific ranges
- [ ] Cross-turn causal effect matrices (which turns influence which)
- [ ] Integration with activation patching (#1) for position-level detail within turns
- [ ] Visualization: turn-level causal flow diagrams

---

### 17. Conversation Attention Analysis

**Why**: In multi-turn chat, attention patterns across turn boundaries reveal how the model routes information between conversation turns. "Does the model look back at the user's name from turn 1 when generating the greeting in turn 3?"

**Deliverables**:
- [ ] Cross-turn attention heatmaps (x-axis = all tokens, y-axis = response tokens, colored by turn)
- [ ] Turn-aggregated attention (collapse token-level to turn-level: "turn 3 attends X% to turn 1")
- [ ] Induction head analysis across turn boundaries
- [ ] "Information retrieval" head detection (heads that attend to specific earlier turns)
- [ ] Interactive visualization with turn-level controls

---

### 18. Token-Level vs Turn-Level Granularity Control

**Why**: Researchers need to zoom between levels of analysis. Sometimes you want "which specific token in turn 1 matters" (token-level), sometimes "which turn matters" (turn-level), sometimes "which sentence within a turn" (span-level).

**Proposed API**:

```python
# Token-level (existing, enhanced)
with model.trace("The capital of France is") as trace:
    model.layers[5].attn.output.save()

# Turn-level (new)
with model.conversation_trace(messages) as ct:
    ct.turns[0].layers[5].output.save()

# Span-level (new — arbitrary token ranges)
with model.trace("The capital of France is Paris and it is beautiful") as trace:
    # Analyze specific spans
    span1 = trace.span(0, 6)   # "The capital of France is Paris"
    span2 = trace.span(6, 10)  # "and it is beautiful"
    span1_activation = span1.layers[5].output.save()  # activations for just that span
```

**Deliverables**:
- [ ] `trace.span(start, end)` for arbitrary token range analysis
- [ ] `ct.turns[i]` for turn-level access within conversations
- [ ] Aggregation functions: mean/max/last-token over spans
- [ ] Span-level patching (patch a specific sentence, not whole turn)
- [ ] Consistent API across all three granularity levels

---

## Tier 5: Agentic Interpretability — mlxterp as a Research Platform

The key insight: mlxterp shouldn't just be a library that humans call — it should be a **toolkit that frontier models can operate autonomously**. A Claude Code agent (or any LLM agent) should be able to pick up mlxterp and run a full interpretability investigation: form hypotheses, design experiments, run analyses, interpret results, and iterate.

This is directly inspired by **Karpathy's AutoResearch** (March 2026, 72k+ GitHub stars) — a pattern where an LLM coding agent autonomously runs ML experiments in a ratchet loop: propose hypothesis → edit code → run experiment → keep if improved → repeat. AutoResearch achieved ~100 experiments/night and discovered novel architecture optimizations. We adapt this pattern from "optimize training loss" to "discover and validate circuits."

**The AutoResearch Three-File Contract, adapted for interpretability:**

| AutoResearch (Karpathy) | AutoInterp (mlxterp) |
|---|---|
| `prepare.py` — data prep (immutable) | `setup.py` — load model + SAE + dataset (immutable) |
| `train.py` — agent edits this | `experiment.py` — agent writes experiments here |
| `program.md` — human instructions | `program.md` — research question + constraints |
| `val_bpb` — single scalar metric | flexible: logit_diff, KL, patching effect, circuit completeness |
| `results.tsv` — experiment log | `results.jsonl` — structured experiment log with metadata |
| git ratchet (keep if val_bpb improves) | git ratchet (keep if finding is informative) |

**Key difference**: AutoResearch optimizes a single scalar. Interpretability research is *exploratory* — the agent must decide what's "informative" (a null result that rules out a hypothesis is valuable). The ratchet keeps findings, not just improvements.

This makes mlxterp unique — no other interpretability library is designed for agent-driven research.

### 19. MCP Server — mlxterp as Claude Code Tools

**Why**: Expose mlxterp's capabilities as MCP (Model Context Protocol) tools so Claude Code can invoke them directly during conversation. The user says "investigate how this model handles negation" and Claude Code autonomously runs traces, patches, visualizes, and reports findings.

**Architecture**:

```
┌─────────────┐     MCP (JSON-RPC)     ┌──────────────────┐
│ Claude Code  │◄──────────────────────►│ mlxterp MCP      │
│ (Agent)      │    tool calls/results  │ Server           │
│              │                        │                  │
│ Plans next   │                        │ ┌──────────────┐ │
│ experiment   │                        │ │InterpretModel│ │
│ based on     │                        │ │(loaded once)  │ │
│ results      │                        │ └──────────────┘ │
└─────────────┘                        │ Runs on Apple    │
                                       │ Silicon / MLX    │
                                       └──────────────────┘
```

**Proposed MCP Tools**:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mlxterp")

@mcp.tool()
def load_model(model_name: str) -> str:
    """Load an MLX model for interpretability analysis."""

@mcp.tool()
def trace_and_get_activations(
    text: str, layers: list[int], positions: list[int]
) -> str:
    """Run a forward pass and return activations at specified layers/positions."""

@mcp.tool()
def activation_patching(
    clean: str, corrupted: str, component: str, metric: str
) -> str:
    """Run activation patching and return causal effect matrix as JSON."""

@mcp.tool()
def detect_attention_patterns(text: str, pattern_type: str) -> str:
    """Detect attention head types (induction, previous_token, etc.)."""

@mcp.tool()
def logit_lens(text: str, layers: list[int], position: int) -> str:
    """Run logit lens and return per-layer predictions."""

@mcp.tool()
def ablate_component(
    text: str, component: str, metric: str
) -> str:
    """Zero out a component and measure effect on output."""

@mcp.tool()
def get_top_sae_features(text: str, layer: int, top_k: int) -> str:
    """Get top-k active SAE features for a text at a layer."""

@mcp.tool()
def save_visualization(viz_type: str, data: str, path: str) -> str:
    """Generate and save a visualization (heatmap, attention, etc.)."""

@mcp.tool()
def conversation_trace(messages: list[dict], layers: list[int]) -> str:
    """Trace a multi-turn conversation and return per-turn activation summary."""
```

**Deliverables**:
- [ ] `mlxterp-server` CLI command to start MCP server
- [ ] MCP tools for all core operations (trace, patch, ablate, visualize, SAE)
- [ ] Structured JSON output from all tools (not raw tensors)
- [ ] Model persistence across tool calls (load once, analyze many times)
- [ ] `.mcp.json` template for easy Claude Code integration
- [ ] Streaming progress for long-running operations (patching sweeps)

**Reference**: MCP Python SDK (`mcp` package), FastMCP pattern

---

### 20. Structured Analysis Output

**Why**: For an agent to reason about results, outputs must be machine-readable, not just human-readable plots. Every analysis function should return structured data (JSON-serializable dicts) alongside optional visualizations.

**Proposed pattern**:

```python
# Current: returns a matplotlib figure (useless to an agent)
fig = model.logit_lens("The capital of France is", plot=True)

# Enhanced: returns structured data + optional plot
result = model.logit_lens("The capital of France is")
result.data        # Dict: {layer_idx: [{token: "Paris", score: 0.82}, ...]}
result.summary     # "Layer 15 predicts 'Paris' (p=0.82), layer 10 predicts 'France' (p=0.34)"
result.plot()      # Optional: generate visualization
result.to_json()   # Serialize for agent consumption
result.to_markdown()  # Human-readable report
```

**Deliverables**:
- [ ] `AnalysisResult` base class with `.data`, `.summary`, `.to_json()`, `.to_markdown()`, `.plot()`
- [ ] Refactor all analysis methods to return `AnalysisResult` objects
- [ ] Auto-generated text summaries for each analysis type
- [ ] JSON schema documentation for each result type (so agents know what to expect)

---

### 21. Research Workflow Primitives

**Why**: An agent needs higher-level primitives beyond individual tool calls. "Investigate which components handle factual recall" is a multi-step workflow: run logit lens, identify candidate layers, run activation patching, narrow to heads, run path patching, report circuit.

**Proposed API**:

```python
from mlxterp.workflows import (
    circuit_discovery,
    feature_investigation,
    behavior_localization,
)

# Pre-built workflow: localize a behavior
report = behavior_localization(
    model=model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    target_token="Paris",
    steps=["logit_lens", "activation_patching", "head_attribution"],
)
# Returns: structured report with identified components, visualizations, and narrative

# Pre-built workflow: investigate an SAE feature
report = feature_investigation(
    model=model,
    sae=sae,
    feature_id=1234,
    dataset=texts,
    steps=["max_activating", "ablation", "logit_effect", "description"],
)
```

**Deliverables**:
- [ ] `behavior_localization` workflow (logit lens → patching → head attribution)
- [ ] `circuit_discovery` workflow (attribution patching → path patching → circuit graph)
- [ ] `feature_investigation` workflow (max examples → ablation → logit effect → auto-description)
- [ ] Workflow step results are individually accessible (for agent inspection mid-workflow)
- [ ] Interruptible workflows (agent can stop, inspect, adjust, continue)
- [ ] Workflow results as `AnalysisResult` objects

---

### 22. AutoInterp — Karpathy-Style Ratchet Loop for Interpretability

**Why**: Karpathy's AutoResearch showed that an LLM agent can run ~100 ML experiments overnight on a single GPU with no human in the loop. The same pattern applies to interpretability: an agent can systematically explore a model's internals, running patching/ablation/feature experiments in a continuous loop, building up a picture of the model's circuits.

The key adaptation: instead of optimizing `val_bpb`, the agent explores and documents circuits. Instead of "keep if loss improves," it's "keep if the finding is informative" (including null results that rule out hypotheses).

**The Three-File Contract for AutoInterp**:

```
autointerpret/
├── setup.py        # IMMUTABLE — loads model, SAE, dataset, defines metrics
├── experiment.py   # AGENT-OWNED — agent writes experiments here
├── program.md      # HUMAN-OWNED — research question + constraints
├── results.jsonl   # Append-only experiment log
└── findings/       # Accumulated findings (kept commits)
    ├── 001_logit_lens_overview.json
    ├── 002_patching_layer5_important.json
    └── ...
```

**`setup.py`** (immutable, defines the research environment):

```python
from mlxterp import InterpretableModel
from mlxterp.autointerpret import MetricRegistry

# Load model (human configures once)
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Define metrics the agent can use
metrics = MetricRegistry()
metrics.register("logit_diff", lambda clean, patched, target, foil:
    patched[0, -1, target] - patched[0, -1, foil])
metrics.register("kl_div", ...)
metrics.register("prob_change", ...)

# Dataset for feature analysis
dataset = [...]  # loaded once
```

**`program.md`** (human writes, agent reads):

```markdown
# Research Program

## Question
How does Llama-3.2-1B recall factual associations (e.g., "The Eiffel Tower is in" → "Paris")?

## Constraints
- Focus on layers 0-15, all attention heads and MLPs
- Use activation patching first, then narrow with attribution patching
- Each experiment should complete in under 2 minutes
- Keep experiments that identify important components OR rule them out
- Do NOT modify setup.py

## Current Priorities
1. Run logit lens to identify where "Paris" first appears
2. Run activation patching across all layers to find critical ones
3. For critical layers, identify specific heads via head-level patching
4. Test causal path between identified heads

## Stop When
- You have identified a circuit (set of components) sufficient for the task
- OR you have run 100 experiments without convergence
```

**The Ratchet Loop** (runs autonomously):

```
┌─────────────────────────────────────────────────────────┐
│  1. Read program.md for research priorities              │
│  2. Read results.jsonl for past findings                 │
│  3. Propose hypothesis based on what's known/unknown     │
│  4. Write experiment in experiment.py                    │
│  5. Run experiment (time-boxed, e.g., 2 min max)        │
│  6. Parse results → structured JSON                      │
│  7. Decide: informative? → append to results.jsonl       │
│             + git commit to findings/                     │
│             uninformative? → git revert, log, move on    │
│  8. Update mental model of the circuit                   │
│  9. Loop to step 1                                       │
└─────────────────────────────────────────────────────────┘
```

**Proposed API — two modes of operation**:

**Mode 1: Claude Code native (zero orchestration)**
Just open Claude Code in the `autointerpret/` directory and say "read program.md and start." The agent uses mlxterp directly, same as Karpathy's approach. No custom orchestration code needed.

**Mode 2: Programmatic agent (for automation/scaling)**

```python
from mlxterp.autointerpret import AutoInterpret

runner = AutoInterpret(
    model=model,
    program="program.md",          # research question
    llm="claude-sonnet-4-6",      # reasoning model
    max_experiments=100,
    time_per_experiment=120,        # seconds
    output_dir="findings/",
)

# Run the loop (can run overnight)
report = runner.run()

# Or run a single iteration
result = runner.step()
```

**Deliverables**:
- [ ] `autointerpret/` scaffold generator (`mlxterp init-autointerpret`)
- [ ] `setup.py` template with model loading and metric registry
- [ ] `program.md` template with example research questions
- [ ] `AutoInterpret` runner class with the ratchet loop
- [ ] Experiment time-boxing (kill if exceeds budget)
- [ ] `results.jsonl` structured logging with schema
- [ ] Git integration (commit findings, revert failures)
- [ ] Summary generation after N experiments
- [ ] "Zero orchestration" mode (works with any LLM coding agent reading program.md)
- [ ] Claude Code integration guide

**Reference**: [Karpathy AutoResearch](https://github.com/karpathy/autoresearch) (March 2026), Anthropic automated interpretability

---

### 23. Agent-Driven Auto-Interpretability (Interactive Mode)

**Why**: AutoInterp (#22) is the overnight batch mode. This is the interactive mode — the user has a conversation with Claude Code, and Claude uses mlxterp MCP tools to investigate on-the-fly. "What's going on in layer 5?" → Claude runs the analysis and explains.

**Integration with Claude Code**:

```
User: "Investigate how Llama-3.2-1B handles subject-verb agreement"

Claude Code (via MCP):
  1. Designs test prompts with agreement/disagreement contrasts
  2. Runs logit_lens to identify where agreement info appears
  3. Runs activation_patching across layers
  4. Identifies candidate attention heads
  5. Runs path_patching to map the circuit
  6. Tests with ablations to confirm
  7. Generates a report with visualizations
  8. Proposes follow-up experiments
```

**Integration with Claude API (programmatic)**:

```python
from mlxterp.agents import InterpretabilityAgent

agent = InterpretabilityAgent(
    model=model,                              # local MLX model to analyze
    llm="claude-sonnet-4-6",                # frontier model for reasoning
    sae=sae,                                  # optional: loaded SAE
)

# Autonomous investigation
report = agent.investigate(
    question="How does this model handle negation?",
    budget=50,            # max tool calls
    verbose=True,         # print progress
)

report.findings          # structured findings
report.circuit           # discovered circuit (if any)
report.visualizations    # generated figures
report.narrative         # natural language writeup
report.next_steps        # suggested follow-up experiments
```

**Deliverables**:
- [ ] `InterpretabilityAgent` class wrapping Claude API + mlxterp tools
- [ ] Hypothesis generation from initial observations
- [ ] Experiment design (which patching/ablation to run)
- [ ] Result interpretation (what do the numbers mean)
- [ ] Iterative refinement (narrow down based on results)
- [ ] Report generation with embedded visualizations
- [ ] Cost/budget tracking (API calls + compute time)
- [ ] Experiment logging (full trace of what the agent did and why)

**Reference**: Anthropic automated interpretability, EleutherAI auto-interp

---

### 23. Interpretability Report Generation

**Why**: The final output of an interpretability investigation should be a shareable, readable report — not a pile of JSON. Whether generated by a human or an agent, mlxterp should produce publication-quality reports.

**Deliverables**:
- [ ] Markdown report generation from workflow/agent results
- [ ] Embedded visualizations (inline base64 images or HTML)
- [ ] LaTeX export for papers
- [ ] Jupyter notebook generation (executable report)
- [ ] Obsidian-compatible notes (for integration with research workflows)

---

## Suggested Implementation Order

| Phase | Items | Rationale |
|-------|-------|-----------|
| **Phase 3a** | Activation Patching API (#1) + Patching Visualization (#10) | Unblocks 80% of causal interp workflows |
| **Phase 3b** | Multi-Input Trace (#2) + Text Generation (#3) | Makes experiments ergonomic, enables ICL study |
| **Phase 3c** | DLA (#6) + Residual Stream (#7) | Foundation for circuit analysis |
| **Phase 4a** | Attribution Patching (#4) + Path Patching (#5) | Fast circuit discovery |
| **Phase 4b** | SAE Feature Circuits (#8) + Feature Dashboards (#9) | Feature-level interpretability |
| **Phase 5a** | Conversation Trace (#15) + Turn-Level Patching (#16) | Multi-turn analysis |
| **Phase 5b** | Conversation Attention (#17) + Granularity Control (#18) | Complete conversation interp |
| **Phase 6a** | Structured Output (#20) + MCP Server (#19) | Agent-ready infrastructure |
| **Phase 6b** | Research Workflows (#21) + AutoInterp Ratchet (#22) | Karpathy-style overnight interpretability |
| **Phase 6c** | Interactive Agent (#23) + Report Generation (#24) | On-demand + shareable research outputs |
| **Phase 7** | ACDC (#11) + Trainable Interventions (#12) | Automated circuit discovery |
| **Frontier** | CLTs (#13) + Auto-Interp (#14) | Research frontier |

| Phase | Items | Rationale |
|-------|-------|-----------|
| **Phase 3a** | Activation Patching API (#1) + Patching Visualization (#10) | Unblocks 80% of causal interp workflows |
| **Phase 3b** | Multi-Input Trace (#2) + Text Generation (#3) | Makes experiments ergonomic, enables ICL study |
| **Phase 3c** | DLA (#6) + Residual Stream (#7) | Foundation for circuit analysis |
| **Phase 4a** | Attribution Patching (#4) + Path Patching (#5) | Fast circuit discovery |
| **Phase 4b** | SAE Feature Circuits (#8) + Feature Dashboards (#9) | Feature-level interpretability |
| **Phase 5** | ACDC (#11) + Trainable Interventions (#12) | Automated circuit discovery |
| **Frontier** | CLTs (#13) + Auto-Interp (#14) | Research frontier |

---

## Competitive Position

| Feature | TransformerLens | nnsight | pyvene | mlxterp (current) | mlxterp (after plan) |
|---|---|---|---|---|---|
| Tracing/hooking | Hook points | Proxy graph | Config-driven | Proxy + context mgr | Same |
| Activation patching | Built-in | Manual | Built-in | Manual (verbose) | Built-in |
| Attribution patching | Community | No | No | No | Built-in |
| Path patching | Built-in | Manual | No | No | Built-in |
| DLA | Built-in | No | No | No | Built-in |
| Trainable interventions | No | No | Built-in | No | Planned |
| SAE training | Via SAELens | No | No | Built-in | Enhanced |
| Feature circuits | Via BauLab | No | No | No | Built-in |
| Text generation + interv. | No | No | Built-in | No | Built-in |
| **Conversation-level analysis** | No | No | No | No | **Built-in** |
| **Turn-level patching** | No | No | No | No | **Built-in** |
| **MCP server / agent tools** | No | No | No | No | **Built-in** |
| **Autonomous interp agent** | No | No | No | No | **Built-in** |
| **AutoResearch-style ratchet** | No | No | No | No | **Built-in** |
| **Structured output for agents** | No | No | No | No | **Built-in** |
| Apple Silicon native | No | No | No | Yes | Yes |
| Model-agnostic | No (specific models) | Yes | Yes (HF) | Yes (any MLX) | Yes |

### mlxterp's unique advantages after this roadmap

1. **Apple Silicon native**: Only full interpretability stack running natively on MLX.
2. **Conversation-level interpretability**: No other library supports multi-turn analysis, turn-level patching, or cross-turn attention visualization. This is an unoccupied niche — all existing libraries analyze single prompts.
3. **Agent-driven interpretability**: No other library is designed to be operated by frontier models. The MCP server + structured output + research workflows combination means Claude Code (or any LLM agent) can autonomously run interpretability investigations on local models. This transforms mlxterp from a tool into a **research platform**.
4. **Full vertical integration**: From raw tracing to autonomous reports — load model, trace, patch, discover circuits, train SAEs, analyze features, generate reports — all in one library, all on one machine.
5. **AutoResearch for interpretability**: Karpathy's ratchet loop adapted for circuit discovery — run 100 interpretability experiments overnight, accumulate findings in git, produce a circuit report by morning. No other library offers this.
