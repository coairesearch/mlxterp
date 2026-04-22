# Agentic Interpretability Guide

## Overview

mlxterp is designed not just as a library for humans — it's a **toolkit that LLM agents can operate autonomously**. Claude Code (or any LLM coding agent) can pick up mlxterp and run a full interpretability investigation: form hypotheses, run experiments, interpret results, and iterate.

This guide covers:
1. **Research Workflows** — pre-built multi-step pipelines
2. **AutoInterp Ratchet** — Karpathy-style overnight experiment loops
3. **Automated Interpretability** — LLM-generated SAE feature labels
4. **Report Generation** — shareable outputs from any analysis

## 1. Research Workflows

Pre-built pipelines that chain together analysis tools and return comprehensive results.

### Behavior Localization

Identifies which model components cause a specific behavior:

```python
from mlxterp.workflows import behavior_localization

result = behavior_localization(
    model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    metric="l2",
    verbose=True,
)

# Multi-step pipeline: DLA → MLP patching → attention patching → head-level
print(result.narrative)
print(result.summary())

# Access individual steps
dla = result.get_step("dla")
mlp_patch = result.get_step("patch_mlp")
attn_patch = result.get_step("patch_attn")

# Export as markdown report
print(result.to_markdown())
```

### Circuit Discovery

Discovers the minimal circuit for a behavior:

```python
from mlxterp.workflows import circuit_discovery

result = circuit_discovery(
    model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    threshold=0.01,
    verbose=True,
)

# Pipeline: attribution patching → activation patching → ACDC
circuit = result.get_step("acdc")
print(f"Circuit: {circuit.nodes}")
```

### Feature Investigation

Analyzes SAE features: finds active features, ablates them, finds max-activating examples:

```python
from mlxterp.workflows import feature_investigation

result = feature_investigation(
    model, sae,
    text="The capital of France is",
    layer=10,
    dataset=dataset_texts,  # Optional: for max-activating examples
    verbose=True,
)

active = result.get_step("active_features")
ablation = result.get_step("ablation")
```

### Running Specific Steps

All workflows accept a `steps` parameter to run only specific parts:

```python
# Only run DLA and MLP patching
result = behavior_localization(
    model, clean, corrupted,
    steps=["dla", "patch_mlp"],
)
```

## 2. AutoInterp: Overnight Experiment Loops

Adapted from Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch) pattern. An LLM agent runs interpretability experiments in a ratchet loop, accumulating findings.

### The Three-File Contract

| File | Owner | Purpose |
|------|-------|---------|
| `setup.py` | Human (immutable) | Loads model, defines metrics, dataset |
| `experiment.py` | Agent (editable) | Agent writes experiments here |
| `program.md` | Human (read by agent) | Research question and constraints |
| `results.jsonl` | Append-only | Structured experiment log |
| `CLAUDE.md` | Auto-generated | Instructions for the agent |

### Scaffold Generation

```python
from mlxterp.autointerpret import init_autointerpret

# Generate the project structure
path = init_autointerpret(
    output_dir="my_investigation",
    model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
    research_question="How does this model recall factual associations?",
    max_experiments=100,
)

print(f"Project created at: {path}")
# my_investigation/
# ├── setup.py          # Model + metrics (don't modify)
# ├── experiment.py     # Agent writes here
# ├── program.md        # Research question
# ├── CLAUDE.md         # Agent instructions
# ├── results.jsonl     # Experiment log
# └── findings/         # Kept findings
```

### Zero-Orchestration Mode (Recommended)

1. Run `init_autointerpret()`
2. Open the directory in Claude Code
3. Say: **"Read program.md and start investigating."**

Claude Code will:
- Read the research question
- Import mlxterp from setup.py
- Design and run experiments
- Log results to results.jsonl
- Commit informative findings
- Iterate until the circuit is found or max experiments reached

### Programmatic Mode

```python
from mlxterp.autointerpret import AutoInterpret
from mlxterp.causal import activation_patching

runner = AutoInterpret(
    model=model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    output_dir="my_experiment",
    max_experiments=50,
)

# Run experiments
entry = runner.run_experiment(
    name="mlp_patching",
    fn=lambda: activation_patching(model, clean, corrupted, component="mlp"),
    hypothesis="MLPs at mid-layers are important",
)

print(f"Result: {entry.conclusion}")
print(f"Informative: {entry.informative}")

# Check progress
print(runner.summary())
print(f"Experiments: {runner.n_experiments}/{runner.max_experiments}")
print(f"Done: {runner.is_done}")
```

### Experiment Logging

```python
from mlxterp.autointerpret import ExperimentLog, ExperimentEntry

log = ExperimentLog("results.jsonl")

# Log an experiment
log.append(ExperimentEntry(
    hypothesis="Layer 5 MLP handles factual recall",
    method="activation_patching",
    result={"layer_5_effect": 0.82, "layer_10_effect": 0.15},
    informative=True,
    conclusion="Confirmed: layer 5 MLP has 82% recovery effect",
    duration_seconds=12.5,
))

# Read back
print(log.summary())
for entry in log.informative_entries():
    print(f"  [{entry.experiment_id}] {entry.conclusion}")
```

### MetricRegistry

```python
from mlxterp.autointerpret import MetricRegistry

metrics = MetricRegistry()
metrics.register("logit_diff", my_logit_diff_fn, description="Logit difference recovery")
metrics.register("prob_correct", my_prob_fn, description="Probability of correct token")

# Agent can discover available metrics
for m in metrics.list():
    print(f"  {m['name']}: {m['description']}")

# Use a metric
fn = metrics.get("logit_diff")
```

## 3. Automated Interpretability

Use Claude (or any LLM) to automatically label SAE features based on their max-activating examples.

### Single Feature Labeling

```python
from mlxterp.auto_interp import auto_label_feature

label = auto_label_feature(
    model, sae,
    feature_id=42,
    texts=dataset_texts,
    layer=10,
    llm_model="claude-sonnet-4-20250514",
)

print(f"Feature {label.feature_id}: {label.label}")
print(f"Description: {label.description}")
print(f"Confidence: {label.confidence:.0%}")
print(f"Evidence: {len(label.evidence)} examples")
```

### Batch Labeling

```python
from mlxterp.auto_interp import auto_label_features

labels = auto_label_features(
    model, sae,
    texts=dataset_texts,
    layer=10,
    top_k_features=20,   # Auto-detect top 20 most active features
    verbose=True,
)

for label in labels:
    print(f"  f{label.feature_id}: {label.label} ({label.confidence:.0%})")
```

### Sensitivity Testing

Validate labels by checking if the feature activates consistently on related inputs:

```python
from mlxterp.auto_interp import sensitivity_test

label = sensitivity_test(
    model, sae, label,
    test_texts=validation_texts,
    layer=10,
)

print(f"Sensitivity test: {'PASS' if label.sensitivity_passed else 'FAIL'}")
print(f"Details: {label.sensitivity_details}")
```

### Requirements

Auto-labeling requires the Anthropic SDK:
```bash
pip install anthropic
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 4. Report Generation

Generate shareable reports from any analysis result.

### Markdown Reports

```python
from mlxterp.reports import generate_report, save_report

# From a single result
report = generate_report(
    patching_result,
    title="Factual Recall Circuit Analysis",
    description="Investigating how Llama-3.2-1B recalls the capital of France.",
)
print(report)

# From multiple results
report = generate_report(
    [patching_result, dla_result, circuit_result],
    title="Complete Investigation",
)

# Save to file
save_report(results, "report.md", title="My Analysis")
```

### HTML Reports

```python
# HTML with embedded plots
save_report(
    results,
    "report.html",
    title="Investigation Report",
    include_plots=True,
)
```

### Workflow Reports

Workflows generate reports automatically:

```python
result = behavior_localization(model, clean, corrupted)

# Markdown report with all steps
print(result.to_markdown())

# JSON for programmatic consumption
print(result.to_json())
```

## Agent-Friendly Design

All mlxterp outputs are designed for agent consumption:

| Feature | Human Use | Agent Use |
|---------|-----------|-----------|
| `result.summary()` | Quick overview | Decision input |
| `result.to_json()` | Data export | Structured parsing |
| `result.to_markdown()` | Reading | Report generation |
| `result.plot()` | Visual inspection | Embed in reports |
| `result.top_components(k)` | Find important parts | Prioritize next experiment |

Claude Code can use all of these directly — no MCP server needed. Just `import mlxterp` and go.
