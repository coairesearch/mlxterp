"""
Scaffold generator for AutoInterp projects.

Creates the three-file contract directory structure that any LLM coding
agent can pick up and run.
"""

import os
from pathlib import Path
from typing import Optional


SETUP_TEMPLATE = '''"""
AutoInterp Setup — IMMUTABLE (do not modify during experiments)

This file defines the research environment: model, metrics, and dataset.
The agent reads this to understand what tools are available.
"""

from mlxterp import InterpretableModel
from mlxterp.autointerpret import MetricRegistry
import mlx.core as mx

# ============================================================
# MODEL — Configure your model here
# ============================================================
MODEL_NAME = "{model_name}"
model = InterpretableModel(MODEL_NAME)

# ============================================================
# METRICS — Define metrics the agent can use
# ============================================================
metrics = MetricRegistry()

# Example metrics (customize for your research question):
metrics.register(
    "logit_diff",
    lambda patched, clean, corrupted, correct_token, incorrect_token, **kw:
        float(patched[0, correct_token] - patched[0, incorrect_token])
        - float(corrupted[0, correct_token] - corrupted[0, incorrect_token]),
    description="Logit difference recovery between correct and incorrect tokens",
)

# ============================================================
# DATASET — Texts for analysis
# ============================================================
dataset = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of Japan is",
]

# ============================================================
# EXPERIMENT INPUTS — Clean/corrupted pairs
# ============================================================
clean_text = "The Eiffel Tower is in"
corrupted_text = "The Colosseum is in"
'''

PROGRAM_TEMPLATE = '''# Research Program

## Question
{research_question}

## Constraints
- Focus on layers 0-{n_layers}, all attention heads and MLPs
- Use activation patching first, then narrow with attribution patching
- Each experiment should complete in under 2 minutes
- Keep experiments that identify important components OR rule them out
- Do NOT modify setup.py

## Available Tools (from mlxterp)
- `activation_patching(model, clean, corrupted, component, metric)` — find important layers
- `direct_logit_attribution(model, text)` — per-component logit contributions
- `attribution_patching(model, clean, corrupted)` — fast approximate patching
- `path_patching(model, clean, corrupted, sender, receiver)` — edge-level analysis
- `acdc(model, clean, corrupted, threshold)` — automated circuit discovery
- `model.generate(prompt, interventions=...)` — generation with interventions
- `ResidualStreamAccessor(trace.activations)` — residual stream decomposition

## Current Priorities
1. Run logit lens or DLA to identify where the answer first appears
2. Run activation patching across all layers to find critical ones
3. For critical layers, identify specific heads via head-level patching
4. Test causal paths between identified components
5. Summarize the discovered circuit

## Stop When
- You have identified a circuit (set of components) sufficient for the task
- OR you have run {max_experiments} experiments without convergence

## How to Run an Experiment
1. Read this file and results.jsonl for context
2. Write your experiment in experiment.py
3. Run it: `python experiment.py`
4. Log results to results.jsonl using ExperimentLog
5. If informative, commit to findings/
'''

EXPERIMENT_TEMPLATE = '''"""
AutoInterp Experiment — AGENT-OWNED

The agent writes experiments here. Each experiment should:
1. State a hypothesis
2. Run an analysis
3. Log results
4. Print a conclusion

Run: python experiment.py
"""

import time
from setup import model, metrics, clean_text, corrupted_text, dataset
from mlxterp.causal import activation_patching, direct_logit_attribution
from mlxterp.autointerpret import ExperimentLog, ExperimentEntry

# ============================================================
# EXPERIMENT — Edit below this line
# ============================================================

log = ExperimentLog("results.jsonl")
start_time = time.time()

# Example: Run DLA to see which layers contribute to the prediction
hypothesis = "Initial scan: which layers contribute to the final prediction?"

result = direct_logit_attribution(model, clean_text)

conclusion = (
    f"Target token: '{result.target_token_str}'. "
    f"Top attention contributions: {result.head_contributions.tolist()[:5]}... "
    f"Top MLP contributions: {result.mlp_contributions.tolist()[:5]}..."
)

duration = time.time() - start_time

# Log the experiment
log.append(ExperimentEntry(
    hypothesis=hypothesis,
    method="direct_logit_attribution",
    result={"target": result.target_token_str, "target_id": result.target_token},
    informative=True,
    conclusion=conclusion,
    duration_seconds=duration,
))

print(f"Experiment complete ({duration:.1f}s)")
print(f"Conclusion: {conclusion}")
print(f"Total experiments: {log.count}")
'''

CLAUDE_MD_TEMPLATE = '''# CLAUDE.md

## AutoInterp Project

This is an AutoInterp project — a Karpathy AutoResearch-style ratchet loop
adapted for mechanistic interpretability.

### Three-File Contract
- **setup.py** — IMMUTABLE. Loads model, defines metrics, dataset. Do NOT modify.
- **experiment.py** — AGENT-OWNED. Write your experiments here.
- **program.md** — HUMAN-OWNED. Research question and constraints. Read before each experiment.

### Workflow
1. Read `program.md` for research priorities
2. Read `results.jsonl` for past findings (use `ExperimentLog("results.jsonl").summary()`)
3. Propose a hypothesis based on what's known/unknown
4. Write the experiment in `experiment.py`
5. Run: `python experiment.py`
6. If informative → commit finding to `findings/`
7. If uninformative → log it and move on
8. Repeat

### Key Libraries
```python
from mlxterp.causal import (
    activation_patching, direct_logit_attribution,
    attribution_patching, path_patching, acdc,
    ResidualStreamAccessor,
)
from mlxterp.autointerpret import ExperimentLog, ExperimentEntry
from setup import model, metrics, clean_text, corrupted_text, dataset
```

### Rules
- Do NOT modify setup.py
- Log every experiment (informative or not) to results.jsonl
- Each experiment should complete in under 2 minutes
- Read program.md before starting a new experiment cycle
'''


def init_autointerpret(
    output_dir: str = "autointerpret",
    model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
    research_question: str = "How does this model recall factual associations?",
    max_experiments: int = 100,
    n_layers: int = 15,
) -> str:
    """
    Generate an AutoInterp project scaffold.

    Creates the three-file contract directory structure that any LLM coding
    agent (Claude Code, Cursor, Aider, etc.) can pick up and run.

    Args:
        output_dir: Directory to create (default: "autointerpret")
        model_name: MLX model name for setup.py
        research_question: Research question for program.md
        max_experiments: Max experiments before stopping
        n_layers: Number of layers in the model (for program.md)

    Returns:
        Path to the created directory
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "findings").mkdir(exist_ok=True)

    # setup.py
    setup_content = SETUP_TEMPLATE.format(model_name=model_name)
    (out / "setup.py").write_text(setup_content)

    # program.md
    program_content = PROGRAM_TEMPLATE.format(
        research_question=research_question,
        n_layers=n_layers,
        max_experiments=max_experiments,
    )
    (out / "program.md").write_text(program_content)

    # experiment.py
    (out / "experiment.py").write_text(EXPERIMENT_TEMPLATE)

    # CLAUDE.md
    (out / "CLAUDE.md").write_text(CLAUDE_MD_TEMPLATE)

    # Empty results.jsonl
    (out / "results.jsonl").touch()

    return str(out.resolve())
