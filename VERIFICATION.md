# Verification Guide

How to verify that mlxterp's agentic interpretability features actually work.

## Levels of Verification

| Level | What It Proves | How |
|-------|---------------|-----|
| **1. Unit tests** | Mechanics are correct | `pytest` |
| **2. Integration tests** | Full pipeline works with real model | `pytest -m slow --no-cov` |
| **3. Demo script** | End-to-end demonstration | `python examples/agentic_demo.py` |
| **4. AutoInterp dry-run** | Ratchet loop works | `python examples/autointerpret_dryrun.py` |
| **5. Claude Code self-test** | Zero-orchestration mode works | Manual — see below |
| **6. Real research output** | The library produces real findings | See `examples/findings/SYCOPHANCY_FINDINGS.md` |

---

## Level 1: Unit Tests (fastest, always-on)

```bash
pytest
```

Expected: **261 tests pass**, all green, ~3-5 seconds.

What it verifies: every public function has behavioral tests with synthetic models. The mechanics of patching, results, workflows, AutoInterp logging, etc. are all correct.

---

## Level 2: Integration Tests (slow, real model)

```bash
# Run only the slow e2e tests (skipped by default)
pytest tests/test_e2e_agentic.py -m slow --no-cov -v
```

Expected: ~8 tests pass, ~2-5 minutes on Apple Silicon (downloads pythia-410m on first run, cached after).

Override the model:
```bash
MLXTERP_TEST_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit pytest -m slow --no-cov
```

What it verifies:
- Research workflows (`behavior_localization`, `circuit_discovery`) work on real models
- AutoInterp scaffold creates valid projects
- AutoInterp runner logs experiments correctly, including failure handling
- Report generation produces valid Markdown and HTML
- Full pipeline (workflow → scaffold → runner → report) works end-to-end

---

## Level 3: Demo Script

```bash
python examples/agentic_demo.py
```

Expected runtime: 2-5 minutes. Outputs to `./demo_output/`:
- `investigation_report.md` — Markdown report
- `investigation_report.html` — HTML report
- `autointerpret_scaffold/` — full three-file-contract project
- `autointerpret_run/results.jsonl` — 5 experiment entries
- `autointerpret_run/findings/` — kept findings

Pass criteria (printed at end):
```
  [x] Markdown report generated
  [x] HTML report generated
  [x] AutoInterp scaffold created
  [x] 5 experiments logged (5)
  [x] Behavior localization workflow ran
  [x] Circuit discovery workflow ran
```

All 6 checkboxes should be `[x]`.

---

## Level 4: AutoInterp Dry-Run

```bash
python examples/autointerpret_dryrun.py
```

Expected runtime: 1-3 minutes. Runs 10 experiments (9 real, 1 deliberately failing) and verifies:

```
[Phase 1] Scaffold generation        — 8 checks
[Phase 2] Model loading              — 2 checks
[Phase 3] Programmatic runner        — runs 10 experiments
[Phase 4] Verifying invariants       — 10 checks
```

Expected final output:
```
AutoInterp dry-run: ALL CHECKS PASSED
```

Exit code `0` on success, `1` if any invariant fails.

What it verifies:
- Scaffold has all expected files with correct content
- Runner handles both successful and failing experiments
- `results.jsonl` is valid JSONL with required fields
- `findings/` contains exactly the informative entries
- `ExperimentLog.summary()` reports correct counts
- Failed experiments are logged but not saved as findings

---

## Level 5: Claude Code Self-Test (manual)

The zero-orchestration mode. The library is ready when an LLM agent can pick it up and run an investigation without instructions.

### Procedure

1. **Generate a scaffold:**
   ```bash
   python -c "
   from mlxterp.autointerpret import init_autointerpret
   init_autointerpret(
       output_dir='selftest',
       model_name='EleutherAI/pythia-410m',
       research_question='Investigate how the model handles negation.'
   )"
   ```

2. **Open in Claude Code:**
   ```bash
   cd selftest
   claude
   ```

3. **Prompt:**
   > Read program.md and start investigating. Use mlxterp. Run at least 5 experiments and log them to results.jsonl.

### Success Criteria

After Claude Code runs:

- [ ] `results.jsonl` has ≥ 5 entries
- [ ] At least 1 entry in `findings/`
- [ ] Claude's experiments use mlxterp's actual APIs (not reinvented)
- [ ] Claude writes conclusions that reference real layer indices / components
- [ ] Claude doesn't modify `setup.py`
- [ ] `python -c "from mlxterp.autointerpret import ExperimentLog; print(ExperimentLog('results.jsonl').summary())"` produces a sensible summary

### Expected Behavior

Claude should:
1. Read `program.md` to understand the research question
2. Read `CLAUDE.md` to understand the three-file contract
3. Run a DLA or patching experiment first to get oriented
4. Log the result
5. Form a hypothesis based on what it found
6. Run follow-up experiments
7. Stop when convergence or after ~5-10 experiments

If Claude gets confused or invents APIs that don't exist, the library isn't agent-friendly enough.

---

## Level 6: Real Research Output

The ultimate test: can mlxterp produce interpretable, defensible findings about a real model?

See **`examples/findings/SYCOPHANCY_FINDINGS.md`** — a complete causal investigation of sycophancy in pythia-410m, run by Claude Code using mlxterp:

- **Confirmed sycophancy** in 2/3 probes (gold, math)
- **Localized the circuit** to:
  - MLP L0 (user context encoding)
  - Attention L5–8 (belief routing)
  - Attention L13/18/21 (answer writing)
  - MLP L23 (final amplification)
- **With figures:** 4 publication-quality plots
- **Reproducible:** single-script, 3 minutes on Apple Silicon
- **Agent-driven:** entire investigation designed and run by Claude Code

This is proof that the library isn't just "working" — it's usable for real interpretability research.

---

## LLM Labeling (optional, requires API)

Auto-labeling SAE features requires the Anthropic SDK and an API key:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...

python -c "
from mlxterp.auto_interp import auto_label_feature
# ... requires a trained SAE and dataset
"
```

Without the API, `auto_label_feature` gracefully returns a placeholder label. The mechanics (prompt construction, response parsing) are unit-tested with mocks.

---

## Common Issues

### "Model loading fails"
- Check HuggingFace auth: `huggingface-cli login` if using gated models
- Try a different model: `MLXTERP_TEST_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit`

### "Attention weight warnings"
- Harmless. Some architectures (GPT-NeoX/pythia) use fused `query_key_value` projection. mlxterp can't extract per-head weights but all patching still works.

### "Coverage fails"
- Expected. The 95% coverage threshold includes legacy code (SAE trainer, tuned_lens) that pre-dates this work. New code has high coverage.
- For quick verification, run with `--no-cov`: `pytest --no-cov`

---

## Summary

| Verification | Command | Time | Proves |
|--------------|---------|------|--------|
| Unit tests | `pytest` | ~5s | Mechanics |
| E2E tests | `pytest -m slow --no-cov` | 2-5m | Real model works |
| Demo | `python examples/agentic_demo.py` | 2-5m | Full stack |
| Dry-run | `python examples/autointerpret_dryrun.py` | 1-3m | AutoInterp |
| Self-test | Manual, Claude Code | 5-15m | Agent-friendly |
| Sycophancy | `examples/findings/SYCOPHANCY_FINDINGS.md` | - | Real research |
