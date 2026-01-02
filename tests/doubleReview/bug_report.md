# mlxterp docs review: bug report and recommendations

## Scope
Re-verified previously reported issues against the current codebase and docs, including synthetic and real-model confirmation tests.

## Status
All previously reported issues are now fixed and verified.

## Fixes Verified
- `Trace.saved_values` now captures values saved inside the context by copying in `__exit__`, while activations remain available inside the context.
- `List[int]` token inputs are wrapped into a batch dimension.
- `collect_activations` respects layer filters and normalizes keys to short form.
- `get_token_predictions` handles batched hidden states.
- `logit_lens` supports `position` for targeted analysis.
- `activation_patching` supports custom model paths, `component="output"`, and division-by-zero protection.
- Docs updated: `.attn` -> `.self_attn`, trace timing in `architecture.md`, activation clearing in `JUPYTER_GUIDE.md`.

## Test Results
Ran:
- `pytest tests/doubleReview/confirmationtests -q --no-cov`

Outcome:
- 16 tests collected
- 16 passed

Notes:
- Real-model confirmations used `mlx-community/Llama-3.2-1B-Instruct-4bit`.
- The confirmation tests that previously xfailed now pass with the fixes applied.

## Recommendations
- Continue monitoring custom/non-mlx-lm model path coverage as new architectures are added.
