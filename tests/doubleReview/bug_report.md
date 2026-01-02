# mlxterp docs review: bug report and recommendations

## Scope
Reviewed all Python snippets under `docs/` against current implementation in `mlxterp/`.

## High-severity issues (likely runtime failures)
- `Trace.saved_values` is copied before any `.save()` calls, so values saved inside a `with model.trace(...)` block are not available afterward. This breaks examples using `trace.saved_values` and `trace.get(...)`. (`mlxterp/core/trace.py:59`, `mlxterp/core/trace.py:91`, `mlxterp/core/trace.py:427`)
- `Trace._process_inputs` treats `List[int]` as a 1D tensor (no batch dimension). Most MLX LM forwards expect `(batch, seq_len)`, so list-of-ints inputs can fail or behave incorrectly. This contradicts the API docs. (`mlxterp/core/trace.py:387`, `mlxterp/core/trace.py:420`, `docs/API.md:70`)
- Many docs use `.attn`, but mlx-lm Llama uses `.self_attn`. These snippets will raise `AttributeError`. (`docs/index.md:27`, `docs/QUICKSTART.md:82`, `docs/examples.md:51`, `docs/API.md:84`, `docs/API.md:631`, `docs/API.md:651`, `docs/architecture.md:112`)
- `logit_lens(position=...)` is referenced in docs but not implemented. The example’s result indexing also doesn’t match the method’s return shape. (`docs/examples.md:386`)

## Medium-severity issues (functional mismatches/incorrect output)
- `get_token_predictions` claims to accept `(batch, hidden_dim)` but the implementation assumes 1D; batched input produces incorrect shapes or indexing. (`mlxterp/analysis.py:28`, `mlxterp/analysis.py:101`, `mlxterp/analysis.py:115`)
- `activation_patching` and `logit_lens` hardcode `model.model.*` paths, so they fail for custom or non-mlx-lm models. This conflicts with “model-agnostic” docs. (`mlxterp/analysis.py:183`, `mlxterp/analysis.py:191`, `mlxterp/analysis.py:462`)
- `activation_patching` advertises `component="output"`, but activations are captured under module names only, so `.output` is never a key. (`mlxterp/analysis.py:366`, `mlxterp/analysis.py:468`)
- `collect_activations` ignores the `layers` filter and returns all activations; cache keys are raw activation names (e.g., `model.model.layers.3`), not `layers.3`. (`mlxterp/core/cache.py:53`, `mlxterp/core/cache.py:78`, `mlxterp/core/cache.py:95`, `docs/API.md:1009`)
- Docs claim trace activations are cleared after context exit, but the `Trace` object keeps its copied activations. (`docs/JUPYTER_GUIDE.md:272`)
- Architecture docs say forward pass happens in `__exit__`, but it executes in `__enter__`. (`docs/architecture.md:121`)

## Low-severity issues (docs inconsistencies)
- SAE component examples use `component="attn"` and `component="output"`; for mlx-lm Llama, the key is `self_attn`, and `output` isn’t captured. (`docs/guides/dictionary_learning.md:387`)
- `Trace.get_activation("layers.3.attn")` uses a short key name; actual activation keys include model prefixes and use `self_attn`. (`docs/API.md:623`)

## Recommendations
1. Fix `Trace.saved_values` to reflect saved items after the trace (copy from `TraceContext` in `__exit__` or keep a reference).
2. In `_process_inputs`, wrap `List[int]` as `[tokens]` to preserve batch dimension; optionally pad with a tokenizer pad ID.
3. Update `get_token_predictions` to handle batched hidden states or narrow the docstring to 1D only.
4. Make `activation_patching` and `logit_lens` use detected layer paths (`self._layer_attr`) instead of hardcoded `model.model.*`.
5. Remove or implement `component="output"` support in `activation_patching` docs.
6. Fix `collect_activations` to respect `layers` and document/normalize cache keys.
7. Update docs to use `self_attn`, correct `logit_lens` usage, and remove claims about cleared activations / `__exit__` execution.

## Notes
No code changes were applied during this review.

## Real-model confirmation (mlx-community/Llama-3.2-1B-Instruct-4bit)

Tests added in `tests/doubleReview/confirmationtests/test_real_model_confirmations.py` and executed with the real mlx-lm model.

Run summary:
- 11 tests collected
- 2 passed
- 9 xfailed (expected failures that confirm the mismatches)

Passed confirmations:
- `.self_attn` exists and `.attn` does not (docs should use `self_attn`).
- Activation keys are model-prefixed; `layers.0.self_attn` returns None while `model.model.layers.0.self_attn` returns data.

Xfailed confirmations (expected failures that reproduce documented issues):
- `Trace.saved_values` does not include values saved inside the context.
- List-of-int inputs are not wrapped to `(1, seq_len)` and fail on real model input handling.
- `trace.activations` is not cleared after context exit.
- Forward pass occurs in `__enter__`, so activations are present inside the context (contrary to docs).
- `logit_lens(position=...)` is not supported (TypeError).
- `get_token_predictions` does not handle batched hidden states.
- `collect_activations` cache keys do not match documented short names.
- `collect_activations` ignores the `layers` filter and returns all activations.
- `activation_patching(component=\"output\")` returns no layer results because no `.output` activation key exists.

Full test output recorded via pytest in this environment.

Not reproducible with the real model (requires custom/non-mlx-lm model paths):
- Hardcoded `model.model.*` paths in `activation_patching` and `logit_lens` only fail on non-mlx-lm models; the real mlx-lm model uses `model.model.*` and does not surface the issue.


  Test Results: 2 passed, 9 xfailed (expected failures confirming bugs)

  All bugs are still valid with the real Llama-3.2-1B-Instruct-4bit model:

  Summary Table

  | Issue                              | Test                                            | Status    | Still Valid?                    |
  |------------------------------------|-------------------------------------------------|-----------|---------------------------------|
  | .self_attn exists, .attn missing   | test_self_attn_exists_attn_missing              | ✅ PASSED | Yes - confirms docs need fixing |
  | Activation keys are model-prefixed | test_trace_get_activation_keying_real_model     | ✅ PASSED | Yes - confirms key mismatch     |
  | saved_values copied too early      | test_trace_saved_values_real_model              | XFAIL     | ✅ Yes                          |
  | List[int] not batched              | test_list_int_inputs_real_model                 | XFAIL     | ✅ Yes                          |
  | Activations not cleared after exit | test_trace_activations_cleared_after_context    | XFAIL     | ✅ Yes (docs wrong)             |
  | Forward in __enter__ not __exit__  | test_trace_context_exec_timing                  | XFAIL     | ✅ Yes (docs wrong)             |
  | logit_lens(position=...) missing   | test_logit_lens_position_param                  | XFAIL     | ✅ Yes                          |
  | get_token_predictions batching     | test_get_token_predictions_batched              | XFAIL     | ✅ Yes                          |
  | collect_activations key mismatch   | test_collect_activations_keying                 | XFAIL     | ✅ Yes                          |
  | collect_activations ignores filter | test_collect_activations_respects_layers_filter | XFAIL     | ✅ Yes                          |
  | component="output" not captured    | test_activation_patching_output_component       | XFAIL     | ✅ Yes                          |

  Categorized Issues

  Code Bugs (need fixing)

  | Bug                                          | Severity | Affects             |
  |----------------------------------------------|----------|---------------------|
  | saved_values copied before .save()           | High     | All models          |
  | List[int] inputs not batched to (1, seq)     | High     | All models          |
  | collect_activations ignores layers filter    | Medium   | All models          |
  | collect_activations keys don't match docs    | Medium   | All models          |
  | get_token_predictions doesn't handle batches | Medium   | Batch use cases     |
  | logit_lens(position=...) not implemented     | Medium   | Specific positions  |
  | component="output" not captured              | Low      | activation_patching |

  Documentation Bugs (need updating)

  | Issue                                         | Location                                                      |
  |-----------------------------------------------|---------------------------------------------------------------|
  | Use .attn instead of .self_attn               | index.md, QUICKSTART.md, examples.md, API.md, architecture.md |
  | Claims activations cleared after context      | JUPYTER_GUIDE.md                                              |
  | Claims forward pass in __exit__               | architecture.md                                               |
  | Short activation keys like layers.0.self_attn | API.md                                                        |

  Conclusion

  All 11 issues from the bug report are still valid. The real model tests confirm:

  1. 9 bugs reproduce (XFAIL tests failed as expected)
  2. 2 tests passed confirming the mismatch exists (docs say .attn, model has .self_attn; docs say short keys, actual keys are prefixed)
