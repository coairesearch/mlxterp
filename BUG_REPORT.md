# Bug Report for mlxterp v0.0.1 Release

Generated: 2026-01-01

This report documents all bugs, issues, and errors discovered during comprehensive testing of the mlxterp library. Testing included running all test files and validating all code examples from the mkdocs documentation.

---

## Summary

| Priority | Issue Count |
|----------|-------------|
| Critical | 2 |
| High | 3 |
| Medium | 4 |
| Low | 2 |

---

## Critical Issues

### 1. SAE Feature Analysis Shape Mismatch

**File:** `mlxterp/sae_mixin.py` (lines 285-294)

**Description:** The `get_top_features_for_text()` and `get_top_texts_for_feature()` methods incorrectly handle activation tensor shapes. The activation tensor from `trace.activations` has shape `(batch, seq_len, d_model)` but the code assumes `(seq_len, d_model)`.

**Current code (line 288):**
```python
activations = trace.activations[activation_key]  # Comment says: (seq_len, d_model)
activations_3d = activations[:, None, :]  # Creates wrong shape!
```

**Actual shapes:**
- `activations` shape: `(1, 7, 2048)` - batch=1, seq_len=7, d_model=2048
- `activations_3d` shape: `(1, 1, 7, 2048)` - WRONG, should be `(7, 1, 2048)`
- `features` shape: `(1, 1, 7, 8192)` - WRONG, should be `(7, 1, 8192)`

**Symptoms:**
- `get_top_features_for_text()` returns incorrect results (sometimes 0 features, sometimes wrong count)
- `get_top_texts_for_feature()` fails with error: `[squeeze] Cannot squeeze axis 2 with size 0`

**Fix required:**
```python
# Correct approach:
activations = trace.activations[activation_key]  # (batch, seq_len, d_model)
# Remove batch dimension and add SAE batch dimension
activations_2d = activations[0]  # (seq_len, d_model)
activations_3d = activations_2d[:, None, :]  # (seq_len, 1, d_model) - correct!
```

---

### 2. Activation Patching Broadcast Error with Different Sequence Lengths

**File:** `mlxterp/core/intervention.py` (line 78) and `mlxterp/analysis.py`

**Description:** When using `iv.replace_with()` or `activation_patching()` with texts of different token lengths, the intervention fails with a broadcast error.

**Error message:**
```
ValueError: [broadcast_shapes] Shapes (1,3,2048) and (1,4,2048) cannot be broadcast.
```

**Example that fails (from examples.md):**
```python
# Clean text = 3 tokens, Corrupted text = 4 tokens
with model.trace("Clean text") as trace:
    clean_mlp = trace.activations["model.model.layers.10.mlp"]

with model.trace("Corrupted text",
                interventions={"layers.10.mlp": iv.replace_with(clean_mlp)}):
    patched = model.output.save()  # FAILS!
```

**Fix required:** Add shape validation or padding/truncation logic in `replace_with()` intervention.

---

## High Priority Issues

### 3. KL Divergence Patching Test Returns NaN

**File:** `tests/test_activation_patching_kl.py`

**Description:** The KL divergence-based activation patching test produces NaN values for all layers.

**Output:**
```
Layer 0: KL = nan
Layer 1: KL = nan
...
Baseline KL: nan
```

**Root cause:** KL divergence is numerically unstable with large vocabularies (128k tokens). With many near-zero probabilities, `log(0)` produces `-inf`.

**Impact:** Users following documentation suggesting KL divergence will get NaN results.

**Recommendation:** Document this limitation and recommend L2, MSE, or Cosine metrics instead.

---

### 4. Test Expects Wrong Default SAEConfig Values

**File:** `tests/test_sae_basic.py` (line 22)

**Description:** Test asserts `config.expansion_factor == 16` but actual default is `32`.

**Test failure:**
```
AssertionError: assert 32 == 16
```

**Fix required:** Update test to expect `expansion_factor == 32` or update SAEConfig default.

---

### 5. Test Expects Wrong SAE Stats Key Name

**File:** `tests/test_sae_basic.py` (line 225)

**Description:** Test asserts `"l1" in stats` but the actual key is `"l1_magnitude"`.

**Test failure:**
```
AssertionError: assert 'l1' in {'l0': 50.0, 'l0_sparsity': ..., 'l1_magnitude': ..., ...}
```

**Fix required:** Change assertion to check for `"l1_magnitude"`.

---

## Medium Priority Issues

### 6. Numerical Instability with Long Texts in Activation Patching

**File:** `mlxterp/analysis.py`

**Description:** When clean and corrupted texts have significantly different lengths, activation patching with L2 metric fails.

**Error:**
```
[broadcast_shapes] Shapes (1,22,2048) and (1,21,2048) cannot be broadcast.
```

**Occurs in:** `test_numerical_stability()` in `test_distance_metrics.py` with "Long texts" test case.

**Fix required:** Handle sequence length mismatches gracefully (e.g., compare last tokens only, or pad/truncate).

---

### 7. Nested Module Access Sometimes Fails

**File:** `tests/test_nested_modules.py`

**Description:** Accessing nested modules sometimes fails with:
```
[convert] Only length-1 arrays can be converted to Python scalars.
```

**Status:** Error is caught but the feature doesn't work in all cases.

---

### 8. test_sae_integration.py Skipped

**File:** `tests/test_sae_integration.py`

**Description:** All tests are skipped with:
```
1 skipped in 0.19s
```

**Possible cause:** Missing skip condition or pytest configuration issue.

---

### 9. Patching __call__ Doesn't Work

**File:** `tests/test_patching.py`

**Description:** Test confirms that `__call__` patching doesn't work:
```
Call count: 0
❌ Patching __call__ doesn't work!
```

**Status:** This is documented behavior, but could be confusing for users.

---

## Low Priority Issues

### 10. Intervention Difference Shows as "inf"

**File:** `tests/test_comprehensive.py`

**Description:** Intervention test shows "inf" for difference norm:
```
Difference norm: inf
```

**Impact:** Makes it unclear if intervention worked correctly. The test passes but output is misleading.

---

### 11. Documentation Examples Use Wrong Activation Key

**File:** `docs/examples.md`, `docs/guides/dictionary_learning.md`

**Description:** Some documentation examples use `trace.activations["model.layers.10.mlp"]` but actual key format is `trace.activations["model.model.layers.10.mlp"]` (double "model").

**Examples affected:**
- Line 58 in dictionary_learning.md: `trace.activations["model.layers.10.mlp"]`
- Line 549-551 in dictionary_learning.md

**Impact:** Users copying examples will get KeyError.

---

## Test Results Summary

### Passing Tests

| Test File | Status |
|-----------|--------|
| test_utilities.py | PASSED (7/7) |
| test_distance_metrics.py | PASSED (7/7) |
| test_sae_feature_analysis_methods.py | PASSED (5/5)* |
| test_comprehensive.py | PASSED |
| test_real_model.py | PASSED |
| test_interventions.py | PASSED |
| test_activation_validity.py | PASSED |
| test_tokenizer_methods.py | PASSED |
| test_logit_lens.py | PASSED |
| test_activation_patching_helper.py | PASSED |
| test_activation_patching_simple.py | PASSED |
| test_activation_patching_mlp.py | PASSED |
| test_proxy_api.py | PASSED |
| test_what_we_capture.py | PASSED |
| test_check_wrappers.py | PASSED |

*Note: SAE feature analysis tests pass but with workarounds/warnings due to shape issues.

### Failing Tests

| Test File | Failures | Issue |
|-----------|----------|-------|
| test_sae_basic.py | 2/20 | Wrong default config value, wrong stats key |
| test_activation_patching_kl.py | All | NaN values from KL divergence |
| test_nested_modules.py | 1 | Array conversion error |
| test_patching.py | 1 | __call__ patching doesn't work |

---

## Documentation Code Validation

All code examples from mkdocs documentation were tested. Results:

| Example | Status | Notes |
|---------|--------|-------|
| Basic imports | PASSED | |
| Simple model tracing | PASSED | |
| get_activations | PASSED | |
| Tokenizer methods | PASSED | |
| Logit lens | PASSED | |
| get_token_predictions | PASSED | |
| scale intervention | PASSED | |
| clamp intervention | PASSED | |
| noise intervention | PASSED | |
| add_vector intervention | PASSED | |
| replace_with intervention | FAILED | Broadcast error with different lengths |
| compose intervention | PASSED | |
| activation_patching helper | PASSED | With same-length texts |
| batch_get_activations | PASSED | |

---

## Recommendations for v0.0.1 Release

### Must Fix (Critical)

1. **SAE shape mismatch** - Users following SAE feature analysis guide will get incorrect results
2. **Broadcast error with replace_with** - Core functionality broken for common use case

### Should Fix (High)

3. Update test assertions to match actual implementation (SAEConfig defaults, stats keys)
4. Document KL divergence limitations clearly

### Nice to Have (Medium/Low)

5. Add sequence length handling for activation patching
6. Fix documentation activation key examples
7. Clean up "inf" output in test logs

---

## Action Items

- [ ] Fix SAE shape handling in `sae_mixin.py`
- [ ] Add shape validation to `replace_with()` intervention
- [ ] Update `test_sae_basic.py` assertions
- [ ] Add warning/documentation about KL divergence numerical issues
- [ ] Update documentation examples with correct activation keys
- [ ] Consider adding sequence padding/truncation for activation patching
