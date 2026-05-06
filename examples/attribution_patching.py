"""Attribution patching demo + correlation check vs brute-force patching.

The point of attribution patching is that it's a fast linear
approximation of the brute-force "patch each layer separately and
measure metric" loop. So we don't expect identical numbers, but we do
expect the layer rankings to correlate.

This script:
  1. Picks a small factual-recall pair: "Paris is the capital of France"
     vs "London is the capital of France".
  2. Runs full activation_patching across all layers (slow path).
  3. Runs attribution_patching across all layers (gradient path, fast).
  4. Reports both, plus the Spearman rank correlation between them.

On Llama-3.2-1B-Instruct-4bit you should expect a clearly positive
correlation — early layers near zero, mid/late layers showing the
factual-recall signal in both methods, with attribution patching
finishing in a fraction of the wall-clock time of brute force.
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")
warnings.filterwarnings("ignore", message=r".*head_dim.*", module=".*")

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel


MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
CLEAN = "Paris is the capital of France"
CORRUPTED = "London is the capital of France"


def spearman(xs, ys):
    """Spearman rank correlation. n is small; readable beats vectorised."""
    n = len(xs)
    if n == 0:
        return float("nan")
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rank_x = [0] * n
    rank_y = [0] * n
    for r, i in enumerate(rx):
        rank_x[i] = r
    for r, i in enumerate(ry):
        rank_y[i] = r
    mean_x = (n - 1) / 2
    mean_y = (n - 1) / 2
    num = sum((rank_x[i] - mean_x) * (rank_y[i] - mean_y) for i in range(n))
    dx = sum((rank_x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    dy = sum((rank_y[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main():
    print(f"Loading {MODEL}...")
    base, tok = load(MODEL)
    model = InterpretableModel(base, tokenizer=tok)
    n_layers = len(model.layers)
    print(f"Model has {n_layers} layers.")
    print()

    print(f"clean:     {CLEAN!r}")
    print(f"corrupted: {CORRUPTED!r}")
    print()

    # 1. Brute-force patching at the layer output (resid_post) with
    #    logit_diff(target=" Paris", foil=" London") so the metric
    #    matches what attribution_patching computes. (The existing
    #    activation_patching uses an L2-recovery metric and saturates
    #    at 100% when patching the full residual; using logit_diff
    #    gives a real per-layer signal.)
    from mlxterp import interventions as iv

    target_ids = tok.encode(" Paris", add_special_tokens=False)
    foil_ids = tok.encode(" London", add_special_tokens=False)
    if tok.bos_token_id is not None:
        if target_ids and target_ids[0] == tok.bos_token_id:
            target_ids = target_ids[1:]
        if foil_ids and foil_ids[0] == tok.bos_token_id:
            foil_ids = foil_ids[1:]
    target_id = target_ids[0]
    foil_id = foil_ids[0]

    def logit_diff_at_last(text, interventions=None):
        with model.trace(text, interventions=interventions or {}):
            logits = model.output.save()
        mx.eval(logits)
        return float(logits[0, -1, target_id] - logits[0, -1, foil_id])

    clean_metric = logit_diff_at_last(CLEAN)
    corrupted_metric = logit_diff_at_last(CORRUPTED)
    print(
        f"clean logit_diff:     {clean_metric:.4f}   "
        f"corrupted logit_diff: {corrupted_metric:.4f}\n"
    )

    # Patching the *full* residual at any layer cascades through the
    # rest of the model and saturates the recovery, so layer-by-layer
    # differences vanish. Patch only the last position to get a
    # localised signal that we can compare to attribution_patching's
    # per-layer numbers.
    print("Brute-force last-position patching with logit_diff metric...")
    t0 = time.perf_counter()
    act_results = {}
    for i in range(n_layers):
        with model.trace(CLEAN) as t:
            for path in (
                f"model.model.layers.{i}",
                f"model.layers.{i}",
                f"layers.{i}",
            ):
                if path in t.activations:
                    clean_act = t.activations[path]
                    if path.startswith("model.model."):
                        iv_key = path[12:]
                    elif path.startswith("model."):
                        iv_key = path[6:]
                    else:
                        iv_key = path
                    break
        mx.eval(clean_act)

        def patch_last(x, ca=clean_act):
            # Replace just the last position of the corrupted activation
            # with the last position of the clean activation.
            out = mx.array(x)
            out[..., -1, :] = ca[..., -1, :]
            return out

        patched = logit_diff_at_last(CORRUPTED, interventions={iv_key: patch_last})
        act_results[i] = patched - corrupted_metric
    act_secs = time.perf_counter() - t0
    print(f"  done in {act_secs:.2f}s\n")

    # 2. Attribution patching with logit_diff(target=' Paris', foil=' London').
    print("Gradient-based attribution_patching across all layers...")
    t0 = time.perf_counter()
    attr_results = model.attribution_patching(
        clean_text=CLEAN,
        corrupted_text=CORRUPTED,
        target_token=" Paris",
        foil_token=" London",
        layers=None,
    )
    attr_secs = time.perf_counter() - t0
    print(f"  done in {attr_secs:.2f}s\n")

    common_layers = sorted(set(act_results) & set(attr_results))
    print(f"layers in both runs: {len(common_layers)}\n")

    print(f"{'layer':>5} | {'patch_effect':>14} | {'attribution':>14}")
    print("-" * 44)
    for i in common_layers:
        print(f"{i:>5} | {act_results[i]:>14.4f} | {attr_results[i]:>14.4f}")
    print()

    rho = spearman(
        [act_results[i] for i in common_layers],
        [attr_results[i] for i in common_layers],
    )
    print(f"Spearman rank correlation: {rho:.3f}")
    print(f"Brute-force time:    {act_secs:.2f}s")
    print(f"Attribution time:    {attr_secs:.2f}s")
    if attr_secs > 0:
        print(f"Speedup:             {act_secs / attr_secs:.2f}x")
    print()
    print(
        "Interpretation: attribution patching is a first-order Taylor "
        "approximation of patching, so we expect a positive (often ~0.5-0.9) "
        "rank correlation, not identical numbers. The win is wall-clock: "
        "one forward + one backward instead of one forward per layer."
    )


if __name__ == "__main__":
    main()
