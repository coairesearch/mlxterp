"""
AutoInterp dry-run: verify the ratchet loop with a real model.

Exercises the full AutoInterp pipeline:
1. Generate scaffold (three-file contract)
2. Run 10 programmatic experiments
3. Verify results.jsonl is populated
4. Verify findings/ directory contains kept findings
5. Print summary and validate invariants

Usage: python examples/autointerpret_dryrun.py

Expected runtime: 1-3 minutes.
Exit code: 0 on success, 1 if any invariant fails.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import mlx.core as mx
from mlxterp import InterpretableModel
from mlxterp.causal import activation_patching, direct_logit_attribution, attribution_patching
from mlxterp.autointerpret import (
    init_autointerpret,
    AutoInterpret,
    ExperimentLog,
    ExperimentEntry,
)


MODEL_NAME = os.environ.get("MLXTERP_DRYRUN_MODEL", "EleutherAI/pythia-410m")
WORK_DIR = Path(os.environ.get("MLXTERP_DRYRUN_DIR", "./autointerpret_dryrun"))


def check(cond, msg):
    """Assertion with nice output."""
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return cond


def main():
    print("=" * 70)
    print("AutoInterp Dry-Run")
    print("=" * 70)

    all_passed = True

    # ========================================================================
    # Phase 1: Scaffold generation
    # ========================================================================
    print("\n[Phase 1] Scaffold generation")

    scaffold_path = init_autointerpret(
        output_dir=str(WORK_DIR / "scaffold"),
        model_name=MODEL_NAME,
        research_question="What components matter for factual recall?",
        max_experiments=20,
    )

    scaffold = Path(scaffold_path)
    all_passed &= check(scaffold.exists(), f"Scaffold created at {scaffold}")
    all_passed &= check((scaffold / "setup.py").exists(), "setup.py present")
    all_passed &= check((scaffold / "experiment.py").exists(), "experiment.py present")
    all_passed &= check((scaffold / "program.md").exists(), "program.md present")
    all_passed &= check((scaffold / "CLAUDE.md").exists(), "CLAUDE.md present")
    all_passed &= check((scaffold / "results.jsonl").exists(), "results.jsonl present")
    all_passed &= check((scaffold / "findings").is_dir(), "findings/ directory present")

    # Verify content
    setup_content = (scaffold / "setup.py").read_text()
    all_passed &= check(MODEL_NAME in setup_content, f"setup.py references {MODEL_NAME}")

    program_content = (scaffold / "program.md").read_text()
    all_passed &= check(
        "factual recall" in program_content,
        "program.md contains research question",
    )

    # ========================================================================
    # Phase 2: Load model
    # ========================================================================
    print("\n[Phase 2] Model loading")
    t0 = time.time()
    model = InterpretableModel(MODEL_NAME)
    load_time = time.time() - t0
    all_passed &= check(load_time < 60, f"Model loaded in {load_time:.1f}s (< 60s)")
    all_passed &= check(len(model.layers) > 0, f"{len(model.layers)} layers detected")

    # ========================================================================
    # Phase 3: Programmatic runner
    # ========================================================================
    print("\n[Phase 3] Programmatic AutoInterpret runner")

    clean = "The capital of France is"
    corrupted = "The capital of Germany is"

    runner = AutoInterpret(
        model=model,
        clean=clean,
        corrupted=corrupted,
        output_dir=str(WORK_DIR / "runner"),
        max_experiments=10,
    )

    all_passed &= check(runner.n_experiments == 0, "Runner starts with 0 experiments")
    all_passed &= check(not runner.is_done, "Runner is not done initially")

    # Run 10 experiments — a mix of real analyses and a deliberately failing one
    experiments_to_run = [
        ("dla_baseline", lambda: direct_logit_attribution(model, clean),
         "Identify top-contributing layers for the clean prediction"),
        ("mlp_patch_l2", lambda: activation_patching(model, clean, corrupted, component="mlp", metric="l2"),
         "Find causally important MLPs (l2 metric)"),
        ("mlp_patch_kl", lambda: activation_patching(model, clean, corrupted, component="mlp", metric="kl"),
         "Verify MLP importance with KL divergence"),
        ("mlp_patch_cosine", lambda: activation_patching(model, clean, corrupted, component="mlp", metric="cosine"),
         "Cross-validate MLP importance with cosine"),
        ("attn_patch", lambda: activation_patching(model, clean, corrupted, component="attn", metric="l2"),
         "Find causally important attention layers"),
        ("resid_patch", lambda: activation_patching(model, clean, corrupted, component="resid_post", metric="l2"),
         "Full residual stream patching"),
        ("attribution", lambda: attribution_patching(model, clean, corrupted, component="resid_post"),
         "Fast attribution approximation"),
        ("mlp_first_half", lambda: activation_patching(model, clean, corrupted, component="mlp", layers=list(range(12))),
         "Focus on first half of MLPs"),
        ("mlp_second_half", lambda: activation_patching(model, clean, corrupted, component="mlp", layers=list(range(12, 24))),
         "Focus on second half of MLPs"),
        # Deliberately failing experiment — tests error handling
        ("should_fail", lambda: (_ for _ in ()).throw(RuntimeError("intentional failure for dry-run testing")),
         "This should fail gracefully"),
    ]

    for name, fn, hypothesis in experiments_to_run:
        t0 = time.time()
        entry = runner.run_experiment(name=name, fn=fn, hypothesis=hypothesis)
        dt = time.time() - t0
        status = "ok" if entry.informative else "fail"
        print(f"    [{runner.n_experiments:2d}/10] {name}: {status} ({dt:.1f}s)")

    # ========================================================================
    # Phase 4: Verify invariants
    # ========================================================================
    print("\n[Phase 4] Verifying invariants")

    all_passed &= check(runner.n_experiments == 10, f"Ran 10 experiments (got {runner.n_experiments})")

    # results.jsonl should have 10 lines
    results_path = Path(runner.output_dir) / "results.jsonl"
    all_passed &= check(results_path.exists(), "results.jsonl exists")

    with open(results_path) as f:
        lines = [l for l in f.readlines() if l.strip()]
    all_passed &= check(len(lines) == 10, f"results.jsonl has 10 entries (got {len(lines)})")

    # Each line should be valid JSON with required fields
    all_valid = True
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            required = {"experiment_id", "hypothesis", "method", "result", "informative", "conclusion", "timestamp"}
            if not required.issubset(entry.keys()):
                all_valid = False
                break
        except json.JSONDecodeError:
            all_valid = False
            break
    all_passed &= check(all_valid, "All entries are valid JSON with required fields")

    # Findings directory: should have 9 files (informative ones, not the failing one)
    findings_dir = Path(runner.output_dir) / "findings"
    findings = list(findings_dir.glob("*.json"))
    all_passed &= check(
        len(findings) == 9,
        f"findings/ has 9 informative entries (got {len(findings)})",
    )

    # Read back via ExperimentLog
    log = ExperimentLog(str(results_path))
    entries = log.entries()
    all_passed &= check(
        len(entries) == 10,
        f"ExperimentLog.entries() returns 10 (got {len(entries)})",
    )

    informative = log.informative_entries()
    all_passed &= check(
        len(informative) == 9,
        f"9 informative entries (got {len(informative)})",
    )

    # Summary string should be non-empty
    summary = log.summary()
    all_passed &= check("10 total" in summary, "Summary mentions 10 total experiments")
    all_passed &= check("9 informative" in summary, "Summary mentions 9 informative")

    # The failing experiment should have informative=False
    failing = [e for e in entries if e.method == "should_fail"]
    all_passed &= check(len(failing) == 1, "Failing experiment was logged")
    if failing:
        all_passed &= check(
            not failing[0].informative,
            "Failing experiment marked not informative",
        )
        all_passed &= check(
            "FAILED" in failing[0].conclusion,
            "Failing experiment conclusion starts with FAILED",
        )

    # ========================================================================
    # Phase 5: Final summary
    # ========================================================================
    print()
    print("=" * 70)
    if all_passed:
        print("AutoInterp dry-run: ALL CHECKS PASSED")
    else:
        print("AutoInterp dry-run: SOME CHECKS FAILED")
    print("=" * 70)

    print(f"\nWork directory: {WORK_DIR.resolve()}")
    print(f"\nSummary from ExperimentLog:")
    print(log.summary())

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
