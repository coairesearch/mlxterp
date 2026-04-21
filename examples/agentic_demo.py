"""
Agentic interpretability end-to-end demo.

Demonstrates mlxterp's Tier 5 features on a real model:
1. Research workflows (behavior_localization, circuit_discovery)
2. AutoInterp ratchet loop (scaffold + programmatic runner)
3. Report generation (Markdown + HTML)

Usage: python examples/agentic_demo.py

Expected runtime: 2-5 minutes on Apple Silicon.
Expected output: Reports in ./demo_output/
"""

import os
import sys
import time
import warnings
from pathlib import Path

import mlx.core as mx

# Suppress expected warnings about attention weight computation
warnings.filterwarnings("ignore", message="Could not compute attention weights")

from mlxterp import InterpretableModel
from mlxterp.causal import activation_patching, direct_logit_attribution
from mlxterp.workflows import behavior_localization, circuit_discovery
from mlxterp.autointerpret import (
    init_autointerpret,
    AutoInterpret,
    ExperimentLog,
)
from mlxterp.reports import save_report, generate_report


MODEL_NAME = os.environ.get("MLXTERP_DEMO_MODEL", "EleutherAI/pythia-410m")
OUTPUT_DIR = Path(os.environ.get("MLXTERP_DEMO_OUTPUT", "./demo_output"))


def section(title):
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("Setup")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR.resolve()}")

    t0 = time.time()
    model = InterpretableModel(MODEL_NAME)
    print(f"Loaded in {time.time() - t0:.1f}s. Layers: {len(model.layers)}")

    clean = "The capital of France is"
    corrupted = "The capital of Germany is"
    print(f"Clean:     '{clean}'")
    print(f"Corrupted: '{corrupted}'")

    # ========================================================================
    # DEMO 1: Research Workflows
    # ========================================================================
    section("Demo 1: Research Workflows")

    print("\n[1a] behavior_localization (DLA + MLP patching + attn patching)...")
    t0 = time.time()
    bl_result = behavior_localization(
        model, clean, corrupted,
        steps=["dla", "patch_mlp", "patch_attn"],
        verbose=False,
    )
    print(f"  Completed in {time.time() - t0:.1f}s")
    print(f"  {bl_result.summary()}")
    print(f"  Narrative: {bl_result.narrative[:200]}...")

    mlp = bl_result.get_step("patch_mlp")
    print(f"  Top MLP layers: {[(l, f'{e:.3f}') for l, e in mlp.top_components(k=3)]}")

    print("\n[1b] circuit_discovery (attribution + activation + ACDC)...")
    t0 = time.time()
    cd_result = circuit_discovery(
        model, clean, corrupted,
        threshold=0.05,
        steps=["attribution", "acdc"],
        verbose=False,
    )
    print(f"  Completed in {time.time() - t0:.1f}s")
    acdc_res = cd_result.get_step("acdc")
    if acdc_res:
        print(f"  Circuit: {len(acdc_res.nodes)} nodes")
        print(f"  Top components: {acdc_res.nodes[:5]}")

    # ========================================================================
    # DEMO 2: AutoInterp Scaffold
    # ========================================================================
    section("Demo 2: AutoInterp Scaffold (Zero-Orchestration Mode)")

    scaffold_dir = OUTPUT_DIR / "autointerpret_scaffold"
    path = init_autointerpret(
        output_dir=str(scaffold_dir),
        model_name=MODEL_NAME,
        research_question="How does this model perform factual recall?",
        max_experiments=50,
    )
    print(f"Created scaffold at: {path}")
    print("Files:")
    for f in sorted(Path(path).iterdir()):
        print(f"  {f.name}" + ("/ (dir)" if f.is_dir() else ""))
    print()
    print("To use this in zero-orchestration mode:")
    print(f"  1. cd {path}")
    print("  2. claude  # Open Claude Code in this directory")
    print("  3. Say: 'Read program.md and start investigating.'")

    # ========================================================================
    # DEMO 3: AutoInterp Programmatic Runner
    # ========================================================================
    section("Demo 3: AutoInterp Programmatic Runner (5 experiments)")

    runner = AutoInterpret(
        model=model,
        clean=clean,
        corrupted=corrupted,
        output_dir=str(OUTPUT_DIR / "autointerpret_run"),
        max_experiments=10,
    )

    experiments = [
        ("dla_scan", lambda: direct_logit_attribution(model, clean),
         "Identify target token and top-contributing layers"),
        ("mlp_patch", lambda: activation_patching(model, clean, corrupted, component="mlp"),
         "Find important MLP layers"),
        ("attn_patch", lambda: activation_patching(model, clean, corrupted, component="attn"),
         "Find important attention layers"),
        ("mlp_kl", lambda: activation_patching(model, clean, corrupted, component="mlp", metric="kl"),
         "Verify MLP importance with KL divergence"),
        ("mlp_cosine", lambda: activation_patching(model, clean, corrupted, component="mlp", metric="cosine"),
         "Cross-validate with cosine metric"),
    ]

    for name, fn, hypothesis in experiments:
        t0 = time.time()
        entry = runner.run_experiment(name=name, fn=fn, hypothesis=hypothesis)
        print(f"  [{runner.n_experiments:2d}] {name}: {entry.conclusion[:60]}... ({time.time() - t0:.1f}s)")

    print(f"\n{runner.summary()}")

    # Check findings directory was populated
    findings_dir = Path(runner.output_dir) / "findings"
    if findings_dir.exists():
        findings = list(findings_dir.glob("*.json"))
        print(f"\nFindings saved: {len(findings)} files in {findings_dir}")

    # ========================================================================
    # DEMO 4: Report Generation
    # ========================================================================
    section("Demo 4: Report Generation")

    report_path = OUTPUT_DIR / "investigation_report.md"
    save_report(
        [bl_result, cd_result],
        str(report_path),
        title="Factual Recall Investigation",
        description=(
            "This report documents a causal investigation of factual recall in "
            f"{MODEL_NAME}. We used behavior_localization and circuit_discovery "
            "workflows to identify which components handle the France→Paris "
            "association."
        ),
    )
    print(f"Markdown report: {report_path}")

    html_path = OUTPUT_DIR / "investigation_report.html"
    save_report(
        [bl_result, cd_result],
        str(html_path),
        title="Factual Recall Investigation",
        description="HTML report with formatted output.",
        include_plots=False,  # Plots require matplotlib; skip for portability
    )
    print(f"HTML report: {html_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    section("Summary")
    print(f"All demos completed successfully.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print()
    print("Verification checklist:")
    print(f"  [{'x' if (OUTPUT_DIR / 'investigation_report.md').exists() else ' '}] Markdown report generated")
    print(f"  [{'x' if (OUTPUT_DIR / 'investigation_report.html').exists() else ' '}] HTML report generated")
    print(f"  [{'x' if (OUTPUT_DIR / 'autointerpret_scaffold' / 'program.md').exists() else ' '}] AutoInterp scaffold created")
    print(f"  [{'x' if runner.n_experiments == 5 else ' '}] 5 experiments logged ({runner.n_experiments})")
    print(f"  [{'x' if len(bl_result.steps) > 0 else ' '}] Behavior localization workflow ran")
    print(f"  [{'x' if len(cd_result.steps) > 0 else ' '}] Circuit discovery workflow ran")
    print()

    return {
        "output_dir": str(OUTPUT_DIR.resolve()),
        "n_experiments": runner.n_experiments,
        "bl_steps": len(bl_result.steps),
        "cd_steps": len(cd_result.steps),
    }


if __name__ == "__main__":
    results = main()
    print(f"\nDemo complete: {results}")
