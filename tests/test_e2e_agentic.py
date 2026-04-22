"""
End-to-end integration tests for the agentic interpretability pipeline.

These tests use a real model (pythia-410m by default) and exercise the
full Tier 5 stack: workflows, AutoInterp, and reports.

Marked @pytest.mark.slow to skip by default (pre-existing unit tests
cover the logic with mocks). Run explicitly with:

    pytest tests/test_e2e_agentic.py -m slow --no-cov -v

Override the model via env var:

    MLXTERP_TEST_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit pytest ...
"""

import json
import os
import warnings
from pathlib import Path

import mlx.core as mx
import pytest

warnings.filterwarnings("ignore", message="Could not compute attention weights")


MODEL_NAME = os.environ.get("MLXTERP_TEST_MODEL", "EleutherAI/pythia-410m")


@pytest.fixture(scope="module")
def real_model():
    """Load a real model once for the test module."""
    from mlxterp import InterpretableModel
    return InterpretableModel(MODEL_NAME)


@pytest.fixture
def probes():
    """Standard clean/corrupted probe pair."""
    return {
        "clean": "The capital of France is",
        "corrupted": "The capital of Germany is",
    }


# ---------------------------------------------------------------------------
# Research Workflows
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBehaviorLocalization:
    def test_runs_end_to_end(self, real_model, probes):
        from mlxterp.workflows import behavior_localization
        result = behavior_localization(
            real_model, probes["clean"], probes["corrupted"],
            steps=["dla", "patch_mlp", "patch_attn"],
            verbose=False,
        )
        assert result is not None
        assert len(result.steps) == 3
        assert result.narrative

    def test_mlp_finds_important_layers(self, real_model, probes):
        from mlxterp.workflows import behavior_localization
        result = behavior_localization(
            real_model, probes["clean"], probes["corrupted"],
            steps=["patch_mlp"],
            verbose=False,
        )
        mlp = result.get_step("patch_mlp")
        assert mlp is not None
        # Some layer should have a non-zero effect
        effects = mlp.effect_matrix.tolist()
        assert any(abs(e) > 0.001 for e in effects)


@pytest.mark.slow
class TestCircuitDiscovery:
    def test_runs_end_to_end(self, real_model, probes):
        from mlxterp.workflows import circuit_discovery
        result = circuit_discovery(
            real_model, probes["clean"], probes["corrupted"],
            threshold=0.05,
            steps=["attribution", "acdc"],
            verbose=False,
        )
        assert result is not None
        assert result.get_step("attribution") is not None
        assert result.get_step("acdc") is not None

    def test_acdc_returns_circuit(self, real_model, probes):
        from mlxterp.workflows import circuit_discovery
        result = circuit_discovery(
            real_model, probes["clean"], probes["corrupted"],
            threshold=0.001,  # Low threshold to ensure we keep some nodes
            steps=["acdc"],
            verbose=False,
        )
        circuit = result.get_step("acdc")
        assert isinstance(circuit.nodes, list)
        # Low threshold => should have at least some nodes
        assert len(circuit.nodes) > 0


# ---------------------------------------------------------------------------
# AutoInterp
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAutoInterpScaffold:
    def test_init_creates_all_files(self, tmp_path):
        from mlxterp.autointerpret import init_autointerpret
        path = init_autointerpret(
            output_dir=str(tmp_path / "scaffold"),
            model_name=MODEL_NAME,
            research_question="Test question",
        )
        p = Path(path)
        assert (p / "setup.py").exists()
        assert (p / "experiment.py").exists()
        assert (p / "program.md").exists()
        assert (p / "CLAUDE.md").exists()
        assert (p / "results.jsonl").exists()
        assert (p / "findings").is_dir()

    def test_setup_references_model(self, tmp_path):
        from mlxterp.autointerpret import init_autointerpret
        path = init_autointerpret(
            output_dir=str(tmp_path / "scaffold"),
            model_name=MODEL_NAME,
        )
        setup = (Path(path) / "setup.py").read_text()
        assert MODEL_NAME in setup


@pytest.mark.slow
class TestAutoInterpRunner:
    def test_runs_five_experiments(self, real_model, probes, tmp_path):
        from mlxterp.autointerpret import AutoInterpret, ExperimentLog
        from mlxterp.causal import activation_patching, direct_logit_attribution

        runner = AutoInterpret(
            model=real_model,
            clean=probes["clean"],
            corrupted=probes["corrupted"],
            output_dir=str(tmp_path / "runner"),
            max_experiments=10,
        )

        # Run 5 mixed experiments
        runner.run_experiment("dla", lambda: direct_logit_attribution(real_model, probes["clean"]),
                              hypothesis="DLA baseline")
        runner.run_experiment("mlp_l2",
                              lambda: activation_patching(real_model, probes["clean"], probes["corrupted"], component="mlp"),
                              hypothesis="MLP importance (l2)")
        runner.run_experiment("mlp_kl",
                              lambda: activation_patching(real_model, probes["clean"], probes["corrupted"], component="mlp", metric="kl"),
                              hypothesis="MLP importance (kl)")
        runner.run_experiment("attn",
                              lambda: activation_patching(real_model, probes["clean"], probes["corrupted"], component="attn"),
                              hypothesis="Attention importance")
        runner.run_experiment("failing",
                              lambda: (_ for _ in ()).throw(RuntimeError("intentional")),
                              hypothesis="Should fail gracefully")

        assert runner.n_experiments == 5

        # Log has 5 entries
        log = ExperimentLog(str(Path(runner.output_dir) / "results.jsonl"))
        entries = log.entries()
        assert len(entries) == 5

        # 4 informative, 1 failing
        informative = log.informative_entries()
        assert len(informative) == 4

        # Findings dir has 4 files
        findings = list((Path(runner.output_dir) / "findings").glob("*.json"))
        assert len(findings) == 4


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestReportGeneration:
    def test_markdown_report_from_real_result(self, real_model, probes, tmp_path):
        from mlxterp.causal import activation_patching
        from mlxterp.reports import save_report

        result = activation_patching(
            real_model, probes["clean"], probes["corrupted"],
            component="mlp",
        )

        path = tmp_path / "report.md"
        save_report(result, str(path), title="E2E Report")
        assert path.exists()
        content = path.read_text()
        assert "E2E Report" in content
        assert "mlp" in content.lower()

    def test_html_report_from_workflow(self, real_model, probes, tmp_path):
        from mlxterp.workflows import behavior_localization
        from mlxterp.reports import save_report

        result = behavior_localization(
            real_model, probes["clean"], probes["corrupted"],
            steps=["patch_mlp"],
            verbose=False,
        )

        path = tmp_path / "report.html"
        save_report(
            result, str(path),
            title="Workflow Report",
            include_plots=False,
        )
        assert path.exists()
        content = path.read_text()
        assert "<html>" in content
        assert "Workflow Report" in content


# ---------------------------------------------------------------------------
# Full pipeline (the "Claude Code self-test" in automated form)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFullAgenticPipeline:
    """
    The complete agentic workflow as a single test:
    1. Load real model
    2. Run workflow to localize behavior
    3. Create AutoInterp scaffold
    4. Run programmatic experiments
    5. Generate report
    """

    def test_full_pipeline(self, real_model, probes, tmp_path):
        from mlxterp.workflows import behavior_localization
        from mlxterp.autointerpret import init_autointerpret, AutoInterpret
        from mlxterp.causal import activation_patching
        from mlxterp.reports import save_report

        # Step 1: Workflow
        wf = behavior_localization(
            real_model, probes["clean"], probes["corrupted"],
            steps=["patch_mlp"],
            verbose=False,
        )
        assert len(wf.steps) == 1

        # Step 2: Scaffold
        scaffold_path = init_autointerpret(
            output_dir=str(tmp_path / "pipeline_scaffold"),
            model_name=MODEL_NAME,
            research_question="Full pipeline test",
        )
        assert Path(scaffold_path).exists()

        # Step 3: Runner
        runner = AutoInterpret(
            model=real_model,
            clean=probes["clean"],
            corrupted=probes["corrupted"],
            output_dir=str(tmp_path / "pipeline_run"),
        )
        runner.run_experiment(
            "mlp_l2",
            lambda: activation_patching(
                real_model, probes["clean"], probes["corrupted"], component="mlp",
            ),
            hypothesis="Full pipeline test",
        )
        assert runner.n_experiments == 1

        # Step 4: Report
        report_path = tmp_path / "pipeline_report.md"
        save_report(wf, str(report_path), title="Pipeline Test Report")
        assert report_path.exists()
        assert report_path.stat().st_size > 100  # Non-trivial content
