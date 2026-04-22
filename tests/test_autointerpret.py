"""Tests for mlxterp.autointerpret module."""

import json
import os
import pytest
from pathlib import Path
import mlx.core as mx

from mlxterp.autointerpret import (
    AutoInterpret,
    MetricRegistry,
    init_autointerpret,
    ExperimentLog,
    ExperimentEntry,
)


class TestMetricRegistry:
    def test_register_and_get(self):
        reg = MetricRegistry()
        reg.register("test", lambda x: x, description="A test metric")
        assert "test" in reg
        assert reg.get("test")("hello") == "hello"

    def test_unknown_raises(self):
        reg = MetricRegistry()
        with pytest.raises(KeyError, match="Unknown metric"):
            reg.get("nonexistent")

    def test_list(self):
        reg = MetricRegistry()
        reg.register("a", lambda: None, description="Metric A")
        reg.register("b", lambda: None, description="Metric B")
        listing = reg.list()
        assert len(listing) == 2
        assert listing[0]["name"] == "a"

    def test_names(self):
        reg = MetricRegistry()
        reg.register("x", lambda: None)
        reg.register("y", lambda: None)
        assert reg.names == ["x", "y"]

    def test_len(self):
        reg = MetricRegistry()
        assert len(reg) == 0
        reg.register("a", lambda: None)
        assert len(reg) == 1

    def test_contains(self):
        reg = MetricRegistry()
        reg.register("exists", lambda: None)
        assert "exists" in reg
        assert "missing" not in reg


class TestExperimentEntry:
    def test_auto_id(self):
        entry = ExperimentEntry(hypothesis="test")
        assert entry.experiment_id.startswith("exp_")

    def test_auto_timestamp(self):
        entry = ExperimentEntry()
        assert entry.timestamp > 0

    def test_to_dict(self):
        entry = ExperimentEntry(
            hypothesis="H1",
            method="patching",
            result={"score": 0.5},
            informative=True,
            conclusion="It works",
        )
        d = entry.to_dict()
        assert d["hypothesis"] == "H1"
        assert d["method"] == "patching"
        assert d["result"]["score"] == 0.5

    def test_to_json(self):
        entry = ExperimentEntry(hypothesis="H1")
        j = entry.to_json()
        parsed = json.loads(j)
        assert parsed["hypothesis"] == "H1"


class TestExperimentLog:
    def test_append_and_read(self, tmp_path):
        log = ExperimentLog(str(tmp_path / "test.jsonl"))
        log.append(ExperimentEntry(hypothesis="H1", conclusion="C1", informative=True))
        log.append(ExperimentEntry(hypothesis="H2", conclusion="C2", informative=False))

        entries = log.entries()
        assert len(entries) == 2
        assert entries[0].hypothesis == "H1"
        assert entries[1].hypothesis == "H2"

    def test_informative_entries(self, tmp_path):
        log = ExperimentLog(str(tmp_path / "test.jsonl"))
        log.append(ExperimentEntry(informative=True, conclusion="good"))
        log.append(ExperimentEntry(informative=False, conclusion="bad"))
        log.append(ExperimentEntry(informative=True, conclusion="also good"))

        informative = log.informative_entries()
        assert len(informative) == 2

    def test_empty_log(self, tmp_path):
        log = ExperimentLog(str(tmp_path / "nonexistent.jsonl"))
        assert log.entries() == []
        assert log.count == 0

    def test_summary(self, tmp_path):
        log = ExperimentLog(str(tmp_path / "test.jsonl"))
        log.append(ExperimentEntry(
            informative=True, conclusion="Found important layer",
            duration_seconds=5.0,
        ))
        s = log.summary()
        assert "1 total" in s
        assert "1 informative" in s
        assert "Found important layer" in s

    def test_summary_empty(self, tmp_path):
        log = ExperimentLog(str(tmp_path / "empty.jsonl"))
        assert "No experiments" in log.summary()

    def test_count(self, tmp_path):
        log = ExperimentLog(str(tmp_path / "test.jsonl"))
        assert log.count == 0
        log.append(ExperimentEntry())
        assert log.count == 1


class TestInitAutoInterpret:
    def test_creates_structure(self, tmp_path):
        out_dir = str(tmp_path / "auto_test")
        result = init_autointerpret(
            output_dir=out_dir,
            model_name="test-model",
            research_question="What does this model know?",
        )

        assert Path(result).exists()
        assert (Path(result) / "setup.py").exists()
        assert (Path(result) / "program.md").exists()
        assert (Path(result) / "experiment.py").exists()
        assert (Path(result) / "CLAUDE.md").exists()
        assert (Path(result) / "results.jsonl").exists()
        assert (Path(result) / "findings").is_dir()

    def test_setup_contains_model(self, tmp_path):
        out_dir = str(tmp_path / "auto_test")
        init_autointerpret(output_dir=out_dir, model_name="my-model")

        content = (Path(out_dir) / "setup.py").read_text()
        assert "my-model" in content

    def test_program_contains_question(self, tmp_path):
        out_dir = str(tmp_path / "auto_test")
        init_autointerpret(
            output_dir=out_dir,
            research_question="How does negation work?",
        )

        content = (Path(out_dir) / "program.md").read_text()
        assert "How does negation work?" in content

    def test_claude_md_has_instructions(self, tmp_path):
        out_dir = str(tmp_path / "auto_test")
        init_autointerpret(output_dir=out_dir)

        content = (Path(out_dir) / "CLAUDE.md").read_text()
        assert "Three-File Contract" in content
        assert "setup.py" in content
        assert "experiment.py" in content


class TestAutoInterpretRunner:
    def test_basic_runner(self, tmp_path):
        import mlx.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 16)
                self.layers = [nn.Linear(16, 16)]
                self.lm_head = nn.Linear(16, 32, bias=False)

            def __call__(self, x):
                return self.lm_head(self.embed(x))

        from mlxterp import InterpretableModel
        m = TinyModel()
        mx.eval(m.parameters())
        model = InterpretableModel(m)

        runner = AutoInterpret(
            model=model,
            output_dir=str(tmp_path / "runner_test"),
            max_experiments=5,
        )

        assert runner.n_experiments == 0
        assert not runner.is_done

    def test_run_experiment(self, tmp_path):
        import mlx.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 16)
                self.layers = [nn.Linear(16, 16)]
                self.lm_head = nn.Linear(16, 32, bias=False)

            def __call__(self, x):
                return self.lm_head(self.embed(x))

        from mlxterp import InterpretableModel
        m = TinyModel()
        mx.eval(m.parameters())
        model = InterpretableModel(m)

        runner = AutoInterpret(
            model=model,
            output_dir=str(tmp_path / "runner_test"),
        )

        entry = runner.run_experiment(
            name="test_exp",
            fn=lambda: {"score": 0.42},
            hypothesis="Testing the runner",
        )

        assert entry.method == "test_exp"
        assert entry.informative is True
        assert runner.n_experiments == 1

    def test_run_failing_experiment(self, tmp_path):
        import mlx.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 16)
                self.layers = [nn.Linear(16, 16)]
                self.lm_head = nn.Linear(16, 32, bias=False)

            def __call__(self, x):
                return self.lm_head(self.embed(x))

        from mlxterp import InterpretableModel
        m = TinyModel()
        mx.eval(m.parameters())
        model = InterpretableModel(m)

        runner = AutoInterpret(
            model=model,
            output_dir=str(tmp_path / "runner_test"),
        )

        entry = runner.run_experiment(
            name="failing_exp",
            fn=lambda: (_ for _ in ()).throw(ValueError("test error")),
            hypothesis="This will fail",
        )

        assert "FAILED" in entry.conclusion
        assert entry.informative is False

    def test_summary(self, tmp_path):
        import mlx.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 16)
                self.layers = [nn.Linear(16, 16)]
                self.lm_head = nn.Linear(16, 32, bias=False)

            def __call__(self, x):
                return self.lm_head(self.embed(x))

        from mlxterp import InterpretableModel
        m = TinyModel()
        mx.eval(m.parameters())
        model = InterpretableModel(m)

        runner = AutoInterpret(
            model=model,
            output_dir=str(tmp_path / "runner_test"),
        )
        runner.run_experiment("e1", lambda: {"x": 1}, hypothesis="H1")

        s = runner.summary()
        assert "1 total" in s
