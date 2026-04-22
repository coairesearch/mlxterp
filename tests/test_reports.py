"""Tests for mlxterp.reports module."""

import pytest
from pathlib import Path
from mlxterp.results import PatchingResult, CircuitResult, GenerationResult
from mlxterp.reports import generate_report, save_report, _md_to_html, _html_escape
import mlx.core as mx


@pytest.fixture
def patching_result():
    return PatchingResult(
        data={"component": "mlp", "metric": "l2"},
        metadata={"clean": "test clean", "n_layers": 3},
        effect_matrix=mx.array([0.1, 0.5, 0.3]),
        layers=[0, 1, 2],
        component="mlp",
        metric_name="l2",
    )


@pytest.fixture
def circuit_result():
    return CircuitResult(
        data={"threshold": 0.01},
        nodes=["L0.attn", "L1.mlp", "L2.attn"],
        edges=[("L0.attn", "L1.mlp", 0.8), ("L1.mlp", "L2.attn", 0.6)],
        threshold=0.01,
    )


class TestGenerateReport:
    def test_markdown_report(self, patching_result):
        report = generate_report(patching_result, title="Test Report")
        assert "# Test Report" in report
        assert "mlxterp" in report
        assert "mlp" in report.lower()

    def test_markdown_multiple_results(self, patching_result, circuit_result):
        report = generate_report(
            [patching_result, circuit_result],
            title="Combined Report",
        )
        assert "Patching" in report
        assert "Circuit" in report
        assert "Combined Report" in report

    def test_markdown_with_description(self, patching_result):
        report = generate_report(
            patching_result,
            description="This is a test analysis.",
        )
        assert "This is a test analysis" in report

    def test_markdown_with_json(self, patching_result):
        report = generate_report(patching_result, include_json=True)
        assert "```json" in report

    def test_html_report(self, patching_result):
        report = generate_report(patching_result, format="html")
        assert "<html>" in report
        assert "<title>" in report
        assert "mlp" in report.lower()

    def test_html_has_styles(self, patching_result):
        report = generate_report(patching_result, format="html")
        assert "<style>" in report
        assert "font-family" in report

    def test_circuit_report(self, circuit_result):
        report = generate_report(circuit_result, title="Circuit Report")
        assert "3" in report  # 3 nodes
        assert "L0.attn" in report

    def test_generation_result(self):
        result = GenerationResult(
            data={}, text="Hello world", tokens=[1, 2], prompt="Hello",
        )
        report = generate_report(result)
        assert "Hello world" in report

    def test_top_components_table(self, patching_result):
        report = generate_report(patching_result, format="markdown")
        assert "Layer" in report
        assert "Effect" in report


class TestSaveReport:
    def test_save_markdown(self, tmp_path, patching_result):
        path = str(tmp_path / "report.md")
        result = save_report(patching_result, path, title="Saved Report")
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "Saved Report" in content

    def test_save_html(self, tmp_path, patching_result):
        path = str(tmp_path / "report.html")
        result = save_report(patching_result, path, title="HTML Report")
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "<html>" in content

    def test_auto_detect_format(self, tmp_path, patching_result):
        md_path = save_report(patching_result, str(tmp_path / "r.md"))
        html_path = save_report(patching_result, str(tmp_path / "r.html"))
        assert "# " in Path(md_path).read_text()
        assert "<html>" in Path(html_path).read_text()


class TestMdToHtml:
    def test_headers(self):
        html = _md_to_html("# Title\n## Subtitle")
        assert "<h1>Title</h1>" in html
        assert "<h2>Subtitle</h2>" in html

    def test_bold(self):
        html = _md_to_html("This is **bold** text")
        assert "<strong>bold</strong>" in html

    def test_code(self):
        html = _md_to_html("Use `my_function()` here")
        assert "<code>my_function()</code>" in html

    def test_list(self):
        html = _md_to_html("- item 1\n- item 2")
        assert "<li>item 1</li>" in html

    def test_code_block(self):
        html = _md_to_html("```\ncode here\n```")
        assert "<pre><code>" in html
        assert "code here" in html

    def test_hr(self):
        html = _md_to_html("---")
        assert "<hr>" in html


class TestHtmlEscape:
    def test_escapes(self):
        assert _html_escape("<b>") == "&lt;b&gt;"
        assert _html_escape("a & b") == "a &amp; b"
        assert _html_escape('"hi"') == "&quot;hi&quot;"
