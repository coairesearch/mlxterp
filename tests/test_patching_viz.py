"""Tests for mlxterp.visualization.patching module."""

import pytest
import mlx.core as mx
from unittest.mock import patch, MagicMock
from mlxterp.results import PatchingResult
from mlxterp.visualization.patching import plot_patching_result, plot_patching_comparison


@pytest.fixture
def patching_result_1d():
    """1D patching result (layer-level)."""
    return PatchingResult(
        data={},
        effect_matrix=mx.array([0.1, 0.5, 0.3, 0.8, 0.2]),
        layers=[0, 1, 2, 3, 4],
        component="mlp",
        metric_name="l2",
    )


@pytest.fixture
def patching_result_2d():
    """2D patching result (layer x head)."""
    return PatchingResult(
        data={},
        effect_matrix=mx.array([[0.1, 0.9, 0.2], [0.4, 0.3, 0.7]]),
        layers=[0, 1],
        component="attn_head",
        metric_name="logit_diff",
    )


class TestPlotPatchingResult:
    """Tests for plot_patching_result."""

    @patch("mlxterp.visualization.patching.plt")
    def test_1d_bar_chart(self, mock_plt, patching_result_1d):
        """1D result should produce a bar chart."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig = plot_patching_result(patching_result_1d)

        mock_plt.subplots.assert_called_once()
        mock_ax.bar.assert_called_once()
        mock_ax.set_xlabel.assert_called()
        mock_ax.set_ylabel.assert_called()
        assert fig is mock_fig

    @patch("mlxterp.visualization.patching.plt")
    def test_2d_heatmap(self, mock_plt, patching_result_2d):
        """2D result should produce a heatmap."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_ax.imshow.return_value = MagicMock()

        fig = plot_patching_result(patching_result_2d)

        mock_plt.subplots.assert_called_once()
        mock_ax.imshow.assert_called_once()
        assert fig is mock_fig

    @patch("mlxterp.visualization.patching.plt")
    def test_custom_title(self, mock_plt, patching_result_1d):
        """Custom title should be used."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_patching_result(patching_result_1d, title="My Custom Title")
        mock_ax.set_title.assert_called_with("My Custom Title")

    def test_none_effect_matrix_raises(self):
        """Should raise if effect_matrix is None."""
        result = PatchingResult(data={})
        with pytest.raises(ValueError, match="No effect_matrix"):
            plot_patching_result(result)


class TestPlotPatchingComparison:
    """Tests for plot_patching_comparison."""

    @patch("mlxterp.visualization.patching.plt")
    def test_comparison(self, mock_plt, patching_result_1d, patching_result_2d):
        """Should create side-by-side plots."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        fig = plot_patching_comparison([patching_result_1d, patching_result_2d])

        mock_plt.subplots.assert_called_once()
        assert fig is mock_fig

    @patch("mlxterp.visualization.patching.plt")
    def test_single_result(self, mock_plt, patching_result_1d):
        """Should handle single result."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig = plot_patching_comparison([patching_result_1d])
        assert fig is mock_fig
