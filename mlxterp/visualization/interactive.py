"""Interactive attention visualization for mlxterp.

This module provides CircuitsViz-style interactive visualizations that can be
displayed in Jupyter notebooks or saved as standalone HTML files.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np


def _get_interactive_template() -> str:
    """Return the HTML template for interactive attention visualization."""
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        .mlxterp-attention-viz {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }
        .mlxterp-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }
        .mlxterp-controls {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        .mlxterp-control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .mlxterp-control-label {
            font-size: 13px;
            color: #666;
            font-weight: 500;
        }
        .mlxterp-layer-slider {
            width: 200px;
            cursor: pointer;
        }
        .mlxterp-direction-toggle {
            display: flex;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        .mlxterp-direction-btn {
            padding: 6px 12px;
            border: none;
            background: transparent;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        .mlxterp-direction-btn.active {
            background: #6366f1;
            color: white;
        }
        .mlxterp-main-container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .mlxterp-heatmap-container {
            flex: 0 0 auto;
        }
        .mlxterp-heatmap {
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
        }
        .mlxterp-head-selector {
            flex: 1;
            min-width: 300px;
        }
        .mlxterp-head-selector-title {
            font-size: 13px;
            color: #666;
            margin-bottom: 10px;
        }
        .mlxterp-head-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
            gap: 8px;
        }
        .mlxterp-head-thumb {
            position: relative;
            cursor: pointer;
            border: 2px solid transparent;
            border-radius: 4px;
            transition: all 0.2s;
            background: white;
        }
        .mlxterp-head-thumb:hover {
            border-color: #6366f1;
            transform: scale(1.05);
        }
        .mlxterp-head-thumb.selected {
            border-color: #6366f1;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        }
        .mlxterp-head-thumb canvas {
            width: 100%;
            height: auto;
            display: block;
            border-radius: 2px;
        }
        .mlxterp-head-label {
            font-size: 10px;
            text-align: center;
            padding: 2px;
            color: #666;
        }
        .mlxterp-tokens-container {
            margin-top: 15px;
            padding: 10px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .mlxterp-tokens-title {
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
        }
        .mlxterp-tokens {
            display: flex;
            flex-wrap: wrap;
            gap: 2px;
            line-height: 1.8;
        }
        .mlxterp-token {
            padding: 2px 4px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 13px;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            transition: all 0.15s;
            border: 1px solid transparent;
        }
        .mlxterp-token:hover {
            border-color: #6366f1;
        }
        .mlxterp-token.selected {
            border-color: #6366f1;
            background: #eef2ff !important;
        }
        .mlxterp-token.highlighted {
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3);
        }
        .mlxterp-layer-flow {
            margin-top: 20px;
            padding: 15px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .mlxterp-layer-flow-title {
            font-size: 13px;
            color: #666;
            margin-bottom: 10px;
            font-weight: 500;
        }
        .mlxterp-layer-flow-grid {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding-bottom: 10px;
        }
        .mlxterp-layer-flow-item {
            flex: 0 0 auto;
            text-align: center;
            cursor: pointer;
        }
        .mlxterp-layer-flow-item:hover {
            opacity: 0.8;
        }
        .mlxterp-layer-flow-item canvas {
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .mlxterp-layer-flow-label {
            font-size: 11px;
            color: #888;
            margin-top: 4px;
        }
        .mlxterp-info-panel {
            margin-top: 15px;
            padding: 10px;
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 4px;
            font-size: 12px;
            color: #0369a1;
        }
        .mlxterp-colorbar {
            display: flex;
            align-items: center;
            gap: 5px;
            margin-top: 10px;
        }
        .mlxterp-colorbar-gradient {
            width: 150px;
            height: 12px;
            border-radius: 2px;
            background: linear-gradient(to right, rgb(247, 251, 255), rgb(47, 101, 205));
        }
        .mlxterp-colorbar-label {
            font-size: 11px;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="mlxterp-attention-viz" id="viz-__VIZ_ID__">
        <div class="mlxterp-title">__TITLE__</div>

        <div class="mlxterp-controls">
            <div class="mlxterp-control-group">
                <span class="mlxterp-control-label">Layer:</span>
                <input type="range" class="mlxterp-layer-slider" id="layer-slider-__VIZ_ID__"
                       min="0" max="__MAX_LAYER__" value="0">
                <span id="layer-display-__VIZ_ID__" class="mlxterp-control-label">Layer 0</span>
            </div>

            <div class="mlxterp-control-group">
                <span class="mlxterp-control-label">Direction:</span>
                <div class="mlxterp-direction-toggle">
                    <button class="mlxterp-direction-btn active" id="dest-btn-__VIZ_ID__"
                            onclick="mlxterp__VIZ_ID__.setDirection('destination')">
                        Source → Dest
                    </button>
                    <button class="mlxterp-direction-btn" id="src-btn-__VIZ_ID__"
                            onclick="mlxterp__VIZ_ID__.setDirection('source')">
                        Source ← Dest
                    </button>
                </div>
            </div>
        </div>

        <div class="mlxterp-main-container">
            <div class="mlxterp-heatmap-container">
                <canvas class="mlxterp-heatmap" id="heatmap-__VIZ_ID__" width="__HEATMAP_SIZE__" height="__HEATMAP_SIZE__"></canvas>
                <div class="mlxterp-colorbar">
                    <span class="mlxterp-colorbar-label">0</span>
                    <div class="mlxterp-colorbar-gradient"></div>
                    <span class="mlxterp-colorbar-label">1</span>
                </div>
            </div>

            <div class="mlxterp-head-selector">
                <div class="mlxterp-head-selector-title">Head selector (hover to preview, click to lock)</div>
                <div class="mlxterp-head-grid" id="head-grid-__VIZ_ID__"></div>
            </div>
        </div>

        <div class="mlxterp-tokens-container">
            <div class="mlxterp-tokens-title">Tokens (click to focus)</div>
            <div class="mlxterp-tokens" id="tokens-__VIZ_ID__"></div>
        </div>

        <div class="mlxterp-layer-flow" id="layer-flow-__VIZ_ID__" style="display: none;">
            <div class="mlxterp-layer-flow-title">Attention flow across layers for selected token</div>
            <div class="mlxterp-layer-flow-grid" id="layer-flow-grid-__VIZ_ID__"></div>
        </div>

        <div class="mlxterp-info-panel" id="info-panel-__VIZ_ID__">
            Hover over tokens or the heatmap to see attention weights. Click a token to see how attention flows across layers.
        </div>
    </div>

    <script>
    var mlxterp__VIZ_ID__ = (function() {
        var vizId = "__VIZ_ID__";
        var data = __DATA_JSON__;
        var tokens = __TOKENS_JSON__;
        var numLayers = __NUM_LAYERS__;
        var numHeads = __NUM_HEADS__;
        var seqLen = tokens.length;

        var currentLayer = 0;
        var currentHead = 0;
        var selectedToken = null;
        var direction = 'destination';
        var lockedHead = null;

        function getColor(value) {
            var r = Math.round(247 - value * 200);
            var g = Math.round(251 - value * 150);
            var b = Math.round(255 - value * 50);
            return "rgb(" + r + ", " + g + ", " + b + ")";
        }

        function getColorRGBA(value, alpha) {
            var r = Math.round(66 + value * 30);
            var g = Math.round(102 + value * 30);
            var b = 241;
            return "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
        }

        function drawHeatmap() {
            var canvas = document.getElementById("heatmap-" + vizId);
            var ctx = canvas.getContext("2d");
            var cellSize = canvas.width / seqLen;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            var attention = data[currentLayer][currentHead];

            for (var i = 0; i < seqLen; i++) {
                for (var j = 0; j < seqLen; j++) {
                    if (j > i) continue;
                    var value = attention[i][j];
                    ctx.fillStyle = getColor(value);
                    ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                }
            }

            ctx.strokeStyle = "#ddd";
            ctx.lineWidth = 0.5;
            for (var i = 0; i <= seqLen; i++) {
                ctx.beginPath();
                ctx.moveTo(i * cellSize, 0);
                ctx.lineTo(i * cellSize, canvas.height);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, i * cellSize);
                ctx.lineTo(canvas.width, i * cellSize);
                ctx.stroke();
            }

            if (selectedToken !== null) {
                ctx.strokeStyle = "#6366f1";
                ctx.lineWidth = 2;
                if (direction === "destination") {
                    ctx.strokeRect(0, selectedToken * cellSize, canvas.width, cellSize);
                } else {
                    ctx.strokeRect(selectedToken * cellSize, 0, cellSize, canvas.height);
                }
            }
        }

        function drawHeadThumb(canvas, layerIdx, headIdx) {
            var ctx = canvas.getContext("2d");
            var size = canvas.width;
            var cellSize = size / seqLen;
            var attention = data[layerIdx][headIdx];

            for (var i = 0; i < seqLen; i++) {
                for (var j = 0; j < seqLen; j++) {
                    if (j > i) continue;
                    var value = attention[i][j];
                    ctx.fillStyle = getColor(value);
                    ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                }
            }
        }

        function createHeadGrid() {
            var grid = document.getElementById("head-grid-" + vizId);
            grid.innerHTML = "";

            for (var h = 0; h < numHeads; h++) {
                var thumb = document.createElement("div");
                thumb.className = "mlxterp-head-thumb" + (h === currentHead ? " selected" : "");
                thumb.dataset.head = h;

                var canvas = document.createElement("canvas");
                canvas.width = 50;
                canvas.height = 50;
                drawHeadThumb(canvas, currentLayer, h);

                var label = document.createElement("div");
                label.className = "mlxterp-head-label";
                label.textContent = "Head " + h;

                thumb.appendChild(canvas);
                thumb.appendChild(label);

                (function(headIdx) {
                    thumb.addEventListener("mouseenter", function() {
                        if (lockedHead === null) {
                            currentHead = headIdx;
                            drawHeatmap();
                            updateTokenHighlights();
                        }
                    });

                    thumb.addEventListener("click", function() {
                        if (lockedHead === headIdx) {
                            lockedHead = null;
                        } else {
                            lockedHead = headIdx;
                            currentHead = headIdx;
                        }
                        updateHeadSelection();
                        drawHeatmap();
                        updateTokenHighlights();
                    });
                })(h);

                grid.appendChild(thumb);
            }
        }

        function updateHeadSelection() {
            var thumbs = document.querySelectorAll("#head-grid-" + vizId + " .mlxterp-head-thumb");
            thumbs.forEach(function(thumb) {
                var h = parseInt(thumb.dataset.head);
                if (h === currentHead) {
                    thumb.classList.add("selected");
                } else {
                    thumb.classList.remove("selected");
                }
            });
        }

        function createTokens() {
            var container = document.getElementById("tokens-" + vizId);
            container.innerHTML = "";

            tokens.forEach(function(token, idx) {
                var span = document.createElement("span");
                span.className = "mlxterp-token";
                span.textContent = token;
                span.dataset.idx = idx;

                span.addEventListener("click", function() {
                    if (selectedToken === idx) {
                        selectedToken = null;
                        document.getElementById("layer-flow-" + vizId).style.display = "none";
                    } else {
                        selectedToken = idx;
                        document.getElementById("layer-flow-" + vizId).style.display = "block";
                        updateLayerFlow();
                    }
                    updateTokenHighlights();
                    drawHeatmap();
                });

                span.addEventListener("mouseenter", function() {
                    updateInfoPanel(idx);
                });

                container.appendChild(span);
            });
        }

        function updateTokenHighlights() {
            var tokenSpans = document.querySelectorAll("#tokens-" + vizId + " .mlxterp-token");
            var attention = data[currentLayer][currentHead];

            tokenSpans.forEach(function(span, idx) {
                if (idx === selectedToken) {
                    span.classList.add("selected");
                } else {
                    span.classList.remove("selected");
                }

                var attnValue = 0;
                if (selectedToken !== null) {
                    if (direction === "destination") {
                        attnValue = idx <= selectedToken ? attention[selectedToken][idx] : 0;
                    } else {
                        attnValue = idx >= selectedToken ? attention[idx][selectedToken] : 0;
                    }
                }

                if (selectedToken !== null && idx !== selectedToken) {
                    span.style.background = getColorRGBA(attnValue, Math.max(0.1, attnValue));
                } else if (idx === selectedToken) {
                    span.style.background = "#eef2ff";
                } else {
                    span.style.background = "";
                }
            });
        }

        function updateLayerFlow() {
            if (selectedToken === null) return;

            var grid = document.getElementById("layer-flow-grid-" + vizId);
            grid.innerHTML = "";

            for (var l = 0; l < numLayers; l++) {
                var item = document.createElement("div");
                item.className = "mlxterp-layer-flow-item";

                var canvas = document.createElement("canvas");
                canvas.width = 80;
                canvas.height = 80;

                var ctx = canvas.getContext("2d");
                var cellSize = canvas.width / seqLen;
                var attention = data[l][currentHead];

                for (var i = 0; i < seqLen; i++) {
                    for (var j = 0; j < seqLen; j++) {
                        if (j > i) continue;
                        var value = attention[i][j];
                        ctx.fillStyle = getColor(value);
                        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                    }
                }

                ctx.strokeStyle = "#ef4444";
                ctx.lineWidth = 2;
                ctx.strokeRect(0, selectedToken * cellSize, canvas.width, cellSize);

                var label = document.createElement("div");
                label.className = "mlxterp-layer-flow-label";
                label.textContent = "Layer " + l;

                item.appendChild(canvas);
                item.appendChild(label);

                (function(layerIdx) {
                    item.addEventListener("click", function() {
                        currentLayer = layerIdx;
                        document.getElementById("layer-slider-" + vizId).value = layerIdx;
                        document.getElementById("layer-display-" + vizId).textContent = "Layer " + layerIdx;
                        createHeadGrid();
                        drawHeatmap();
                        updateTokenHighlights();
                    });
                })(l);

                grid.appendChild(item);
            }
        }

        function updateInfoPanel(tokenIdx) {
            var panel = document.getElementById("info-panel-" + vizId);
            var attention = data[currentLayer][currentHead];

            if (selectedToken !== null) {
                var info = "<strong>\\"" + tokens[selectedToken] + "\\"</strong> (pos " + selectedToken + ") at L" + currentLayer + "H" + currentHead + ":<br>";

                if (direction === "destination") {
                    info += "Attends to: ";
                    var attns = [];
                    for (var j = 0; j <= selectedToken; j++) {
                        if (attention[selectedToken][j] > 0.05) {
                            attns.push("\\"" + tokens[j] + "\\" (" + (attention[selectedToken][j] * 100).toFixed(1) + "%)");
                        }
                    }
                    info += attns.slice(0, 5).join(", ");
                    if (attns.length > 5) info += "...";
                } else {
                    info += "Attended by: ";
                    var attns = [];
                    for (var i = selectedToken; i < seqLen; i++) {
                        if (attention[i][selectedToken] > 0.05) {
                            attns.push("\\"" + tokens[i] + "\\" (" + (attention[i][selectedToken] * 100).toFixed(1) + "%)");
                        }
                    }
                    info += attns.slice(0, 5).join(", ");
                    if (attns.length > 5) info += "...";
                }

                panel.innerHTML = info;
            } else if (tokenIdx !== undefined) {
                panel.innerHTML = "Hover: <strong>\\"" + tokens[tokenIdx] + "\\"</strong> (pos " + tokenIdx + "). Click to focus and see attention flow.";
            } else {
                panel.innerHTML = "Hover over tokens or the heatmap to see attention weights. Click a token to see how attention flows across layers.";
            }
        }

        function setDirection(dir) {
            direction = dir;
            var destBtn = document.getElementById("dest-btn-" + vizId);
            var srcBtn = document.getElementById("src-btn-" + vizId);
            if (dir === "destination") {
                destBtn.classList.add("active");
                srcBtn.classList.remove("active");
            } else {
                destBtn.classList.remove("active");
                srcBtn.classList.add("active");
            }
            drawHeatmap();
            updateTokenHighlights();
            updateInfoPanel();
        }

        document.getElementById("layer-slider-" + vizId).addEventListener("input", function(e) {
            currentLayer = parseInt(e.target.value);
            document.getElementById("layer-display-" + vizId).textContent = "Layer " + currentLayer;
            createHeadGrid();
            drawHeatmap();
            updateTokenHighlights();
            if (selectedToken !== null) updateLayerFlow();
        });

        var heatmapCanvas = document.getElementById("heatmap-" + vizId);
        heatmapCanvas.addEventListener("mousemove", function(e) {
            var rect = heatmapCanvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
            var cellSize = heatmapCanvas.width / seqLen;
            var col = Math.floor(x / cellSize);
            var row = Math.floor(y / cellSize);

            if (row >= 0 && row < seqLen && col >= 0 && col <= row) {
                var attention = data[currentLayer][currentHead];
                var value = attention[row][col];
                var panel = document.getElementById("info-panel-" + vizId);
                panel.innerHTML = "<strong>\\"" + tokens[row] + "\\"</strong> → <strong>\\"" + tokens[col] + "\\"</strong>: " + (value * 100).toFixed(1) + "% attention (L" + currentLayer + "H" + currentHead + ")";
            }
        });

        createHeadGrid();
        createTokens();
        drawHeatmap();

        return {
            setDirection: setDirection
        };
    })();
    </script>
</body>
</html>'''


@dataclass
class InteractiveAttentionConfig:
    """Configuration for interactive attention visualization."""

    title: str = "Attention Patterns"
    heatmap_size: int = 350
    colorscale: str = "Blues"
    show_layer_flow: bool = True


def interactive_attention(
    attention_patterns: Dict[int, np.ndarray],
    tokens: List[str],
    config: Optional[InteractiveAttentionConfig] = None,
) -> str:
    """Create an interactive attention visualization.

    This creates a CircuitsViz-style interactive visualization with:
    - Token bar with click-to-focus highlighting
    - Head selector with hover preview and click-to-lock
    - Main attention heatmap
    - Source ↔ Destination toggle
    - Layer slider
    - Cross-layer attention flow view

    Args:
        attention_patterns: Dict mapping layer index to attention tensors.
            Each tensor has shape (batch, num_heads, seq_len, seq_len).
        tokens: List of token strings.
        config: Visualization configuration.

    Returns:
        HTML string that can be displayed in Jupyter or saved to file.

    Example:
        >>> from mlxterp import InterpretableModel
        >>> from mlxterp.visualization import get_attention_patterns, interactive_attention
        >>>
        >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
        >>> with model.trace("The cat sat on the mat") as trace:
        ...     pass
        >>>
        >>> patterns = get_attention_patterns(trace)
        >>> tokens = model.to_str_tokens("The cat sat on the mat")
        >>>
        >>> html = interactive_attention(patterns, tokens)
        >>> # Display in Jupyter
        >>> from IPython.display import HTML, display
        >>> display(HTML(html))
    """
    if config is None:
        config = InteractiveAttentionConfig()

    # Convert attention patterns to nested list for JSON
    layers = sorted(attention_patterns.keys())
    num_layers = len(layers)

    # Get first layer to determine num_heads
    first_layer = layers[0]
    num_heads = attention_patterns[first_layer].shape[1]

    # Build data structure: data[layer][head] = attention matrix
    data = {}
    for layer_idx in layers:
        attn = attention_patterns[layer_idx]
        # Take first batch
        attn = np.array(attn[0])  # (num_heads, seq_len, seq_len)
        data[layer_idx] = {}
        for head_idx in range(num_heads):
            # Convert to list for JSON
            data[layer_idx][head_idx] = attn[head_idx].tolist()

    # Reindex layers to 0-based for JavaScript
    reindexed_data = {}
    for i, layer_idx in enumerate(layers):
        reindexed_data[i] = data[layer_idx]

    # Generate unique ID for this visualization
    import random
    viz_id = f"v{random.randint(10000, 99999)}"

    # Get template and replace placeholders
    html = _get_interactive_template()

    # Replace all placeholders
    html = html.replace("__VIZ_ID__", viz_id)
    html = html.replace("__TITLE__", config.title)
    html = html.replace("__MAX_LAYER__", str(num_layers - 1))
    html = html.replace("__HEATMAP_SIZE__", str(config.heatmap_size))
    html = html.replace("__DATA_JSON__", json.dumps(reindexed_data))
    html = html.replace("__TOKENS_JSON__", json.dumps(tokens))
    html = html.replace("__NUM_LAYERS__", str(num_layers))
    html = html.replace("__NUM_HEADS__", str(num_heads))

    return html


def display_interactive_attention(
    attention_patterns: Dict[int, np.ndarray],
    tokens: List[str],
    config: Optional[InteractiveAttentionConfig] = None,
) -> None:
    """Display interactive attention visualization in Jupyter notebook.

    Args:
        attention_patterns: Dict mapping layer index to attention tensors.
        tokens: List of token strings.
        config: Visualization configuration.
    """
    try:
        from IPython.display import HTML, display
    except ImportError:
        raise ImportError("IPython is required for display. Use interactive_attention() to get HTML string.")

    html = interactive_attention(attention_patterns, tokens, config)
    display(HTML(html))


def save_interactive_attention(
    attention_patterns: Dict[int, np.ndarray],
    tokens: List[str],
    filepath: Union[str, Path],
    config: Optional[InteractiveAttentionConfig] = None,
) -> None:
    """Save interactive attention visualization to HTML file.

    Args:
        attention_patterns: Dict mapping layer index to attention tensors.
        tokens: List of token strings.
        filepath: Path to save HTML file.
        config: Visualization configuration.
    """
    html = interactive_attention(attention_patterns, tokens, config)

    filepath = Path(filepath)
    filepath.write_text(html)
    print(f"Saved interactive visualization to {filepath}")


def interactive_attention_from_trace(
    trace: Any,
    tokens: List[str],
    layers: Optional[List[int]] = None,
    config: Optional[InteractiveAttentionConfig] = None,
) -> str:
    """Create interactive attention visualization directly from a trace.

    Args:
        trace: Trace context from model.trace().
        tokens: List of token strings.
        layers: Optional list of layer indices to include.
        config: Visualization configuration.

    Returns:
        HTML string for the interactive visualization.
    """
    from .attention import get_attention_patterns

    patterns = get_attention_patterns(trace, layers=layers)
    return interactive_attention(patterns, tokens, config)
