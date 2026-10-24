"""Interactive FX DAGs and optional observed Inductor scheduler boundaries.

Importing this package and rendering saved captures does not import PyTorch.
Live graph/repro APIs import it only when called.
"""

from .api import VisualizationMode, visualize_graph, visualize_repro
from .rendering import render_capture, render_matmul_comparison, visualize_matmul_comparison, write_index

__all__ = [
    "VisualizationMode", "visualize_graph", "visualize_repro", "render_capture",
    "render_matmul_comparison", "visualize_matmul_comparison", "write_index",
]
