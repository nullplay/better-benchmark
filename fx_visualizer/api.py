"""Public entry points for generic FX graphs and Better Benchmark repros."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from .rendering import _visualize_scheduler, render_capture, visualize_matmul_comparison

if TYPE_CHECKING:
    from torch.fx import GraphModule


VisualizationMode = Literal["fx", "scheduler", "matmul_comparison"]


def visualize_graph(
    gm: GraphModule,
    example_inputs: Sequence[Any],
    output_dir: str | Path,
    *,
    mode: VisualizationMode = "fx",
    title: str | None = None,
    config: Mapping[str, Any] | None = None,
    schedule_only: bool = True,
    triton_matmul: bool = False,
    clustered: bool = True,
) -> dict[str, Any]:
    """Export a directed interactive graph with optional compiler boundaries.

    ``fx`` only reads the supplied graph and input metadata. It neither traces
    nor executes it and requires no GPU or Inductor capture. ``scheduler``
    records real lowering and fusion; its default stops before codegen and
    autotuning. ``matmul_comparison`` records template and native dot
    schedules over the same original graph.

    Shapes, strides, offsets, aliases and nested argument ports are retained.
    Intermediate/output geometry is shown when present in FX node metadata.
    Compiler captures require compatible PyTorch internals and a dedicated
    process. Full captures with ``schedule_only=False`` execute graph effects.
    The result contains paths to HTML, SVG, widget and structured JSON files.
    """
    if mode not in ("fx", "scheduler", "matmul_comparison"):
        raise ValueError(f"Unknown visualization mode {mode!r}")
    if mode == "fx" and (config or triton_matmul):
        raise ValueError("Compiler options require scheduler or matmul_comparison mode")
    if mode == "matmul_comparison":
        if triton_matmul:
            raise ValueError("matmul_comparison controls its template and native matmul lowering modes")
        return visualize_matmul_comparison(gm, example_inputs, output_dir, config=config, schedule_only=schedule_only, title=title)
    if mode == "scheduler":
        return _visualize_scheduler(
            gm, example_inputs, output_dir, config=config, triton_matmul=triton_matmul,
            schedule_only=schedule_only,
            title=title, clustered=clustered,
        )

    import torch

    from .graph import input_geometry, normalize_fx_graph

    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(f"Expected torch.fx.GraphModule, got {type(gm).__name__}")
    inputs = tuple(example_inputs)
    original, _ = normalize_fx_graph(gm, inputs)
    original["name"] = Path(output_dir).name
    geometries, aliases, storage_groups = input_geometry(inputs)
    capture = {
        "schema_version": 1, "name": original["name"], "original_graph": original,
        "compile": {
            "mode": "fx_only", "status": "success", "inputs": geometries,
            "input_aliases": aliases, "input_storage_groups": storage_groups,
            "execution_succeeded": False, "scheduling_performed": False,
            "metadata_capture": original["metadata_capture"],
        },
        "stages": {"lowering": {"scheduler_nodes": []}, "fusion": {"scheduler_nodes": []}},
    }
    return render_capture(capture, output_dir, title=title, clustered=False)


def visualize_repro(
    repro_path: str | Path,
    output_dir: str | Path,
    *,
    configuration: int | str | Mapping[str, Any] = 0,
    mode: VisualizationMode = "fx",
    title: str | None = None,
    config: Mapping[str, Any] | None = None,
    schedule_only: bool = True,
    triton_matmul: bool = False,
    clustered: bool = True,
) -> dict[str, Any]:
    """Visualize any BB ``repro.py`` with one saved input configuration.

    ``repro_path`` accepts a directory or its ``repro.py``. ``configuration``
    accepts a saved index, its key, or an explicit configuration dictionary.
    The BB loaders and local input adapter preserve storage and compile policies. The
    FX-only mode creates inputs under strict FakeTensorMode, without allocating
    CUDA tensors. It records literal-only ATen calls during symbolic tracing.
    Pure ATen/prims repros also receive strict fake intermediate/output metadata.
    Graphs containing effects or unrecognized Python/module calls retain input
    metadata with an explicit partial-metadata note. No matmul manifest or region
    metadata is required.
    """
    from oracle_harness import load_repro_module
    from repro_harness import compile_policy_from_config

    from .inputs import load_input_configs, make_repro_inputs
    from .repro import populate_metadata, trace_repro

    if mode not in ("fx", "scheduler", "matmul_comparison"):
        raise ValueError(f"Unknown visualization mode {mode!r}")
    if mode == "matmul_comparison" and (triton_matmul):
        raise ValueError("matmul_comparison controls its template and native matmul lowering modes")
    if mode == "fx":
        from torch._subclasses.fake_tensor import FakeTensorMode

        factory_context = FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False)
    else:
        factory_context = nullcontext()
    path = Path(repro_path).resolve()
    directory = path.parent if path.is_file() else path
    with factory_context:
        module = load_repro_module(directory)
    points = load_input_configs(directory / "repro.py")
    if isinstance(configuration, Mapping):
        point = dict(configuration)
    elif isinstance(configuration, str):
        if configuration not in points:
            raise ValueError(f"Unknown saved input configuration {configuration!r}")
        point = points[configuration]
    elif isinstance(configuration, int) and not isinstance(configuration, bool):
        if configuration < 0 or configuration >= max(len(points), 1):
            raise ValueError(f"Configuration index must be in 0..{max(len(points), 1) - 1}")
        point = list(points.values())[configuration] if points else None
    else:
        raise TypeError("configuration must be a saved index, key or dictionary")
    policy = dict(compile_policy_from_config(point).get("inductor", {}))
    for key, value in (config or {}).items():
        if key in policy and (type(policy[key]) is not type(value) or policy[key] != value):
            raise ValueError(f"Compiler option {key!r} conflicts with the saved configuration's compile_policy")
        policy[key] = value
    if mode == "matmul_comparison" and "triton.native_matmul" in policy:
        raise ValueError("matmul_comparison controls triton.native_matmul; remove the saved or explicit option")
    if mode == "fx" and (config or triton_matmul):
        raise ValueError("Compiler options require scheduler or matmul_comparison mode")

    with factory_context:
        inputs = make_repro_inputs(point, module.make_inputs, geometry_only=mode == "fx")
        gm = trace_repro(module.Repro())
        if mode == "fx":
            gm.meta["bb_metadata_capture"] = populate_metadata(gm, inputs, factory_context)
    return visualize_graph(
        gm, inputs, output_dir, mode=mode, title=title or directory.name,
        config=policy if mode != "fx" else None, schedule_only=schedule_only,
        triton_matmul=triton_matmul,
        clustered=clustered,
    )
