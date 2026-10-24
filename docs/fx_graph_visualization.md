# FX visualizer

`fx_visualizer` exports an interactive directed FX DAG for any Better Benchmark
repro or `torch.fx.GraphModule`. Matmul extraction metadata is not required.
The default only reads graph/input metadata; compiler capture is optional.

## Python API

```python
from fx_visualizer import visualize_graph, visualize_repro

# An existing FX graph, including non-matmul graphs on CPU.
artifacts = visualize_graph(gm, example_inputs, "results/fx")
print(artifacts["html"])

# A canonical repro and one saved setting, selected by index or key.
artifacts = visualize_repro(
    "repros/canonical/my_pattern",
    "results/repro_fx",
    configuration=0,
)

# Actual lowering and fusion, stopping before codegen and autotuning.
artifacts = visualize_graph(
    gm, example_inputs, "results/scheduler", mode="scheduler",
)

# Template/native modes with local interactive buttons over the same original DAG.
artifacts = visualize_graph(
    gm, example_inputs, "results/matmul", mode="matmul_comparison",
)
```

`visualize_graph(gm, example_inputs, output_dir, ...)` takes a GraphModule and
its positional inputs, including nested tuples/lists and scalar/shape values.
`visualize_repro(repro_path, output_dir, ...)` accepts a directory or its
`repro.py`, and `configuration` accepts an index, saved key or explicit BB input
configuration dictionary. It uses the existing BB input and compile policy
loaders, with a package-local adapter for nested input structures and standalone
storage offsets. No matmul extraction modules or shared runtime changes are
required.

Both return a dictionary of absolute artifact paths and captured group counts.
Their keyword arguments are:

| Argument | Default | Meaning |
| --- | --- | --- |
| `mode` | `"fx"` | `"fx"`, `"scheduler"` or `"matmul_comparison"` |
| `title` | Graph/repro name | Display title |
| `config` | `None` | Inductor overrides for compiler modes |
| `schedule_only` | `True` | Stop after the actual scheduler fusion pass |
| `triton_matmul` | `False` | Force Triton GEMM templates in scheduler mode |
| `clustered` | `True` | Compact region layout for a single scheduler viewer |

FX-only mode does not compile or execute real tensors. Repro inputs are created
from saved geometry under strict FakeTensorMode, retaining saved CUDA device,
shape, stride, storage offset and aliases without sampling values or allocating
CUDA tensors. Real compiler modes reuse BB input generators and restore saved
offsets and nested containers locally. Literal-only ATen factories
are recorded during repro tracing instead of eagerly folded. Fresh pure
ATen/prims repros receive intermediate/output metadata through FakeTensorProp
with fallback kernels and the fake tensor cache disabled. Effects and arbitrary
Python/module calls skip that propagation and show an explicit partial-metadata
note. `visualize_graph` only reads metadata already present in the supplied graph.
Compiler options require a compiler mode. Saved precision policies are respected;
conflicting overrides are rejected before inputs are created.

## Command line

```bash
# Default: original FX only, no GPU allocation or compilation.
python -m fx_visualizer \
    --repro repros/canonical/my_pattern --configuration 0 \
    --output-dir results/fx

# Real lowering/fusion boundaries; stop before kernel codegen and autotuning.
python -m fx_visualizer \
    --repro repros/canonical/my_pattern --mode scheduler \
    --triton-matmul --output-dir results/scheduler

# Template / native dot comparison.
python -m fx_visualizer \
    --repro repros/canonical/my_pattern --mode matmul_comparison \
    --output-dir results/comparison
```

`--inductor-config 'KEY=JSON'` is repeatable. `--full-compile` requests final
codegen, compilation and one compiled execution. The repository script
`scripts/visualize_fx_graph.py` calls the same package CLI.

## Render existing reports

Saved reports can be rendered with Python and Graphviz's `dot`, without
importing PyTorch or compiling again:

```python
from fx_visualizer import render_capture, render_matmul_comparison, write_index

one = render_capture("capture.json", "results/one")
comparison = render_matmul_comparison(
    "template/capture.json", "native/capture.json", "results/comparison",
)
write_index([one, comparison], "results/gallery")
```

```bash
python -m fx_visualizer --capture capture_a.json --capture capture_b.json \
    --output-dir results/gallery

python -m fx_visualizer --mode matmul_comparison \
    --capture template/capture.json --capture native/capture.json \
    --output-dir results/comparison
```

A comparison requires identical original FX nodes, directed ports and input
geometry/aliases. All modes use the same original node positions, directed
edge curves and full viewBox. Buttons only switch recorded evidence locally.

## Boundaries, views and interaction

The original graph is preserved and laid out from left to right. Solid broad
rounded convex envelopes show actual lowering groups. Thicker dashed envelopes
show actual fused groups and contain their complete lowering envelopes. No
holes, obstacle cutouts or inward dents are added. Unrelated nodes can appear
inside an envelope; geometry never establishes membership.

Click a boundary or its label to highlight exactly its observed members and
fade other nodes. Selecting a fused boundary also highlights its constituent
lowering boundaries and labels. Click the same boundary again to restore all
visible groups. Layer toggles, group selection, zoom and pan remain available.
The two comparison modes keep independent selection and camera state.

Click a node to inspect its original FX name, arguments and recursive geometry
including shape, stride, dtype, device and storage offset. Expand **FX input and
output arguments** to inspect boundary inputs, nested argument values and the
original return tree/producer paths. Ports and repeated references are retained.

Computation membership follows actual final IR bindings and the transformed
FX argument DAG to materialized buffer boundaries. Indexing membership records
observed view IR and pass-through identities. Views such as `view`, `permute`
and `expand` can therefore contribute to consumers without having independent
computation scheduler nodes. A wrapper `reinterpret_tensor` creates metadata
over the same storage; it is not evidence of a GPU materialization.

Fusion membership uses exact pre-fusion leaves of the actual fused scheduler
object. Full historical origin sets remain available as evidence and are not
used as an exclusive execution partition. Internal split passes with incomplete
bindings are marked `partial`, retaining unresolved source IDs. No contour is
invented for an empty mapping. Native matmul classification uses live captured
LoopBody FX operations, reduction types and exact source roots. Generated Python,
Triton and IR source dumps are retained as artifacts and are never parsed for
membership or launch counts. Existing annotated reports retain their evidence.

## Matmul comparison

- **Template GEMM:** Triton GEMM template lowering and epilogue fusion.
- **native_matmul=True:** Inductor broadcast dot/reduction IR retaining `tl.dot`.
  Eligible `mm`, `addmm`, `bmm` and `baddbmm` use this path; size-one dimensions
  can retain their existing path. Scaled/grouped GEMMs are not decomposed.


Schedule-only capture stops after real scheduler fusion and its configured
post-fusion pass, before loop merging, deferred template finalization and final
kernel/wrapper codegen. It uses deterministic unbenchmarked algorithm selection,
disables fusion benchmarking and pipelined/distributed GEMM tuning, and guards
against unexpected precompilation/autotuning. Template candidate construction
can still render source during lowering. Counts describe this heuristic schedule
preview and do not establish runtime kernel launches.

Compiler collection requires compatible PyTorch internals and the graph's
backend. Scoped hooks and lowering overrides are restored on exit. Run capture
in a dedicated process when other threads might compile concurrently. Tensor
inputs and module attributes keep their storage; full capture executes graph
mutation/RNG effects under `torch.no_grad()`. Use disposable inputs for such
full captures. FX-only mode reads or safely propagates fake metadata and runs no
graph effects on real tensors.

## Outputs and verification

Every viewer writes standalone HTML, SVG, an inline widget, `region_payload.json`,
`source_capture.json` and `visualization.json`. Live compiler evidence is saved
under `compiler/`. Comparison exports also include `matmul_comparison.html`,
its widget and manifest plus each mode's individual artifacts. Generated results
belong in a local scratch/results directory and are not part of the corpus.
The browser uses inline scripts, theme-aware SVG and local gzip decoding;
no network scripts or data are required.

```bash
python -m pytest tests/test_fx_visualizer.py -q
```

Tests exercise rounded envelopes, exact fusion child highlighting, shared view
membership, original ports and nested arguments, comparison input compatibility,
fixed layouts, torch-free saved rendering, and a CUDA-shaped non-matmul repro
under FakeTensorMode with CUDA initialization forbidden.

## Package layout

- `api.py`: public generic graph/repro API and compile policy handling.
- `inputs.py`: local saved-input decoding and geometry/replay adaptation.
- `graph.py`, `repro.py`: structured original FX metadata and BB repro tracing.
- `capture.py`: scoped real compiler/LoopBody snapshots.
- `execution.py`: computation/indexing attribution from recorded live data.
- `regions.py`: directed layout and smooth broad boundary geometry.
- `rendering.py`: interactive HTML, SVG and saved-report exports.
- `cli.py`, `__main__.py`: one package command-line entry point.
