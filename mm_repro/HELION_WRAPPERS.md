# First-shape Helion wrappers

Every `canonical/<repro>/helion/standalone.py` corresponds to the first point in
that repro's `shapes.json`. Native Triton MM definitions and launches are replaced
by the Helion sources emitted by `pytorch-nullplay`. Other Triton kernels remain.

`call(args)` uses the Inductor boxed interface and consumes the list.
`make_inputs()` / `get_args()` supplies the first-shape tensor arguments in the
compiled order, retaining source generators, strides and alias groups. Shape-list
arguments are specialized during compilation.

Generation stops immediately after Inductor source codegen, before loading the
compiled module. FakeTensor fallback, compile-time autotuning and fusion
benchmarking are disabled. No generated wrapper or GPU kernel was executed.
Helion device compilation, autotuning and numeric validation are deferred.

## Results

- Source wrappers generated: **341 / 373**.
- Helion preview unsupported: **25**.
- Native-MM fallback to external GEMM: **7**.
- Two host-wrapper tensor references were retained because Helion needs the base
  view after Inductor's original `del`; no new tensor allocation is introduced.
- `pytorch-nullplay` now supports `index_expr`, masked loads, `abs`, `log`,
  `logical_not`, `bitwise_and`, `and_` and `logical_and` in Helion source codegen.
- The original **19 / 19** `index_expr` failures and the additional `abs` and
  `logical_not` cases now generate wrappers successfully.
- All **341** successful wrapper sources passed syntax, global-name and native
  dot replacement checks. The targeted epilogue source-codegen tests passed
  (**26 tests**). Generated wrappers and GPU kernels were not executed.

| Failure reason | Repros |
|---|---:|
| Unsupported `load_seed` | 6 |
| Unsupported `check_bounds` | 2 |
| Cannot reconstruct tensor indices | 11 |
| Unsupported mutation mode; preview supports `atomic_add` | 6 |
| External GEMM fallback: addmm / mm / bmm | 2 / 2 / 3 |

Failures have an explicit error marker in `standalone.py` and a full traceback in
`helion/codegen_status.json`. They must be excluded from benchmarking. Successful
source generation does not certify Helion runtime compilation or numerical
correctness.

## Reproduce

```bash
python -B scripts/generate_helion_wrappers.py \
  --torch-root /home/jaeyeonw/pytorch-nullplay --workers 4
```

The generator uses the activated venv and prioritizes its `nvidia/cu13/lib`,
`cudnn/lib`, `nccl/lib` and `cusparselt/lib` before system CUDA libraries. Worker
processes import Torch from the specified source tree; the installed wheel lacks
the `trace.helion_code` option. These settings apply only to the generator process.
Use the same source-tree Torch and CUDA library environment when benchmarking.

[Full per-repro report](helion_codegen_report.json).
[Requested-operation retry report](helion_ops_retry_report.json).

Masked bodies pass their mask to `hl.load(..., extra_mask=...)`, then select the
fallback value with `torch.where` and `torch.full_like` to preserve its dtype.
Nested masks combine with `&`. Masked stores remain unsupported.
