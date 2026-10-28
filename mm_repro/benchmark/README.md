# H100 first-shape matmul benchmark

The sweep attempts all 373 canonical repros with five backends:

- `aten`: `max_autotune=True`, GEMM backends `ATEN`.
- `triton`: `max_autotune=True`, GEMM backends `TRITON`.
- `nvgemm`: `max_autotune=True`, GEMM backends `NVGEMM`.
- `native_triton`: `triton.native_matmul=True`, Helion codegen off.
- `native_helion`: the previously exported Helion/Triton standalone wrapper;
  Helion's default `full` autotuning effort.

## Fairness and measurement

All modes use the same seed and the **first** `shapes.json` point, preserving
dtype, strides, offsets and alias groups. All five measurements for a repro run
on the same H100; different repros are distributed across four GPUs. There is
one active benchmark worker per GPU, including autotuning.

The benchmark input helper pads private storages when a captured view has a
nonzero offset, preserving its generated values and strides. The two cases that
initially failed this geometry check are retried on their original GPUs; their
input-generation failures are retained under `input_geometry_failure/`.

CUDA Graph capture is explicitly blocked. Inductor and Helion graph-based
autotune timers are also disabled. FP32 uses IEEE/highest precision with TF32
disabled. FP16/BF16 reduced-precision accumulation is disabled for every mode.
Unsupported TRITON/NVGEMM choices fail rather than silently using ATen.
Native matmul may retain compiler fallbacks for operations it cannot lower.

Primary latency is the median of **100 individual CUDA-event samples** in five
rotated rounds. Each interval begins after a **64 MiB L2 flush** and restoration
of mutated inputs, both excluded from event timing. Events include dispatch gaps
between kernels. A separate warm-cache wall measurement includes Python calls,
final synchronization, and mutation restoration. Profiler timings and the
per-kernel launch list are collected separately; profiler overhead does not
affect primary timings. Memcpy/memset activities are excluded from kernel counts.
`generated_kernel_count` counts selected generated program definitions, excluding
autotune candidates and prebuilt vendor GEMMs. `kernel_launch_count` and
`distinct_kernel_count` describe the actual profiled invocation, including vendor
GEMM launches. The separate Inductor metric includes generated autotune candidates.

Compile/tune seconds include source compilation, autotuning and the first
execution, excluding framework import, input generation and reference execution.
Every case/backend has a separate initially empty Inductor/Triton cache. Vendor
initialization and driver state are shared within a case. CPU compilation runs
concurrently across the four workers. Native Helion time includes wrapper import
and full tuning, but excludes the already completed FX-to-wrapper export.
Each backend has a 900-second watchdog; timeouts are recorded as failures.
If a later backend interrupts a case after another backend compiled successfully,
resuming preserves that backend's original cold compile time and records recovery
time separately. Saved Helion configurations prevent full retuning during recovery.

The source Torch reports `torch.version.cuda=None` despite its CUDA-enabled
binary. Each benchmark process derives the value from
`torch._C._cuda_getCompiledVersion()` before using Helion's tuning cache. This
changes process metadata only, identically for all five modes; source files
are not changed. Original and effective values are recorded in `case.json`.

## Accuracy

The reference promotes floating input storages while preserving their strides,
offsets and aliases, and promotes floating dtype arguments in the
eager graph to FP64. A failure falls back to a promoted FP32 eager graph, with
the FP64 error recorded. Every ordered tensor leaf is compared, allowing the
boxed Inductor tuple to wrap an original eager tensor. Shapes must agree;
integer/bool outputs must match exactly; nonfinite positions must match.

Floating error must satisfy `RMSE <= atol + rtol * reference_RMS`. Tolerances
follow the weakest input/output floating precision: BF16 `(0.02, 1e-5)`, FP16
`(0.002, 1e-6)`, FP32 `(0.0002, 1e-6)`. A BF16 result cast to FP32 retains the
BF16 threshold. Max absolute error, NRMSE and finite/nonfinite counts are saved.
Large mask sentinels (`abs(reference) >= finfo(output_dtype).max / 16`) are
excluded from the RMS scale and checked separately, so they cannot hide errors
in ordinary values. Every finite element must also satisfy
`abs(error) <= 2 * (atol + rtol * (abs(reference) + ordinary_reference_RMS))`.
The extra elementwise allowance accommodates BF16 rounding in the original
eager program while the RMS criterion remains unchanged. The allowance is
identical for all backends. Outputs whose reference has no finite values are
marked `reference_nonfinite`, rather than attributed to a compiler failure.
Stochastic graphs are explicitly unverified and excluded from valid comparisons.

## Files and reproduction

- `manifest.json`: versions, protocol, seeds and worker settings.
- `results.json`, `results.csv`, `summary.json`, `report.html`: report data.
- `results/<repro>/case.json`: exact first shape, seed, reference and GPU metadata.
- `results/<repro>/<backend>/result.json`: times, accuracy, kernel counts/errors.
- `winning_configs.json`: GEMM candidates/winner and chosen Triton/Helion configs.
- `compiled.py`: original generated code for that backend.
- `standalone_tuned.py`: Helion wrapper with its winning configurations pinned.
- `kernel_trace.json`: one separately profiled invocation and per-kernel times.
- `cache/`: reusable Inductor, Triton and Helion caches. Retain these with results.
- `compile.log`, `error.txt`, debug files: full tuning/compilation diagnostics.

Run or resume the sweep:

```bash
/home/jaeyeonw/pytorch-nullplay/venv/bin/python -B \
  scripts/benchmark_mm_backends.py --gpus 0,1,2,3 --resume
python scripts/render_mm_backend_report.py
```

Replay saved compiled code with winner settings and no retuning:

```bash
python scripts/replay_mm_backend.py \
  --case <canonical-id> --backend native_helion --gpu 0
```

The replay helper selects the same first shape and seed and uses pinned Helion
configs plus saved Triton winners/caches. It reports the same cold-L2 event
latency as the main report; `--warm-cache` requests a separately labeled
warm-cache measurement. Compare only
accuracy-valid results and use the same GPU for absolute timing comparisons.
