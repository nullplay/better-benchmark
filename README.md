# better-benchmark

Standalone kernel microbenchmark infrastructure for `torch.compile` / TorchInductor.

Extracts fused kernel regions from real model compilations as lightweight, self-contained repros — then benchmarks them against memory-bandwidth SOL with proper GPU isolation and lock-safe parallel execution.

## What This Does

1. **Kernel extraction** — captures post-grad FX graphs, partitions fusible nodes into subgraphs, and emits one standalone `torch.nn.Module` per kernel region. Saves both full post-grad graphs and per-kernel repros, so you can repartition without re-running models.

2. **Deduplication** — content-addressed hashing (ops + wiring, ignoring shapes). Same kernel pattern from different models → one canonical repro, multiple shape configs.

3. **Multi-shape benchmarking** — each repro supports multiple input shapes from different models via automatic shape parametrization. Strides, index bounds, and memory formats preserved.

4. **Parallel benchmark runner** — persistent workers per GPU, overlapped repro compilation, exclusive lock for timing isolation, CUDAGraph replay, worker recovery on CUDA errors.

5. **Reference kernels** — canonical patterns carry reference implementations for comparison with `torch.compile` output. Existing BB kernels use optimized oracles; the new matmul regions start with marked PyTorch eager placeholders.

6. **Model-level perf estimation** — per-model kernel aggregation gives an approximate accounting of how much improving an individual kernel translates to end-to-end speedup.

## Corpus

**2283 canonical patterns and 5646 input configurations**: the original BB set contains 1727 patterns / 4977 configurations, and the matmul collection adds 556 filtered 8-hop structures / 669 configurations. **364 full model graphs** are retained across torchbench, HuggingFace, timm and genai.

The original BB subset was measured in ~36 min across ~5K points on 4x B200. The expanded matmul set has not been timed as a full sweep.

<details>
<summary>Coverage breakdown</summary>

| Suite | Captures | Mode |
|-------|-------:|------|
| torchbench | 50 | infer + train |
| hf | 70 | infer + train |
| timm | 35 | infer + train |
| genai | 8 | microbenchmarks (fwd + bwd) |

Each canonical repro is a flat directory:

```
repros/canonical/mean_014afd4984e6/
  repro.py       # standalone nn.Module + make_inputs()
  meta.json      # pattern hash, ops, reduction types, model list
  shapes.json    # per-model shape configs
  oracle.py      # reference implementation / optimization target (see Oracles below)
```

Per-model directories hold full post-grad graphs for recapture and a manifest listing which canonical patterns that model produces.

</details>

## Quickstart

```bash
# Benchmark one repro
python repros/canonical/mean_014afd4984e6/repro.py

# Parallel sweep of entire corpus
python scripts/bench_parallel.py repros/canonical --gpus 0,1 --max-workers 2 \
  --output results.json

# Parallel sweep of the oracle references (per shape point)
python scripts/bench_parallel.py repros/canonical --oracles --gpus 0,1 \
  --max-workers 2 --all-shapes --output results/all_oracle_timings.json

# Parallel sweep of saved model full graphs
# (latency + input constraints; legacy annotation-only graphs are skipped)
python scripts/bench_parallel.py --full-graphs --gpus 0,1 \
  --workers-per-gpu 2 \
  --inductor-config combo_kernels=True \
  --inductor-config combo_kernel_per_subkernel_blocks=True \
  --output full_graph_results.json

# Validate all repros run correctly in eager mode
python scripts/validate_eager.py --gpus 0,1 --max-workers 4
```

## Oracles

Each canonical repro carries an **oracle** — a reference implementation
serving as a comparison for `torch.compile` output. Optimized oracles use
Triton, CUDA, CUTLASS or another compiler's output. New matmul oracles
use PyTorch eager and are explicitly marked as placeholders in metadata.

Oracles must match the repro's full scope and numerics (no fast-math
substitutions), and are timed exclusively through
`oracle_harness.bench_oracle()`.

```bash
python repros/canonical/mean_014afd4984e6/oracle.py --bench
```

Each oracle declares its operating point (hardware + input signature) so the
bench dispatches the right variant per shape and reports the match honestly.
To write a new oracle, start from `scripts/oracle_template.py`; for the
registration/dispatch model see `scripts/oracle_dispatch_design.md`.

</details>

## Full model artifacts

The original full FX graphs are retained per model, so they can be re-partitioned
under different fusion rules or run and benchmarked directly end-to-end.

## Model accounting

Per-kernel timings aggregate to the **model level**: partition a model's full post-grad graph, map each fusible partition back to its canonical pattern, and sum the per-kernel times weighted by occurrence — giving an approximate accounting of how much improving an individual kernel would speed up the whole model. Works with oracle times, SOL, or the compile path.

The 8-hop matmul regions overlap. Their provenance is in
`repros/matmul_manifest.json`; they are excluded from disjoint model partition
manifests and model-latency sums.

```bash
# Aggregate per-kernel times to a model-level estimate
python scripts/model_graph_accounting.py --model BertForMaskedLM --timings results/all_oracle_timings.json
```

## Measuring a compiler change

To A/B a `torch.compile` change: sweep the corpus at two commits, diff the
kernels, and roll the diff up to a per-model end-to-end estimate.

```bash
# (a) Sweep each commit into a JSON (locked GPU path).
python scripts/bench_parallel.py repros/canonical --all-shapes \
    --gpus 0,1,2,3 --tag baseA --output base.json
python scripts/bench_parallel.py repros/canonical --all-shapes \
    --gpus 0,1,2,3 --tag headB --output head.json

# (b) Per-kernel A/B table — which kernel families moved.
python scripts/bench_report.py --compare base.json head.json --output-md ab.md

# (c) Per-model end-to-end rollup (no GPU needed).
python scripts/perf_ab_rollup.py --base base.json --head head.json
```

Step (b) emits a per-kernel markdown table of the biggest movers:

```markdown
### Improvements
| Kernel | Base (us) | Head (us) | Delta |
|--------|-----------|-----------|-------|
| var_mean_5a22dd21d88e[mobilenet_v3_large] | 348.1 | 18.0 | -94.8% |
| var_mean_42fad1ece813[mnasnet1_0]         | 241.4 | 23.9 | -90.1% |
```

Step (c) reuses the per-model accounting (fusible partitions + externs) to
report a shape-matched, occurrence-weighted per-model geomean/median.

The two are different metrics on purpose: the per-kernel table shows *which*
kernels moved, while the rollup weights each kernel by how a model actually
uses it and dilutes it by the extern (conv/GEMM/SDPA) time the change never
touches — so a large kernel win translates to a much smaller model-level
number. When reporting a model-level result, use the rollup's number, not one
extrapolated from the kernel table.

### PyTorch dashboard export

The dashboard exporter consumes one sweep, so any two uploaded CI workflows can
be compared later without rerunning either side:

```bash
# Occasionally regenerate model accounting on the target hardware. The
# persistent extern cache makes interrupted runs resumable. On a dedicated
# 8-GPU B200 runner, shard models across every GPU.
python scripts/generate_occurrence_sidecars.py \
    --corpus-root repros \
    --all \
    --output-dir results/b200/occurrences \
    --extern-cache results/b200/extern_cache.json \
    --devices 0,1,2,3,4,5,6,7 \
    --resume
python scripts/build_model_accounting.py \
    --occdir results/b200/occurrences \
    --output benchmarks/model_accounting/b200 \
    --prune

# Export every kernel sweep using that versioned accounting snapshot.
python scripts/dashboard_export.py \
    --input current.json \
    --model-accounting benchmarks/model_accounting/b200 \
    --timing auto \
    --ci-json test/test-reports/inductor_kernel_benchmark.json
```

It emits stable pattern/shape kernel records plus absolute projected model
latencies. The model values use the same occurrence weighting and unchanged
extern dilution and timing selection as `perf_ab_rollup.py`. Records include
the timing policy and accounting-artifact digest so incompatible workflows are
not combined. B200 accounting is versioned as one self-contained file per
suite/mode/model under `benchmarks/model_accounting/b200`, so individual models
can be added or refreshed without rewriting a shared artifact. The exporter
validates and aggregates the directory at runtime. These files can be
regenerated from occurrence sidecars with
`scripts/generate_occurrence_sidecars.py` and
`scripts/build_model_accounting.py`. The generator reuses the model-attribution
tracer and crash-isolated extern benchmark, writes a provenance manifest, and
fingerprints its persistent cache by GPU, PyTorch, CUDA, and timing method.
Generation and compaction fail by default if any external operation remains
unpriced.

## Extraction

Collect filtered 8-hop matmul regions from the saved real-world FX graphs:

```bash
python scripts/extract_matmuls.py
python scripts/extract_matmuls.py --validate-only
```

Selection walks producer and consumer edges as an undirected graph around
each matmul-related ATen anchor. Fanout, effects, shared constants and shape
metadata act as traversal barriers. Protected paths retain vertical MM chains,
shared activations and genuine horizontal result joins. Only outputs depending
on an in-region matmul are retained, followed by dead code and unused-input
elimination.

Exact directed structure matching produces one
`repros/canonical/<gemm-prefix>_<12-hex>/` directory per structure, with every
distinct retained input configuration in `shapes.json` and a PyTorch eager
`oracle.py`. The selection first keeps graphs with at most eight matmul-related
calls. Directed WL features at rounds 0, 1 and 2 then remove structures with
similarity strictly greater than 50%. Compatible shape arguments are then
coalesced in the executable replay interface. Export validates staged artifacts
before publication and preserves existing optimized oracles on identical
regeneration. Saved full graphs are reused.

See [matmul collection](docs/matmul_collection.md) for the essential code,
hash rules, source manifest, replay fidelity and validation commands.

Generate per-structure and per-configuration statistics from the saved corpus:

```bash
python scripts/report_matmul_corpus.py --output-dir results/matmul_statistics
```

## Original capture hook

The original fusible-partition capture path remains:

```python
from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import temporary_capture_for_merge

with temporary_capture_for_merge(Path("repros"), "my_model", suite="hf") as capture:
    graph_dir = Path("repros/models/hf/infer/my_model")
    install_capture_hook(
        str(capture.capture_dir),
        label="my_model",
        graph_dir=str(graph_dir),
        capture_only=True,
    )
    compiled_model(inputs)  # triggers capture
    uninstall_capture_hook()
    capture.merge()  # dedup + write to repros/canonical/
```

The hook:
- Filters view-only regions (permute, reshape, split — no real compute)
- Preserves strides, dtypes, and index bounds
- Lifts shape params from reshape/view/expand targets so repros run across shapes
- Validates each captured repro runs in eager before writing

For a model deployed with AOTInductor's runtime constant folding, pass
`constant_folding=True`: in inference graphs, work derived only from module
parameters, buffers and tensor attributes runs once at load time, so it is left
out of the regions and of model accounting. Merging does not remove a model's
earlier points, so recapture into a clean model entry when switching it.
Capture through `torch.compile`: a graph compiled by `aoti_compile_and_package`
itself has no AOTAutograd input metadata and is captured whole. Folding needs
one capture session per model (constants are tracked by storage address), and
`model_attribution` reports null region+extern/e2e ratios for a model with
folded work, since its e2e replay still runs that work.

## Benchmarking Architecture

<details>
<summary>Parallel runner internals</summary>

```
bench_parallel.py
├── N persistent worker subprocesses (one per GPU)
│   ├── Prefetch: background thread pre-imports next repro module
│   ├── Compile: torch.compile + CUDAGraph capture
│   ├── Time: do_bench with exclusive flock (no cross-process interference)
│   └── SOL: torch.add(src, 1, out=dst) at same transfer size
├── Shared inductor cache (skip redundant compilation across runs)
├── Worker recovery: respawn on CUDA device-side assert
└── Output: JSON with per-repro/per-graph metrics + __failures__ + __summary__
```

The same runner (workers, GPU lock, recovery) drives three workloads:

| Mode | What it times |
|------|---------------|
| default | the `torch.compile` path per repro |
| `--oracles` | the oracle reference implementations (per shape point; numerics check-gated) |
| `--full-graphs` | whole captured model graphs (latency + input constraints) |

Timing uses exclusive `flock` on a per-GPU lock file — multiple processes can compile in parallel, but only one times on a given GPU at once.

</details>

## Key Metrics

<details>
<summary>Per-repro measurements</summary>

| Metric | Meaning |
|--------|---------|
| `compiled_us` | Default inductor compilation, CUDAGraph replay |
| `compiled_nocudagraphs_us` | Default inductor compilation, direct call without CUDAGraph replay |
| `coord_descent_us` | With `coordinate_descent_tuning=True` |
| `memcopy_sol_us` | `torch.add(a, 1, out=b)` at same read+write bytes |
| `gap_default` | `compiled_us / memcopy_sol_us` — 1.0x = at bandwidth ceiling |

`compiled_nocudagraphs_us` is off by default (direct-call timing is noisy).
Pass `--compiled-nocudagraphs` to enable this measurement.

</details>

<details>
<summary>Per-repro: compile vs oracle reference (`--oracles`)</summary>

| Metric | Meaning |
|--------|---------|
| `oracle_us` | The reference implementation's time (CUDAGraph replay, GPU lock) |
| `compile_us` | The `torch.compile` time at the same shape point |
| `ratio` | `compile_us / oracle_us` — how far compile sits above the reference |
| `status` | Per-point verdict (e.g. `GOOD`); numerics-checked against the repro |
| `fallback` | True if a cross-hardware kernel was dispatched (reference may be soft) |

Per-shape data lives under `points_by_shape`; top-level `oracle_us`/`compile_us` are the median across valid points.

</details>

<details>
<summary>Per-model: aggregate vs measured e2e</summary>

| Metric | Meaning |
|--------|---------|
| `sum_fusible_us` | Σ oracle times over fusible kernel occurrences |
| `sum_extern_us` | Σ standalone times of non-fusible ops (conv/GEMM/SDPA — not oracle targets) |
| `corrected_parts_us` | `parts − G·(n_occ − n_graphs)`, the CUDAGraph launch-floor correction |
| `e2e_us` | Measured full-graph end-to-end time |
| `ratio_corrected` | `corrected_parts_us / e2e_us` — how well the aggregate accounts for e2e |

</details>

## Project Layout

<details>
<summary>File tree</summary>

```
capture_hook.py          # post-grad capture hook + partitioner
matmul_capture.py        # saved FX loader + protected 8-hop selection and region encoding
matmul_dedup.py          # exact structure/config equivalence and representative hashes
matmul_shape_cleanup.py # coalesce shape arguments without losing input settings
matmul_similarity.py    # directed WL structure similarity and representative selection
repro_harness.py         # shared: parse_shapes_config, SOL measurement, benchmark_repro
oracle_harness.py        # oracle check/bench infra + oracle_impl dispatch registry
merge_captures.py        # dedup + merge into canonical set
canonicalize_repros.py   # v2 format generation
byte_accounting.py       # effective bytes for gather/scatter/embedding
extract_reductions.py    # batch extraction from dynamo suite
extract_vllm.py          # vLLM/HF model capture driver

scripts/
  extract_matmuls.py     # complete filtered 8-hop export + persisted-artifact validation
  report_matmul_corpus.py # FX/input/output/matmul counts and source-verified motifs
  bench_parallel.py      # parallel GPU benchmark runner (compile, --oracles, --full-graphs)
  bench_oracles_parallel.py  # parallel oracle sweep (resumable)
  oracle_template.py     # starting point for new oracles
  validate_oracles.py    # oracle correctness/scope validation
  validate_eager.py      # subprocess-isolated eager validation
  validate_corpus_invariants.py  # hard/soft corpus invariants (CI / pre-commit)
  model_graph_accounting.py  # aggregate per-kernel times to a model-level estimate
  model_attribution.py   # reconcile the aggregate against measured e2e (launch-corrected)
  gc_corpus.py           # corpus reference-counting / migration transaction tool
  bench_report.py        # before/after comparison reports (raw per-kernel A/B table)
  perf_ab_rollup.py      # careful A/B: shape-matched, genai-excl, per-model-e2e rollup
  generate_occurrence_sidecars.py  # trace models + price externs reproducibly
  dashboard_export.py    # single-run PyTorch v3 kernel + projected-model records
  build_model_accounting.py  # compact occurrence sidecars for dashboard export
  test_adversarial.py    # infrastructure regression tests
  test_merge_into.py     # --merge-into adversarial tests
  test_bench_recovery.py # worker failure recovery tests

repros/
  canonical/             # 2283 flat canonical dirs; all selected matmul configs retained
  matmul_manifest.json   # overlapping 8-hop regions, selection rules and source checksums
  models/                # <suite>/<mode>/<model>/: manifest.json + full_graph_*.py (+ .meta.json)

benchmarks/              # frozen benchmark set definitions
```

</details>
