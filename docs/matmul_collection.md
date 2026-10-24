# Matmul regions from captured FX graphs

The collection starts from real-world post-grad FX graphs already saved in
`repros/models/`. Each final graph contains at most eight matmul-related ATen
calls. The final structure and input-setting counts are recorded in
`repros/matmul_manifest.json`; statistics consume the actual serialized FX graphs.

## Pipeline

1. Extract protected 8-hop neighborhoods around actual matmul nodes.
2. Exclude effects and stop ordinary traversal through high-fanout hubs,
   shared constants, and shape metadata.
3. Remove escaping outputs without an internal matmul ancestor, then eliminate
   dead computation and unused boundary inputs.
4. Deduplicate exact directed structure and retain every distinct input setting.
5. Keep structures with **at most eight matmul calls**, preserving every own
   configuration of each eligible graph.
6. Remove eligible structures with directed WL similarity **strictly above 50%**.
7. Coalesce compatible shape arguments in the executable replay interface.

The manifest records source graph checksums, each retained anchor and cut, the
exact population before the matmul limit, the eligible population before WL,
and the final structure and input-setting counts. Every run rediscovers saved
graphs and extracts candidates from their actual FX DAGs.

## Essential code

| File | Responsibility |
| --- | --- |
| `matmul_capture.py` | Strict FakeTensor source loading, protected neighborhoods, directed structure and boundary encoding, executable reconstruction, and required replay generators. |
| `matmul_dedup.py` | Exact directed graph equivalence and representative SHA-256 digests. |
| `matmul_similarity.py` | Directed WL rounds 0, 1, and 2, multiset Jaccard similarity, and deterministic strict-threshold selection. |
| `matmul_shape_cleanup.py` | Replay shape coalescing across every own retained configuration; logical structure and settings stay intact. |
| `scripts/extract_matmuls.py` | The complete extraction, filtering, artifact validation, and publication pipeline. |
| `scripts/report_matmul_corpus.py` | Actual FX, input/output, matmul, and source-verified motif statistics for the retained corpus. |
| `full_graph_harness.py` | Existing saved-graph loading and consumer-based index domains, including `index_add`. |
| `input_codec.py` | Shared lossless tensor, scalar, shape, tuple, and list input encoding. |
| `repro_harness.py` | Shared input generation, storage aliases, and benchmarking. |
| `oracle_harness.py` | Existing eager oracle loading and correctness interface. |

## Neighborhood selection

Anchors are recognized by exact ATen schema names, including `mm`, `addmm`,
`bmm`, grouped and scaled MM, integer and sparse variants, `mv`, `addmv`,
`dot`, `vdot`, and other contraction schemas. All overloads are recognized;
effectful overloads are excluded. The source corpus has `mm`, `addmm`, `bmm`,
and `_grouped_mm` anchors. It has no native `aten.mv` or `aten.dot` calls;
MM and BMM configurations with vector-like dimensions remain included.

Producer-consumer edges are traversed in both directions. Breadth-first search
starts at each anchor and stops at distance eight. Repeated use of one operand
adds one neighbor. The source output node is terminal, so a return tuple cannot
connect unrelated outputs. Exported computation retains the original directed
edges and complete argument ports.

Fanout means distinct consumers in the **full source FX graph**, including
consumers outside a cut. At fanout **eight or greater**, ordinary expansion
stops. Another node may reach and retain the hub, but expansion from that hub
uses only protected matmul routes. Other BFS branches continue.

Protected value-data routes preserve:

- A matmul result flowing through computation into another matmul, including
  data-dependent scalar reductions.
- A shared primary contraction operand feeding independent matmuls through
  separate branches. For `addmm`, this is `mat1`, rather than its bias.
  Floating or complex non-scalar values and direct integer matmul operands
  qualify; Boolean masks and index tensors do not qualify as activation sources.
- Independent matmul results meeting at an actual computation. A return tuple
  is not a join. Protection ends at the joining consumer.

Protection never extends the radius or crosses an excluded effect. Horizontal
independence uses original source ancestry, including excluded effects, so
cutting a vertical path does not create a horizontal pair.

Literal-only deterministic `full`, `zeros`, `ones`, and `scalar_tensor` factories
can be retained locally but cannot bridge unrelated consumers. Shape metadata
queries are terminal; explicit shape and stride argument edges are excluded
from transit. No additional node, input, or output cap is applied during BFS.

### Effect barriers and boundary inputs

Compute calls are excluded when their operator schema is mutable, their FX node
reports `is_impure()`, or their target has a nondeterministic tag. The collector
also excludes three exact compiler RNG/state schemas whose tags omit those
effects: `prims::inductor_random`, `prims::inductor_randint`, and
`inductor_prims::inductor_reserve_rng_state`.

The first two accept explicit seeds for compiler lowering, but their eager
implementations consume global RNG state. The reservation operation advances its
CUDA generator. `prims::inductor_lookup_seed` is deterministic indexing and clone,
so it remains eligible and disappears through ordinary DCE when unused.
Placeholders and graph outputs are not classified as effects. An explicit seed
does not override a nondeterministic tag or the three schema exclusions.

Excluded calls and incident traversal edges are removed before BFS and route
construction. If live computation needs an excluded result, that result becomes
a boundary input. A mutation result is not replaced by its pre-mutation operand.
Replay preserves captured storage membership and conditions on supplied values;
outside-model implicit alias state is unavailable.

### Outputs and dead code

Matmul ancestry is propagated through the producer DAG **inside the extracted
region**. An initially escaping value is returned only if its internal ancestry
contains a matmul, including the producer itself. Backward liveness from those
outputs removes dead operations. Reconstruction lifts only inputs used by live
calls, so unused tensor, scalar, shape, and compound inputs disappear too.

The retained output set stays authoritative after DCE. Recomputing escapes from
full-source users would incorrectly restore removed outputs.

## Exact structure and input settings

Structure includes op and overload, directed wiring, argument positions and
names, repeated operands, nested containers, typed semantic constants, input
roles, and escaping output membership. Concrete shape and stride arguments are
lifted into configuration inputs; semantic constants remain in the structure.

Fingerprints and color refinement narrow candidates. Every merge requires
serialized equality or verified exact directed isomorphism with original node
colors and complete edge ports. Similarity never establishes equivalence.

The first representative in sorted source-file and FX order names a structure.
Names use up to its first three matmul schema names plus 12 hexadecimal digits
of the representative SHA-256, such as `mm_mm_<12-hex>` or
`addmm_addmm_grouped_mm_<12-hex>`. Full digests are retained and truncated
collisions rejected. IDs are representative labels, not a global canonical
labeling of every isomorphic graph.

Every retained structure keeps all its distinct configurations: shape, stride,
dtype, device, storage offset, layout, gradient and view flags, typed scalar
and shape values, nested containers, generators, observations, and alias
membership and allocation metadata. Matching accounts for structural
automorphisms and maps settings into representative input order.

Each setting is a `shapes.json.points` entry. `shape_hash` is the first eight
hexadecimal digits of the exact configured graph's SHA-256; `config_sha256`
stores the full digest. `input_settings` preserves the original descriptors
and aliases independently of compact replay inputs.

Only lifted shape arguments with identical typed specifications across **every
own retained setting** are coalesced into a physical replay input. Tensor and
ordinary scalar inputs are never coalesced. Logical structure, original settings,
and configuration hashes remain unchanged; physical replay structure, settings,
and hashes are recorded separately. Captured equality does not establish a
symbolic shape relationship.

## Directed structure similarity

Each compute call contributes its node label and directed WL neighborhood labels
after rounds one and two. Refinement includes all boundary nodes, complete
argument-port bundles, semantic constants, and escaping flags. Boundary nodes
are refined but are not feature centers. Concrete input settings do not enter
the score.

Similarity is multiset Jaccard:

```text
sum(min(feature_count_A, feature_count_B))
---------------------------------------
sum(max(feature_count_A, feature_count_B))
```

Candidates are visited by descending compute-call count, descending matmul-call
count, then ascending full structure SHA-256. A candidate is discarded when its
similarity to a previously retained representative is **greater than 50%**.
Exactly 50% survives. Threshold and witness comparisons use exact integers;
the closest qualifying representative wins, with earlier priority breaking ties.

The score describes local structural overlap. Different global connections and
rare fusion opportunities can still be discarded. Configurations of discarded
nonidentical structures are not transplanted into retained representatives.

The eight-matmul limit runs before similarity selection. An oversized graph
cannot act as a representative or discard a smaller eligible graph. The manifest
records both selection stages and their populations.

## Artifacts and replay fidelity

```text
repros/
  canonical/<matmul-prefix>_<12-hex>/
    repro.py              # Executable FX-generated Repro and shared input CLI
    meta.json             # Logical/replay structure, hashes, ops, source origin
    shapes.json           # Every own configuration and replay metadata
    oracle.py             # PyTorch eager placeholder
  matmul_manifest.json    # Source checksums, retained cuts, anchors, and filters
```

Regions overlap. Occurrences describe coverage and cannot be summed into model
latency. Existing model partition manifests remain disjoint; matmul provenance
is separate and participates in corpus GC.

Source sidecars do not retain every real tensor value or input allocation size.
Unknown alias allocation sizes remain `null` in exact settings. Replay uses a
separately recorded minimum safe aligned span. Necessary index and permutation
generators are inferred without changing captured descriptors or hashes.
Grouped-MM offsets use cumulative end offsets with valid synthetic expert
routing when original routing values are unavailable. These generators provide
valid replay inputs, not original numerical correlations.

Every oracle initially calls a cached `Repro()` in PyTorch eager mode through
`@oracle_impl`. Metadata marks it as an eager placeholder. Optimized oracles can
be filled in later through the existing BB interface.

## Regeneration and validation

Use the repository's existing PyTorch environment with NetworkX and NumPy:

```bash
# Run all stages; validate serialized artifacts before publication.
python scripts/extract_matmuls.py

# Regenerate separately without replacing existing artifacts.
python scripts/extract_matmuls.py --canonical-dir /tmp/bb_matmul/canonical

# Inspect one source; repeat --graph for additional exact source paths.
python scripts/extract_matmuls.py \
  --graph models/torchbench/train/BERT_pytorch/full_graph_000.py \
  --canonical-dir /tmp/bb_matmul_smoke/canonical

# Load and execute every persisted matmul repro and eager oracle setting.
python scripts/extract_matmuls.py --validate-only

# Report the retained corpus from its serialized graphs and all saved settings.
python scripts/report_matmul_corpus.py --output-dir /tmp/bb_matmul_statistics

# Mandatory gate before committing repros/ changes.
python scripts/validate_corpus_invariants.py
```

Defaults are `--radius 8 --fanout-limit 8 --similarity-threshold 0.5
--max-matmul-ops 8`. Effect exclusion, output DCE, exact deduplication, and shape
cleanup are always enabled. Only final retained canonical structures are
published; intermediate corpora are not materialized.

The statistics command writes `statistics.json`, `summary.md`, `graph_stats.csv`,
`configuration_stats.csv`, `configuration_details.jsonl`, and
`interesting_examples.json`. For a scratch corpus, pass its `--manifest` and
`--canonical-dir`; `--corpus-root` continues to identify the captured source graphs.

Validation loads serialized modules through the BB shared loader, traces their
actual FX graphs, checks exact logical and physical structure and configuration
hashes, generates every saved input setting, and executes repro and eager oracle
with strict FakeTensor inputs and no real-kernel fallback. It checks zero effects,
no unused physical inputs, matmul-dependent outputs, actual matmul metadata, and
the final matmul limit. Generated Python is consumed through FX rather than
parsed with regular expressions or AST tools.

Source tracing and persisted validation disable the FakeTensor dispatch cache
within their own scopes to preserve copy/view storage offsets on repeated calls.

Fake validation checks executable and tensor-metadata consistency. Numerical and
performance evaluation uses existing BB runners. Identical publication preserves
edited oracles; changed graph artifacts or hash collisions raise an error.

## Relationship to original BB extraction

The original hook saves post-grad graphs, then uses Inductor's fusion partitioning
and fusibility predicate to collect supported partitions. External GEMM calls
were excluded from those partitions while remaining in saved full graphs.
This collector reuses the full graphs and BB's artifact, input, and oracle
interfaces. Region selection is independent of current Inductor fusion decisions,
so it can preserve future vertical and horizontal MM fusion opportunities.
