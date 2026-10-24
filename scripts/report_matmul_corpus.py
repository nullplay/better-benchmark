#!/usr/bin/env python3
"""Report actual FX, input/output, and source-verified matmul corpus statistics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import operator
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from matmul_capture import (CALL_OPS, RNG_EFFECT_SCHEMAS, concrete, dumps, is_effectful_node,
                            is_matmul_node, literal, load_matmul_graph,
                            primary_matmul_inputs, trace_region)
from oracle_harness import load_repro_module
from repro_harness import load_shape_configs

RANK2_OPS = frozenset({"mm", "addmm", "addmm_"})
BATCH_OPS = frozenset({"bmm", "baddbmm", "baddbmm_", "addbmm", "addbmm_"})
VECTOR_OPS = frozenset({"mv", "mv_", "addmv", "addmv_", "dot", "vdot"})
MOTIFS = ("vertical", "direct_shared_activation", "shared_activation", "horizontal_join",
          "horizontal", "small_reduction", "vertical_small_reduction", "native_vector",
          "rank2_mv_like", "batched_mv_like", "any_mv", "grouped_mm")
FLOW_METRICS = ("vertical_pairs", "longest_vertical_chain", "direct_shared_activation_pairs",
                "shared_activation_pairs", "shared_activation_sources", "horizontal_join_pairs",
                "horizontal_join_consumers")
METRICS = ("fx_nodes", "compute_calls", "aten_ops", "mm_related_ops", "formal_arguments",
           "logical_formal_arguments", "tensor_input_arguments", "tensor_input_leaves",
           "shape_arguments", "scalar_arguments", "container_arguments", "escaping_values",
           "output_tensor_leaves", "dead_compute_calls", "unused_input_arguments", "effect_calls",
           *FLOW_METRICS, "small_reduction_ops", "vertical_small_reduction_ops", "native_vector_ops",
           "rank2_mv_like_ops", "batched_mv_like_ops", "grouped_mm_ops")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def tensor_leaves(value, path=()):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, (*path, index))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_leaves(item, (*path, dumps(literal(key))))


def value_metadata(value):
    if isinstance(value, torch.Tensor):
        if not isinstance(value, FakeTensor):
            raise TypeError("Statistics require FakeTensor values, not real tensors")
        return {"kind": "tensor", "shape": [int(concrete(v)) for v in value.shape],
                "stride": [int(concrete(v)) for v in value.stride()], "dtype": str(value.dtype),
                "device": str(value.device), "layout": str(value.layout),
                "storage_offset": int(concrete(value.storage_offset())),
                "requires_grad": value.requires_grad, "is_conj": value.is_conj(), "is_neg": value.is_neg()}
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__, "items": [value_metadata(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[literal(key), value_metadata(item)] for key, item in value.items()]}
    return {"kind": "scalar", "value": literal(concrete(value))}


def storage_aliases(inputs):
    groups = {}
    for index, value in enumerate(inputs):
        for path, tensor in tensor_leaves(value):
            storage = tensor.untyped_storage()
            group = groups.setdefault((str(tensor.device), storage._cdata),
                                      {"nbytes": storage.nbytes(), "members": []})
            group["members"].append({"argument": index, "path": list(path)})
    return list(groups.values())


def bind_arguments(node, args=None, kwargs=None):
    args = node.args if args is None else args
    kwargs = node.kwargs if kwargs is None else kwargs
    positional = iter(args)
    return {argument.name: kwargs.get(argument.name, argument.default_value) if argument.kwarg_only else (
        kwargs[argument.name] if argument.name in kwargs else next(positional, argument.default_value))
        for argument in node.target._schema.arguments}


def tensor_argument(schema_type):
    kind = schema_type.kind()
    if kind == "TensorType":
        return True
    if kind in {"OptionalType", "ListType"}:
        return tensor_argument(schema_type.getElementType())
    if kind == "TupleType":
        return any(tensor_argument(item) for item in schema_type.elements())
    return False


class RecordingInterpreter(fx.Interpreter):
    def __init__(self, module):
        super().__init__(module, garbage_collect_values=False)
        self.values = {}
        self.arguments = {}

    def run_node(self, node):
        if node.op in CALL_OPS and hasattr(node.target, "_schema"):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            self.arguments[node] = bind_arguments(node, args, kwargs)
        value = super().run_node(node)
        self.values[node] = value
        return value


def flow_from_fx(module, values):
    """Serialize tensor dataflow from actual FX nodes, including complete-source effects."""
    nodes = tuple(module.graph.nodes)
    positions = {node: index for index, node in enumerate(nodes)}
    records = []
    for node in nodes:
        value = values[node]
        tensors = [tensor for _, tensor in tensor_leaves(value)]
        parents = set()
        if node.op in CALL_OPS and tensors:
            schema = getattr(node.target, "_schema", None)
            if schema is not None:
                arguments = bind_arguments(node)
                operands = [arguments[arg.name] for arg in schema.arguments if tensor_argument(arg.type)]
            elif node.op == "call_function" and node.target is operator.getitem:
                operands = [node.args[0]]
            else:
                operands = [node.args, node.kwargs]
            for operand in operands:
                refs = []
                fx.map_arg(operand, refs.append)
                parents.update(positions[parent] for parent in refs if any(tensor_leaves(values[parent])))
        schema = getattr(node.target, "_schema", None)
        name = schema.name.split("::", 1)[-1] if schema is not None else None
        records.append({"name": node.name, "op": node.op, "target": str(node.target), "parents": sorted(parents),
                        "matmul": name if is_matmul_node(node) else None,
                        "primary": sorted(positions[parent] for parent in primary_matmul_inputs(node)),
                        "tensor": bool(tensors), "non_scalar": any(tensor.ndim > 0 for tensor in tensors),
                        "activation": any(tensor.ndim > 0 and (tensor.is_floating_point() or tensor.is_complex())
                                          for tensor in tensors)})
    mm_positions = [index for index, record in enumerate(records) if record["matmul"]]
    bits = {position: 1 << rank for rank, position in enumerate(mm_positions)}
    dependent, ancestors = [], []
    for index, record in enumerate(records):
        dependent.append(record["op"] == "placeholder" and record["tensor"] or any(dependent[p] for p in record["parents"]))
        upstream = 0
        for parent in record["parents"]:
            upstream |= ancestors[parent]
        ancestors.append(upstream | bits.get(index, 0))
        record["dependent"] = dependent[-1]
        if record["matmul"]:
            record["matmul_ancestors"] = [position for position, bit in bits.items() if upstream & bit]
    return records


def analyze_flow(records, selected=None):
    """Analyze retained routes while checking horizontal independence in the full source."""
    selected = set(range(len(records))) if selected is None else set(selected)
    boundaries = {parent for node in selected for parent in records[node]["parents"]} - selected
    nodes = sorted(selected | boundaries)
    parents = {node: records[node]["parents"] if node in selected else [] for node in nodes}
    children = {node: [] for node in nodes}
    for node in nodes:
        for parent in parents[node]:
            children[parent].append(node)
    mm = {node for node in selected if records[node]["matmul"]}
    pairs, paths = set(), {}
    for anchor in sorted(mm):
        previous, queue = {anchor: None}, deque([anchor])
        while queue:
            node = queue.popleft()
            for child in children[node]:
                if child in previous:
                    continue
                previous[child] = node
                if child in mm:
                    path = [child]
                    while previous[path[-1]] is not None:
                        path.append(previous[path[-1]])
                    pairs.add((anchor, child))
                    paths[anchor, child] = list(reversed(path))
                else:
                    queue.append(child)
    longest = {node: 1 for node in mm}
    for source, target in sorted(pairs):
        longest[target] = max(longest[target], longest[source] + 1)

    def independent(first, second):
        return first != second and first not in records[second].get("matmul_ancestors", []) and second not in records[first].get("matmul_ancestors", [])

    origins, joins, join_pairs = {}, [], set()
    for node in nodes:
        branches = [(parent, origins.get(parent, set())) for parent in parents[node]]
        found = set()
        for (_, left), (_, right) in itertools.combinations(branches, 2):
            found.update(tuple(sorted((a, b))) for a in left - right for b in right - left if independent(a, b))
        if found:
            join_pairs.update(found)
            joins.append({"consumer": node, "pairs": sorted(found)})
        origins[node] = {node} if node in mm else set().union(*(branch for _, branch in branches)) if branches else set()
    sinks = {}
    for node in reversed(nodes):
        reachable = set()
        for child in children[node]:
            if child in mm:
                if node in records[child]["primary"]:
                    reachable.add(child)
            else:
                reachable.update(sinks[child])
        sinks[node] = reachable
    shared, direct, activations = set(), set(), []
    for node in nodes:
        direct_targets = {child for child in children[node] if child in mm and node in records[child]["primary"]}
        record = records[node]
        if not record["dependent"] or record["op"] == "get_attr" or not (
                record["activation"] or direct_targets and record["non_scalar"]):
            continue
        found_direct = {tuple(sorted((a, b))) for a, b in itertools.combinations(sorted(direct_targets), 2) if independent(a, b)}
        direct.update(found_direct)
        branches = [{child} if child in mm and node in records[child]["primary"] else set() if child in mm else sinks[child]
                    for child in children[node]]
        found = set()
        for left, right in itertools.combinations(branches, 2):
            found.update(tuple(sorted((a, b))) for a in left - right for b in right - left if independent(a, b))
        if found:
            shared.update(found)
            activations.append({"activation": node, "pairs": sorted(found)})
    counts = {"vertical_pairs": len(pairs), "longest_vertical_chain": max(longest.values(), default=0),
              "direct_shared_activation_pairs": len(direct), "shared_activation_pairs": len(shared),
              "shared_activation_sources": len(activations), "horizontal_join_pairs": len(join_pairs),
              "horizontal_join_consumers": len(joins)}
    flags = {"vertical": bool(pairs), "direct_shared_activation": bool(direct), "shared_activation": bool(shared),
             "horizontal_join": bool(join_pairs), "horizontal": bool(shared or join_pairs)}
    examples = {"vertical": list(paths.values())[:5], "shared_activation": activations[:5], "horizontal_join": joins[:5]}
    return counts, flags, examples, {node for pair in pairs for node in pair}


def describe_examples(examples, describe):
    return {"vertical": [[describe(node) for node in path] for path in examples["vertical"]],
            "shared_activation": [{"activation": describe(item["activation"]),
                                   "pairs": [[describe(a), describe(b)] for a, b in item["pairs"]]}
                                  for item in examples["shared_activation"]],
            "horizontal_join": [{"consumer": describe(item["consumer"]),
                                 "pairs": [[describe(a), describe(b)] for a, b in item["pairs"]]}
                                for item in examples["horizontal_join"]]}


def matmul_geometry(node, arguments, output, small_k, vertical):
    name = node.target._schema.name.split("::", 1)[1]
    if name in {"addmm", "addmm_", "_sparse_addmm", "_sparse_semi_structured_addmm"}:
        roles = ("mat1", "mat2")
    elif name in BATCH_OPS - {"bmm"}:
        roles = ("batch1", "batch2")
    elif name in {"addmv", "addmv_"}:
        roles = ("mat", "vec")
    elif name in {"mv", "mv_"}:
        roles = ("self", "vec")
    elif name == "linear":
        roles = ("input", "weight")
    else:
        roles = next((pair for pair in (("self", "mat2"), ("self", "other"), ("self", "tensor"))
                      if all(key in arguments for key in pair)), None)
    left, right = (arguments.get(role) for role in roles) if roles else (None, None)
    dimensions = {}
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        left_shape, right_shape = list(left.shape), list(right.shape)
        if name == "linear" and left_shape and len(right_shape) == 2:
            dimensions = {"M": math.prod(left_shape[:-1]), "K": left_shape[-1], "N": right_shape[0]}
        elif len(left_shape) >= 2 and len(right_shape) >= 2:
            dimensions = {"M": left_shape[-2], "K": left_shape[-1], "N": right_shape[-1]}
        elif left_shape and right_shape:
            dimensions = {"M": left_shape[-2] if len(left_shape) > 1 else 1,
                          "K": left_shape[-1], "N": right_shape[-1] if len(right_shape) > 1 else 1}
        if "grouped" in name and len(left_shape) == len(right_shape) == 2:
            dimensions["K"] = None
        dimensions = {key: int(concrete(value)) if value is not None else None for key, value in dimensions.items()}
    vector = name in VECTOR_OPS or name == "matmul" and isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and (left.ndim == 1 or right.ndim == 1)
    mv_like = dimensions.get("M") == 1 or dimensions.get("N") == 1
    small = name in RANK2_OPS | BATCH_OPS and dimensions.get("K", small_k) < small_k
    flags = {"native_vector": bool(vector), "rank2_mv_like": name in RANK2_OPS and mv_like,
             "batched_mv_like": name in BATCH_OPS and isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.ndim >= 3 and right.ndim >= 3 and mv_like,
             "small_reduction": small, "vertical_small_reduction": small and vertical,
             "grouped_mm": "grouped" in name}
    result = {"operator": name, "matrix_roles": roles, **dimensions, **flags,
              "tensor_arguments": {key: value_metadata(value) for key, value in arguments.items() if any(tensor_leaves(value))},
              "output": value_metadata(output)}
    if "grouped" in name:
        result["geometry_note"] = "Aggregate tensor dimensions; offset segment sizes are not materialized. Packed reduction K may be unknown."
    return result


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "histogram": {}}
    position = (len(values) - 1) * 0.9
    low, high = math.floor(position), math.ceil(position)
    return {"count": len(values), "min": values[0], "median": statistics.median(values),
            "mean": statistics.fmean(values), "p90": values[low] + (values[high] - values[low]) * (position - low),
            "max": values[-1], "histogram": dict(sorted(Counter(values).items()))}


def source_motifs(manifest, names, corpus_root, cache_dir):
    """Check every retained source cut, using ancestry before the cut and effect barriers."""
    observations = defaultdict(list)
    for source in manifest.get("source_graphs", []):
        cuts = [cut for cut in source["cuts"] if cut["pattern"] in names]
        if not cuts:
            continue
        path = corpus_root / source["graph"]
        identity = {"graph_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sidecar_sha256": hashlib.sha256(path.with_suffix(".meta.json").read_bytes()).hexdigest(),
                    "torch_version": str(torch.__version__),
                    "effect_classifier": {"version": 1, "rng_schemas": sorted(RNG_EFFECT_SCHEMAS)}}
        if any(identity[key] != source[key] for key in ("graph_sha256", "sidecar_sha256")):
            raise ValueError(f"Source graph changed: {source['graph']}")
        cache_file = cache_dir / (hashlib.sha256(source["graph"].encode()).hexdigest() + ".json")
        saved = json.loads(cache_file.read_text()) if cache_file.exists() else None
        if saved is None or saved.get("schema_version") != 1 or saved.get("identity") != identity:
            with torch._dynamo.config.patch(fake_tensor_cache_enabled=False):
                module = load_matmul_graph(path)
                records = flow_from_fx(module, {node: node.meta.get("val") for node in module.graph.nodes})
            saved = {"schema_version": 1, "identity": identity, "records": records}
            write_json(cache_file, saved)
        records = saved["records"]
        memo = {}
        for cut in cuts:
            selected = tuple(position for position in cut["selected_positions"] if records[position]["op"] in CALL_OPS)
            actual_anchors = {records[position]["name"]: records[position]["matmul"] for position in selected if records[position]["matmul"]}
            if any(actual_anchors.get(anchor) != op for anchor, op in cut.get("anchor_ops", {}).items()):
                raise ValueError(f"Source anchor schema changed: {source['graph']}")
            if not set(cut["anchors"]) <= actual_anchors.keys():
                raise ValueError(f"Source cut lost its anchor: {source['graph']}")
            if selected not in memo:
                counts, flags, examples, _ = analyze_flow(records, selected)
                examples = describe_examples(examples, lambda position: {"source_position": position,
                    "source_name": records[position]["name"], "target": records[position]["target"]})
                memo[selected] = {"counts": counts, "flags": flags, "examples": examples}
            observations[cut["pattern"], cut["shape_hash"]].append({"graph": source["graph"], **memo[selected]})
        print(f"Source motifs: {source['graph']} ({len(cuts)} cuts)", flush=True)
    return observations


def input_signature(metadata, tensor_fields):
    def project(value):
        if value["kind"] == "tensor":
            return {"kind": "tensor", **{key: value[key] for key in tensor_fields}}
        if value["kind"] in {"tuple", "list"}:
            return {"kind": value["kind"], "items": [project(item) for item in value["items"]]}
        if value["kind"] == "dict":
            return {"kind": "dict", "items": [[key, project(item)] for key, item in value["items"]]}
        return value
    return json.dumps([project(value) for value in metadata], sort_keys=True)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def report_corpus(canonical, manifest, output_dir, corpus_root, *, small_k=256, patterns=None, source_cache=None):
    canonical, output_dir, corpus_root = Path(canonical), Path(output_dir), Path(corpus_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(source_cache) if source_cache is not None else output_dir / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = manifest.get("patterns", [])
    if patterns is not None:
        names = set(patterns)
        entries = [entry for entry in entries if entry["name"] in names]
        if names != {entry["name"] for entry in entries}:
            raise ValueError("Unknown selected corpus pattern")
    names = {entry["name"] for entry in entries}
    sources = source_motifs(manifest, names, corpus_root, cache_dir)
    graph_rows, config_rows, interesting = [], [], defaultdict(list)
    packets = {"structures": Counter(), "settings": Counter()}
    details_path = output_dir / "configuration_details.jsonl"
    with details_path.open("w") as details_stream:
        for entry in entries:
            directory = canonical / entry["name"]
            meta = json.loads((directory / "meta.json").read_text())
            points = json.loads((directory / "shapes.json").read_text())["points"]
            module = load_repro_module(directory)
            traced = trace_region(module.Repro())
            nodes = tuple(traced.graph.nodes)
            positions = {node: index for index, node in enumerate(nodes)}
            placeholders = [node for node in nodes if node.op == "placeholder"]
            calls = [node for node in nodes if node.op in CALL_OPS]
            returned = next(node for node in nodes if node.op == "output")
            returned_nodes = returned.all_input_nodes
            physical = meta.get("replay_exact_structure", meta.get("exact_structure", {}))
            core_ids = {node: index for index, node in enumerate(calls)}
            core_ids.update(zip(placeholders, physical.get("input_nodes", range(len(calls), len(calls) + len(placeholders)))))
            logical_for_physical = defaultdict(list)
            for binding in meta.get("replay_interface", {}).get("logical_to_physical", []):
                logical_for_physical[binding["physical"]].append(binding["logical"])
            for node in placeholders:
                if not logical_for_physical[core_ids[node]]:
                    logical_for_physical[core_ids[node]] = [core_ids[node]]
            logical_roles = meta.get("exact_structure", {}).get("input_roles", {})
            input_roles = {node: next((logical_roles["input_" + str(logical)]
                                      for logical in logical_for_physical[core_ids[node]]
                                      if "input_" + str(logical) in logical_roles), logical_roles.get(node.name))
                           for node in placeholders}
            provenance = {record["core_id"]: record for record in meta.get("representative", {}).get("source_calls", [])}

            def describe(node):
                return {"name": node.name, "core_id": core_ids.get(node), "target": str(node.target),
                        **{key: value for key, value in provenance.get(core_ids.get(node), {}).items()
                           if key in {"source_name", "source_position"}}}

            ops = Counter(getattr(getattr(node.target, "_schema", None), "name", str(node.target)) for node in calls)
            mm_count = sum(is_matmul_node(node) for node in calls)
            if mm_count != entry["gemm_count"] or len(points) != entry["input_configs"]:
                raise ValueError(f"Saved corpus counts differ from actual FX: {directory.name}")
            live = set(returned_nodes)
            queue = list(live)
            for node in queue:
                for parent in node.all_input_nodes:
                    if parent not in live:
                        live.add(parent)
                        queue.append(parent)
            contains = {}
            for node in nodes:
                contains[node] = is_matmul_node(node) or any(contains[parent] for parent in node.all_input_nodes)
            base = {"canonical": directory.name, "configs": len(points), "fx_nodes": len(nodes),
                    "compute_calls": len(calls), "aten_ops": sum(count for name, count in ops.items() if name.startswith("aten::")),
                    "mm_related_ops": mm_count, "formal_arguments": len(placeholders), "escaping_values": len(returned_nodes),
                    "logical_formal_arguments": len(meta.get("exact_structure", {}).get("input_nodes", placeholders)),
                    "dead_compute_calls": sum(node not in live for node in calls),
                    "unused_input_arguments": sum(not node.users for node in placeholders),
                    "effect_calls": sum(is_effectful_node(node) for node in calls),
                    "outputs_with_matmul_ancestry": sum(contains[node] for node in returned_nodes), "op_counts": dict(ops)}
            configs = load_shape_configs(str(directory / "repro.py"))
            if len(configs) != len(points):
                raise ValueError(f"Shared input loader changed saved setting count: {directory.name}")
            rows, signatures = [], [set(), set(), set()]
            for point in points:
                shape_hash = point["shape_hash"]
                matches = [(label, config) for label, config in configs.items() if label == shape_hash or label.endswith("_" + shape_hash)]
                if len(matches) != 1:
                    raise ValueError(f"Saved input configuration is not uniquely loadable: {directory.name}/{shape_hash}")
                label, config = matches[0]
                with torch._dynamo.config.patch(fake_tensor_cache_enabled=False), FakeTensorMode(allow_fallback_kernels=False), torch.no_grad():
                    inputs = module.make_inputs(config)
                    interpreter = RecordingInterpreter(traced)
                    output = interpreter.run(*inputs)
                    flow = flow_from_fx(traced, interpreter.values)
                    counts, flags, examples, vertical = analyze_flow(flow)
                    geometries = [{**describe(node), **matmul_geometry(node, interpreter.arguments[node], interpreter.values[node], small_k, positions[node] in vertical)}
                                  for node in calls if is_matmul_node(node)]
                    flags.update({key: any(geometry[key] for geometry in geometries)
                                  for key in ("small_reduction", "vertical_small_reduction", "native_vector", "rank2_mv_like", "batched_mv_like", "grouped_mm")})
                    flags["any_mv"] = any(flags[key] for key in ("native_vector", "rank2_mv_like", "batched_mv_like"))
                    saved = point.get("replay_input_settings", point.get("input_settings", {}))
                    specs = saved.get("inputs", [])
                    input_meta = [value_metadata(value) for value in inputs]
                    row = {**base, "shape_hash": shape_hash, "config_sha256": point.get("config_sha256", shape_hash),
                           "config_label": label, "tensor_input_arguments": sum(any(tensor_leaves(value)) for value in inputs),
                           "tensor_input_leaves": sum(1 for value in inputs for _ in tensor_leaves(value)),
                           "shape_arguments": sum(spec["spec"]["kind"] == "shape" for spec in specs),
                           "scalar_arguments": sum(spec["spec"]["kind"] == "scalar" for spec in specs),
                           "container_arguments": sum(spec["spec"]["kind"] in {"tuple", "list", "dict"} for spec in specs),
                           "output_tensor_leaves": sum(1 for _ in tensor_leaves(output)), **counts,
                           "input_dtype_counts": dict(Counter(str(tensor.dtype) for value in inputs for _, tensor in tensor_leaves(value))),
                           "output_dtype_counts": dict(Counter(str(tensor.dtype) for _, tensor in tensor_leaves(output))),
                           **{key + "_ops": sum(geometry[key] for geometry in geometries)
                              for key in ("small_reduction", "vertical_small_reduction", "native_vector", "rank2_mv_like", "batched_mv_like", "grouped_mm")},
                           **{"has_" + key: flags[key] for key in MOTIFS}}
                    observed = sources[directory.name, shape_hash]
                    row["source_verified"] = bool(observed)
                    row["source_cuts"] = len(observed)
                    for key in FLOW_METRICS:
                        row["source_" + key] = max((item["counts"][key] for item in observed), default=0)
                    for key in ("vertical", "direct_shared_activation", "shared_activation", "horizontal_join", "horizontal"):
                        row["has_source_" + key] = any(item["flags"][key] for item in observed)
                        row["has_all_source_" + key] = bool(observed) and all(item["flags"][key] for item in observed)
                    detail = {"canonical": directory.name, "shape_hash": shape_hash, "config_sha256": row["config_sha256"],
                              "models": point.get("models", {}), "source_graph": meta.get("representative", {}).get("graph"),
                              "inputs": [{**describe(node), "argument": index, "role": input_roles[node],
                                          "logical_nodes": logical_for_physical[core_ids[node]],
                                          "value": value, "saved_spec": specs[index]["spec"] if specs else None}
                                         for index, (node, value) in enumerate(zip(placeholders, input_meta))],
                              "input_aliases": storage_aliases(inputs), "outputs": [{**describe(node), "argument": index,
                                  "value": value_metadata(interpreter.values[node])} for index, node in enumerate(returned_nodes)],
                              "output_tree": value_metadata(output), "matmul_geometries": geometries,
                              "counts": row, "motif_examples": describe_examples(examples, lambda position: describe(nodes[position])),
                              "source_observations": observed,
                              "replay_fidelity": point.get("fidelity", {})}
                    details_stream.write(json.dumps(detail, ensure_ascii=True, allow_nan=False) + "\n")
                    signatures[0].add(input_signature(input_meta, ("shape",)))
                    signatures[1].add(input_signature(input_meta, ("shape", "stride", "storage_offset")))
                    signatures[2].add(json.dumps({"inputs": input_meta, "aliases": detail["input_aliases"]}, sort_keys=True))
                rows.append(row)
                config_rows.append(row)
                for key in MOTIFS:
                    flag = row.get("has_source_" + key, row["has_" + key])
                    if flag and len(interesting[key]) < 8:
                        interesting[key].append({"canonical": directory.name, "shape_hash": shape_hash,
                                                 "mm_related_ops": mm_count, "source_verified": row["source_verified"]})
            graph_row = {**base, "input_shape_variants": len(signatures[0]), "input_shape_stride_variants": len(signatures[1]),
                         "input_geometry_variants": len(signatures[2])}
            for key in METRICS + tuple("source_" + key for key in FLOW_METRICS):
                graph_row[key] = max(row[key] for row in rows)
            for key in rows[0]:
                if key.startswith("has_") or key == "source_verified":
                    graph_row[key] = any(row[key] for row in rows)
            for side in ("input", "output"):
                key = side + "_dtype_counts"
                graph_row[key] = {dtype: max(row[key].get(dtype, 0) for row in rows)
                                  for dtype in {dtype for row in rows for dtype in row[key]}}
            graph_rows.append(graph_row)
            packets["structures"].update(ops)
            packets["settings"].update({key: count * len(points) for key, count in ops.items()})
            print(f"Graph statistics: {directory.name} ({len(points)} configurations)", flush=True)
    numeric = METRICS + tuple("source_" + key for key in FLOW_METRICS)
    result = {"schema_version": 1, "structures": len(graph_rows), "input_configs": len(config_rows),
              "source_totals": manifest.get("totals"),
              "source_verified_structures": sum(row["source_verified"] for row in graph_rows),
              "source_verified_input_configs": sum(row["source_verified"] for row in config_rows),
              "selection": manifest.get("selection"), "matmul_limit": manifest.get("matmul_limit"), "sampling": manifest.get("sampling"),
              "distributions": {unit: {metric: distribution(row[metric] for row in rows) for metric in numeric}
                                for unit, rows in (("structures", graph_rows), ("settings", config_rows))},
              "settings_per_structure": distribution(row["configs"] for row in graph_rows),
              "motifs": {key: {unit: sum(row["has_" + key] for row in rows)
                               for unit, rows in (("structures", graph_rows), ("settings", config_rows))} for key in MOTIFS},
              "source_motifs": {key: {unit: sum(row["has_source_" + key] for row in rows)
                                      for unit, rows in (("structures", graph_rows), ("settings", config_rows))}
                                for key in ("vertical", "direct_shared_activation", "shared_activation", "horizontal_join", "horizontal")},
              "op_totals": {unit: dict(sorted(counter.items())) for unit, counter in packets.items()},
              "tensor_dtypes": {side: {dtype: {"structures": sum(dtype in row[side + "_dtype_counts"] for row in graph_rows),
                                               "settings": sum(dtype in row[side + "_dtype_counts"] for row in config_rows),
                                               "leaves": sum(row[side + "_dtype_counts"].get(dtype, 0) for row in config_rows)}
                                       for dtype in sorted({dtype for row in config_rows for dtype in row[side + "_dtype_counts"]})}
                                for side in ("input", "output")},
              "notes": ["Structure metrics use the maximum across all own configurations; motif presence uses any configuration.",
                        "Settings are counted individually using their actual shapes, strides, and tensor metadata.",
                        "Source motifs require full-source independence; unavailable provenance is reported as unverified.",
                        "Shared activation denotes a shared data-dependent primary matmul operand; semantic parameter names are not inferred.",
                        "Tensor-container arguments overlap tensor-bearing arguments; tensor leaves are counted separately.",
                        f"Small reduction means ordinary MM or batched GEMM with K < {small_k}; grouped segment sizes are not inferred.",
                        "FakeTensor cache is disabled and real-kernel fallback is forbidden; these are not GPU timing or numerical results."]}
    write_json(output_dir / "statistics.json", result)
    write_json(output_dir / "interesting_examples.json", dict(interesting))
    write_csv(output_dir / "graph_stats.csv", graph_rows)
    write_csv(output_dir / "configuration_stats.csv", config_rows)
    lines = ["# Matmul corpus statistics", "", f"Structures: **{len(graph_rows)}**. Saved input configurations: **{len(config_rows)}**.", ""]
    if manifest.get("matmul_limit") and manifest.get("sampling"):
        limit, sampling = manifest["matmul_limit"], manifest["sampling"]
        if limit["stage"] == "before_similarity":
            stages = [("Exact extracted structures", limit["source_totals"]),
                      ("Matmul-count selection", sampling["source_totals"]), ("Similarity selection", manifest["totals"])]
        else:
            stages = [("Exact extracted structures", sampling["source_totals"]),
                      ("Similarity selection", limit["source_totals"]), ("Matmul-count selection", manifest["totals"])]
        lines.extend(["## Selection stages", "", "| Stage | Structures | Settings |", "| --- | ---: | ---: |",
                      *[f"| {stage} | {counts['structures']} | {counts['input_configs']} |" for stage, counts in stages], ""])
    lines.extend(["## Per-structure distributions", "", "| Metric | Min | Median | Mean | P90 | Max |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for metric in ("aten_ops", "mm_related_ops", "fx_nodes", "formal_arguments", "escaping_values"):
        values = result["distributions"]["structures"][metric]
        if values["count"]:
            lines.append(f"| {metric} | {values['min']} | {values['median']:g} | {values['mean']:.2f} | {values['p90']:g} | {values['max']} |")
    lines.extend(["", "## Extracted graph motifs", "", "| Motif | Structures | Settings |", "| --- | ---: | ---: |"])
    lines.extend(f"| {key} | {value['structures']} | {value['settings']} |" for key, value in result["motifs"].items())
    lines.extend(["", "## Source-verified motifs", "", "| Motif | Structures | Settings |", "| --- | ---: | ---: |"])
    lines.extend(f"| {key} | {value['structures']} | {value['settings']} |" for key, value in result["source_motifs"].items())
    lines.extend(["", "## Notes", "", *["- " + note for note in result["notes"]], ""])
    (output_dir / "summary.md").write_text("\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "repros", help="Root containing captured models/")
    parser.add_argument("--manifest", type=Path, help="Default: <corpus-root>/matmul_manifest.json")
    parser.add_argument("--canonical-dir", type=Path, help="Default: <manifest-parent>/canonical")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, help="Optional reusable cache keyed by source checksums and PyTorch version")
    parser.add_argument("--pattern", action="append", help="Report selected canonical names; repeatable")
    parser.add_argument("--small-k", type=int, default=256)
    args = parser.parse_args(argv)
    if args.small_k < 1:
        parser.error("--small-k must be positive")
    manifest_path = args.manifest or args.corpus_root / "matmul_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    canonical = args.canonical_dir or manifest_path.parent / "canonical"
    report = report_corpus(canonical, manifest, args.output_dir, args.corpus_root,
                           small_k=args.small_k, patterns=args.pattern, source_cache=args.source_cache)
    print(json.dumps({"structures": report["structures"], "input_configs": report["input_configs"],
                      "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
