#!/usr/bin/env python3
"""Export exact matmul-centered regions from saved real-world FX graphs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import shutil
import sys
import tempfile
import textwrap
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
import torch

from full_graph_harness import load_full_graph_sidecar
from input_codec import compact_from_spec
from matmul_capture import (MATMUL_ATEN_NAMES, CALL_OPS, GraphContext, MatmulTraversal, RegionEncoding,
                            configuration_graph, dumps, is_effectful_node, is_matmul_node,
                            load_matmul_graph, native_input_spec, prune_matmul_outputs, trace_region)
from matmul_dedup import (DIGEST_POLICY, MATCHER_POLICY, ExactGraphIndex, graph_data,
                         graph_digest, graph_from_data)
from matmul_similarity import (METRIC, REPRESENTATIVE_ORDER, SETTINGS_POLICY,
                               intersection_matrix, select_representatives, structural_features)

MANIFEST_NAME = "matmul_manifest.json"
SELECTION = {
    "radius": 8,
    "distance": "breadth-first hops on undirected source edges with transit barriers",
    "placeholder_get_attr_transit": True,
    "output_terminal": True,
    "semantic_barriers": True,
    "convex_closure": False,
    "node_cap": None,
    "gemm_cap": None,
    "fanout_limit": 8,
    "fanout": "distinct full-source consumers; protected MM routes may pass",
    "constant_transit": False,
    "metadata_transit": False,
    "shape_argument_transit": False,
    "protected_routes": ["MM to MM computation", "shared primary contraction operand", "horizontal result join"],
    "prune_effects": True,
    "effect_transit": False,
    "effect_policy": "Call-only mutable schema, FX-impure, nondeterministic tag, or explicit compiler RNG/state schema",
    "effects": "Exclude effect calls before traversal; protection cannot cross them; results may become boundary inputs",
    "outputs": "Keep escaping values with an internal matmul ancestor; DCE from those outputs only",
    "shape_interface": "Coalesce typed shape values equal across every own config; preserve logical structure and settings",
}
LIMIT_SETTINGS_POLICY = "Preserve every own input configuration of each eligible structure"

ORACLE_SOURCE = '''"""PyTorch eager baseline. Optimized oracle pending."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from oracle_harness import get_repro_instance, oracle_impl

REPRO_DIR = Path(__file__).resolve().parent
_REPRO = get_repro_instance(REPRO_DIR)


@oracle_impl(description="PyTorch eager baseline; optimization pending")
def oracle_forward(inputs):
    return _REPRO(*inputs)


if __name__ == "__main__":
    from oracle_harness import _runner_main
    raise SystemExit(_runner_main([str(REPRO_DIR), *sys.argv[1:]]))
'''


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
    temporary.replace(path)


def filter_manifest(manifest, selected, *, sampling=None, max_matmul_ops=None):
    """Filter source provenance while preserving complete retained configurations."""
    entries = {entry["name"]: entry for entry in manifest["patterns"]}
    selected = set(selected)
    if len(entries) != len(manifest["patterns"]) or not selected <= entries.keys():
        raise ValueError("Selected structures do not match the source manifest")
    if max_matmul_ops is not None:
        if type(max_matmul_ops) is not int or max_matmul_ops < 1:
            raise ValueError("Matmul limit must be a positive integer")
        if any(entries[name]["gemm_count"] > max_matmul_ops for name in selected):
            raise ValueError("A selected structure exceeds the matmul limit")
    fields = {"schema_version", "collection", "selection", "anchor_ops", "pytorch_version",
              "networkx_version", "digest_policy", "exact_matcher", "matmul_limit"}
    filtered = {key: copy.deepcopy(value) for key, value in manifest.items() if key in fields}
    filtered["patterns"] = [copy.deepcopy(entry) for entry in manifest["patterns"] if entry["name"] in selected]
    filtered["source_graphs"], anchor_counts = [], Counter()
    source_fields = {"graph", "suite", "mode", "model", "graph_sha256", "sidecar_sha256"}
    cut_fields = {"selected_positions", "anchors", "pattern", "shape_hash", "boundary_sources",
                  "escaping_positions", "anchor_ops"}
    for source in manifest["source_graphs"]:
        retained = {key: copy.deepcopy(value) for key, value in source.items() if key in source_fields}
        retained["cuts"] = [{key: copy.deepcopy(value) for key, value in cut.items() if key in cut_fields}
                            for cut in source["cuts"] if cut["pattern"] in selected]
        retained["anchors"] = sum(len(cut["anchors"]) for cut in retained["cuts"])
        for cut in retained["cuts"]:
            if set(cut.get("anchor_ops", {})) != set(cut["anchors"]):
                raise ValueError(f"Missing actual source anchor schemas: {source['graph']}")
            anchor_counts.update(cut["anchor_ops"].values())
        filtered["source_graphs"].append(retained)
    filtered["anchor_counts"] = dict(sorted(anchor_counts.items()))
    filtered["totals"] = {
        "source_graphs": len(filtered["source_graphs"]),
        "graphs_with_anchors": sum(bool(source["anchors"]) for source in filtered["source_graphs"]),
        "anchors": sum(source["anchors"] for source in filtered["source_graphs"]),
        "source_cuts": sum(len(source["cuts"]) for source in filtered["source_graphs"]),
        "structures": len(filtered["patterns"]),
        "input_configs": sum(entry["input_configs"] for entry in filtered["patterns"]),
    }
    sampling = sampling if sampling is not None else manifest.get("sampling")
    if sampling is not None:
        sampling_fields = {"metric", "threshold_percent", "representative_order", "comparison",
                           "removal_rule", "source_totals", "settings_policy"}
        filtered["sampling"] = {key: copy.deepcopy(value) for key, value in sampling.items() if key in sampling_fields}
    if max_matmul_ops is not None:
        filtered["selection"]["max_matmul_ops"] = max_matmul_ops
        filtered["matmul_limit"] = {
            "max_ops": max_matmul_ops, "stage": "before_similarity",
            "source_totals": copy.deepcopy(manifest["totals"]),
            "removed_structures": manifest["totals"]["structures"] - filtered["totals"]["structures"],
            "removed_input_configs": manifest["totals"]["input_configs"] - filtered["totals"]["input_configs"],
            "settings_policy": LIMIT_SETTINGS_POLICY,
        }
    filtered["validation"] = {"status": "pending"}
    return filtered


def sample_manifest(manifest, graphs, threshold=Fraction(1, 2)):
    """Select directed WL representatives from the matmul-eligible population."""
    if len(graphs) != len(manifest["patterns"]):
        raise ValueError("Structure graphs do not match the manifest population")
    counts, digests, features = [], [], []
    for graph, entry in zip(graphs, manifest["patterns"]):
        digest = graph_digest(graph)
        if digest[:12] != entry["pattern_hash"]:
            raise ValueError(f"Structure hash differs from the manifest: {entry['name']}")
        counts.append(sum(json.loads(data["color"])[0] in CALL_OPS for _, data in graph.nodes(data=True)))
        digests.append(digest)
        features.append(structural_features(graph))
    order = sorted(range(len(graphs)), key=lambda i: (-counts[i], -manifest["patterns"][i]["gemm_count"], digests[i]))
    result = select_representatives(intersection_matrix(features), [sum(counter.values()) for counter in features], order, threshold)
    percentage = threshold * 100
    sampling = {
        "metric": METRIC, "threshold_percent": int(percentage) if percentage.denominator == 1 else float(percentage),
        "representative_order": REPRESENTATIVE_ORDER, "comparison": "gt", "removal_rule": "similarity > threshold",
        "source_totals": manifest["totals"], "settings_policy": SETTINGS_POLICY,
    }
    return filter_manifest(manifest, [manifest["patterns"][i]["name"] for i in result["kept"]], sampling=sampling)


def render_repro(module, radius=8):
    """Serialize FX-generated code and its actual Python global bindings."""
    code = module.graph.python_code("self")
    imports = []
    for name, value in sorted(code.globals.items()):
        if name in {"torch", "inf", "nan"}:
            continue
        if value is type(None):
            imports.append(f"{name} = type(None)")
        elif isinstance(value, ModuleType):
            imports.append(f"import {value.__name__} as {name}")
        else:
            owner = getattr(value, "__module__", None)
            qualified = getattr(value, "__qualname__", None)
            if not owner or not qualified or "<locals>" in qualified:
                raise TypeError(f"FX global cannot be imported: {name} ({type(value).__name__})")
            resolved = importlib.import_module(owner)
            for part in qualified.split("."):
                resolved = getattr(resolved, part)
            if resolved is not value:
                raise ValueError(f"FX global binding is not reproducible: {name}")
            alias = f"_fx_global_{name}"
            imports.extend([f"import {owner} as {alias}", f"{name} = {alias}.{qualified}"])
    bindings = "\n".join(imports)
    forward = textwrap.indent(code.src.strip(), "    ")
    description = f'{radius}-hop'
    return f'''"""{description} region from a captured real-world FX graph.

Exact structure and source provenance: meta.json and ../../matmul_manifest.json.
All retained input settings: shapes.json.
"""
import sys
from math import inf, nan
from pathlib import Path

import torch
import torch._inductor.inductor_prims
{bindings}

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import default_shape_config, make_inputs_from_config, benchmark_repro

_repro_version = 3


class Repro(torch.nn.Module):
{forward}


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
'''


def remap_settings(encoding, mapping):
    inputs = sorted([{"node": mapping[e["node"]], "spec": copy.deepcopy(e["spec"])}
                     for e in encoding.inputs], key=lambda e: e["node"])
    aliases = [{**alias, "members": sorted(
        [{"node": mapping[m["node"]], "path": m["path"]} for m in alias["members"]], key=dumps)}
        for alias in encoding.aliases]
    return {"inputs": inputs, "aliases": sorted(aliases, key=dumps)}


def replay_config(settings, constraints, mapping):
    specs = {e["node"]: native_input_spec(e["spec"]) for e in settings["inputs"]}

    def leaf(node, path):
        spec = specs[node]
        for position in path:
            if spec["kind"] not in {"tuple", "list"}:
                raise TypeError("Replay requires tuple/list paths for nested tensor inputs")
            spec = spec["items"][position]
        return spec

    group_bytes, unknown_sizes = [], []
    for group, alias in enumerate(settings["aliases"]):
        minimum, alignment = 0, 1
        for member in alias["members"]:
            spec = leaf(member["node"], member["path"])
            dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
            itemsize = dtype.itemsize
            shape, stride = spec["shape"], spec["stride"]
            extent = 0 if 0 in shape else 1 + sum((d - 1) * s for d, s in zip(shape, stride))
            minimum = max(minimum, (spec.get("storage_offset", 0) + extent) * itemsize)
            alignment = math.lcm(alignment, itemsize)
            spec["alias_group"] = group
        nbytes = alias["nbytes"]
        if nbytes is None:
            nbytes = (minimum + alignment - 1) // alignment * alignment
            unknown_sizes.append(group)
        if nbytes < minimum or nbytes % alignment:
            raise ValueError(f"Captured alias allocation cannot contain its views: group {group}")
        group_bytes.append(nbytes)
    overrides = []
    for override in constraints["overrides"]:
        entry = {**override, "node": mapping[override["node"]]}
        leaf(entry["node"], entry["path"])["gen"] = entry["gen"]
        overrides.append(entry)
    fidelity = dict(constraints["fidelity"])
    if unknown_sizes:
        fidelity["unretained_alias_allocation_groups"] = unknown_sizes
        fidelity["alias_allocation_replay"] = "minimum safe span; original allocation size unavailable"
    return {
        "inputs": [compact_from_spec(specs[e["node"]]) for e in settings["inputs"]],
        "alias_group_nbytes": group_bytes,
        "replay_overrides": sorted(overrides, key=dumps),
        "fidelity": fidelity,
    }


def cleanup_shape_interface(directory, meta, points, radius):
    """Reduce replay arity without using captured shape equality in sampling."""
    from oracle_harness import load_repro_module
    from matmul_shape_cleanup import coalesced_layout, coalesce_module, replay_settings, replay_structure

    core = graph_from_data(meta['exact_structure']['graph'])
    physical, layout = replay_structure(core, coalesced_layout(core, points))
    traced = trace_region(load_repro_module(directory).Repro())
    module = coalesce_module(traced, layout)
    logical_order = {node: position for position, node in enumerate(layout['logical_input_nodes'])}
    representatives = {entry['physical']: entry['representative'] for entry in layout['physical_ids']}
    for point in points:
        logical_inputs = point['inputs']
        if len(logical_inputs) != len(logical_order):
            raise ValueError('Logical replay input arity changed before shape cleanup')
        settings = replay_settings(point['input_settings'], layout)
        point['inputs'] = [logical_inputs[logical_order[representatives[node]]] for node in layout['physical_input_nodes']]
        point['replay_input_settings'] = settings
        point['replay_config_sha256'] = graph_digest(configuration_graph(physical, settings['inputs'], settings['aliases']))
    meta['replay_interface'] = layout
    meta['replay_exact_structure'] = {
        'sha256': graph_digest(physical), 'graph': graph_data(physical),
        'input_nodes': layout['physical_input_nodes'],
        'input_roles': {f'input_{node}': json.loads(physical.nodes[node]['color'])[0]
                        for node in layout['physical_input_nodes']},
        'escaping_nodes': meta['exact_structure']['escaping_nodes'],
        'effect_nodes': meta['exact_structure']['effect_nodes'],
    }
    (directory / 'repro.py').write_text(render_repro(module, radius))
    write_json(directory / 'meta.json', meta)
    write_json(directory / 'shapes.json', {'points': points})


class Collection:
    def __init__(self, radius=8, fanout_limit=8, similarity_threshold=Fraction(1, 2), max_matmul_ops=8):
        if type(radius) is not int or radius < 0:
            raise ValueError("Neighborhood radius must be a nonnegative integer")
        if type(fanout_limit) is not int or fanout_limit < 1:
            raise ValueError("Fanout limit must be a positive integer")
        if not isinstance(similarity_threshold, Fraction) or not 0 < similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be a Fraction in (0, 1]")
        if type(max_matmul_ops) is not int or max_matmul_ops < 1:
            raise ValueError("Matmul limit must be a positive integer")
        self.radius = radius
        self.fanout_limit = fanout_limit
        self.similarity_threshold = similarity_threshold
        self.max_matmul_ops = max_matmul_ops
        self.collection = f'matmul_effect_barrier_{radius}hop_v1'
        self.selection = {**SELECTION, 'radius': radius, 'fanout_limit': fanout_limit}
        self.structures = ExactGraphIndex()
        self.patterns = []
        self.sources = []
        self.anchor_counts = Counter()

    def collect(self, path, corpus):
        gm, metadata = load_matmul_graph(path, with_metadata=True)
        context = GraphContext(gm, metadata)
        traversal = MatmulTraversal(context, self.fanout_limit)
        relative = path.relative_to(corpus)
        parts = path.relative_to(corpus / "models").parts
        source = {"suite": parts[0], "mode": parts[1], "model": "/".join(parts[2:-1])}
        source.update({k: v for k, v in load_full_graph_sidecar(path).get("source", {}).items()
                       if k in {"suite", "mode", "model"} and isinstance(v, str) and v})
        model = "/".join(source[k] for k in ("suite", "mode", "model"))
        proposals = {}
        for anchor, selected in context.neighborhoods(radius=self.radius, names=MATMUL_ATEN_NAMES, traversal=traversal):
            proposals.setdefault(selected, []).append(anchor)
            self.anchor_counts[anchor.target._schema.name.split("::", 1)[1]] += 1
        cuts = []
        for raw_selected, anchors in proposals.items():
            if set(raw_selected) & traversal.excluded:
                raise ValueError(f'Effect barrier was included in a neighborhood: {path}')
            selected, outputs, _ = prune_matmul_outputs(context, raw_selected)
            if any(context.positions[anchor] not in selected for anchor in anchors):
                raise ValueError(f'Output pruning removed its own matmul anchor: {path}')
            encoding = RegionEncoding(context, selected, outputs)
            core = encoding.graph
            index, mapping, new = self.structures.add(core)
            if new:
                gemms = [n.target._schema.name.split("::", 1)[1] for n in encoding.calls
                         if is_matmul_node(n, MATMUL_ATEN_NAMES)]
                full_hash = graph_digest(core)
                prefix = "_".join(op.lstrip("_") for op in gemms[:3])
                self.patterns.append({
                    "name": f"{prefix}_{full_hash[:12]}", "sha256": full_hash,
                    "code": render_repro(encoding.build_module(), self.radius),
                    "input_nodes": [e["node"] for e in encoding.inputs],
                    "escape_ids": encoding.escape_ids, "effect_ids": encoding.effects,
                    "n_ops": len(encoding.calls), "gemms": gemms,
                    "origin_ops": [str(n.target) for n in encoding.calls],
                    "representative": {"graph": str(relative), "selected_positions": list(selected),
                                       "anchors": [n.name for n in anchors]},
                    "configs": [], "config_index": ExactGraphIndex(),
                    "models": Counter(), "sources": set(), "source_cuts": 0,
                })
                self.patterns[-1]['representative']['source_calls'] = [
                    {'core_id': encoding.call_ids[node], 'source_position': context.positions[node],
                     'source_name': node.name, 'schema_name': getattr(getattr(node.target, '_schema', None), 'name', None),
                     'target': str(node.target)} for node in encoding.calls]
            pattern = self.patterns[index]
            settings = remap_settings(encoding, mapping)
            configured = configuration_graph(self.structures.graphs[index], settings["inputs"], settings["aliases"])
            config, config_mapping, config_new = pattern["config_index"].add(configured)
            if config_new:
                config_hash = graph_digest(configured)
                if any(p["shape_hash"] == config_hash[:8] for p in pattern["configs"]):
                    raise ValueError("Truncated input configuration hash collision")
                pattern["configs"].append({
                    "shape_hash": config_hash[:8], "config_sha256": config_hash,
                    "input_settings": settings,
                    **replay_config(settings, encoding.replay_constraints(), mapping),
                    "models": {},
                })
            point = pattern["configs"][config]
            point["models"].setdefault(model, {"occurrences": 0})["occurrences"] += len(anchors)
            pattern["models"][model] += len(anchors)
            pattern["sources"].add(str(relative))
            pattern["source_cuts"] += 1
            cuts.append({"selected_positions": list(selected), "anchors": [n.name for n in anchors],
                         "pattern": pattern["name"], "shape_hash": point["shape_hash"],
                         "boundary_sources": {str(config_mapping[mapping[int(node)]]): name
                                              for node, name in encoding.input_sources.items()},
                         "escaping_positions": sorted(context.positions[node] for node in outputs),
                         "anchor_ops": {node.name: node.target._schema.name.split('::', 1)[1] for node in anchors}})
        sidecar = path.with_suffix(".meta.json")
        self.sources.append({"graph": str(relative), **source,
                             "graph_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                             "anchors": sum(len(a) for a in proposals.values()), "cuts": cuts})

    def write(self, canonical):
        manifest = {
            "schema_version": 1, "collection": self.collection, "selection": self.selection,
            "anchor_ops": sorted(MATMUL_ATEN_NAMES), "pytorch_version": str(torch.__version__),
            "networkx_version": nx.__version__, "digest_policy": DIGEST_POLICY,
            "exact_matcher": MATCHER_POLICY,
            "totals": {"source_graphs": len(self.sources),
                       "graphs_with_anchors": sum(s["anchors"] > 0 for s in self.sources),
                       "anchors": sum(self.anchor_counts.values()),
                       "source_cuts": sum(len(s["cuts"]) for s in self.sources),
                       "structures": len(self.patterns),
                       "input_configs": sum(len(p["configs"]) for p in self.patterns)},
            "anchor_counts": dict(sorted(self.anchor_counts.items())),
            "patterns": [{"name": p["name"], "pattern_hash": p["sha256"][:12],
                          "input_configs": len(p["configs"]), "gemm_count": len(p["gemms"])}
                         for p in self.patterns],
            "source_graphs": self.sources,
        }
        eligible = {entry["name"] for entry in manifest["patterns"] if entry["gemm_count"] <= self.max_matmul_ops}
        graphs = [graph for graph, pattern in zip(self.structures.graphs, self.patterns) if pattern["name"] in eligible]
        manifest = filter_manifest(manifest, eligible, max_matmul_ops=self.max_matmul_ops)
        manifest = sample_manifest(manifest, graphs, self.similarity_threshold)
        retained = {entry["name"] for entry in manifest["patterns"]}
        canonical.mkdir(parents=True, exist_ok=True)
        hashes = set()
        for index, pattern in enumerate(self.patterns):
            if pattern["name"] not in retained:
                continue
            pattern_hash = pattern["sha256"][:12]
            if pattern_hash in hashes:
                raise ValueError("Truncated structure hash collision")
            hashes.add(pattern_hash)
            directory = canonical / pattern["name"]
            directory.mkdir()
            core = self.structures.graphs[index]
            roles = {f"input_{node}": json.loads(core.nodes[node]["color"])[0]
                     for node in pattern["input_nodes"]}
            meta = {
                "pattern_hash": pattern_hash, "kind": "matmul_region", "collection": self.collection,
                "n_ops": pattern["n_ops"], "origin_ops": pattern["origin_ops"],
                "gemm_count": len(pattern["gemms"]), "gemm_ops": dict(Counter(pattern["gemms"])),
                "models": sorted(pattern["models"]), "n_models": len(pattern["models"]),
                "occurrences": sum(pattern["models"].values()), "source_cuts": pattern["source_cuts"],
                "source_graphs": sorted(pattern["sources"]), "representative": pattern["representative"],
                "selection": self.selection,
                "exact_structure": {"sha256": pattern["sha256"], "digest_policy": DIGEST_POLICY,
                                    "matcher": MATCHER_POLICY, "graph": graph_data(core),
                                    "input_nodes": pattern["input_nodes"], "input_roles": roles,
                                    "escaping_nodes": pattern["escape_ids"], "effect_nodes": pattern["effect_ids"]},
                "input_config_hash": "SHA-256 of exact structure, retained settings and alias membership; first 8 hex digits",
                "oracle": {"implementation": "pytorch_eager", "status": "placeholder"},
                "model_accounting": "overlapping regions; excluded from disjoint model partition manifests",
            }
            (directory / "repro.py").write_text(pattern["code"])
            (directory / "oracle.py").write_text(ORACLE_SOURCE)
            write_json(directory / "meta.json", meta)
            points = copy.deepcopy(pattern['configs'])
            write_json(directory / "shapes.json", {"points": points})
            cleanup_shape_interface(directory, meta, points, self.radius)
        return manifest


@torch._dynamo.config.patch(fake_tensor_cache_enabled=False)
def validate(canonical, manifest, progress=None):
    """Load the persisted corpus, check exact structure/configuration, execute fake eager."""
    from oracle_harness import _load_oracle_module, load_repro_module
    from repro_harness import load_shape_configs
    from torch._subclasses.fake_tensor import FakeTensorMode

    structures, points_checked = ExactGraphIndex(), 0
    for entry in manifest["patterns"]:
        directory = canonical / entry["name"]
        meta = json.loads((directory / "meta.json").read_text())
        points = json.loads((directory / "shapes.json").read_text())["points"]
        exact = meta["exact_structure"]
        core = graph_from_data(exact["graph"])
        if graph_digest(core) != exact["sha256"] or exact["sha256"][:12] != meta["pattern_hash"]:
            raise ValueError(f"Structure hash mismatch: {directory.name}")
        _, _, new = structures.add(core)
        if not new or len(points) != entry["input_configs"]:
            raise ValueError(f"Duplicate structure or missing input configurations: {directory.name}")
        module = load_repro_module(directory)
        configs = list(load_shape_configs(str(directory / "repro.py")).values())
        if len(configs) != len(points) or getattr(module, "_repro_version", None) != 3:
            raise ValueError(f"Shared loader lost an input configuration: {directory.name}")
        traced = trace_region(module.Repro())
        placeholders = [n for n in traced.graph.nodes if n.op == "placeholder"]
        physical, layout = core, meta.get('replay_interface')
        input_nodes = exact['input_nodes']
        if layout is not None:
            from matmul_shape_cleanup import coalesced_layout, replay_settings, replay_structure
            physical, expected_layout = replay_structure(core, coalesced_layout(core, points))
            replay = meta['replay_exact_structure']
            if expected_layout != layout or graph_data(physical) != replay['graph'] or graph_digest(physical) != replay['sha256']:
                raise ValueError(f'Shape replay interface differs from every retained point: {directory.name}')
            input_nodes = layout['physical_input_nodes']
        if len(placeholders) != len(input_nodes):
            raise ValueError(f"Serialized input arity differs: {directory.name}")
        if exact['effect_nodes'] or any(is_effectful_node(node) for node in traced.graph.nodes):
            raise ValueError(f'An effect call survived extraction: {directory.name}')
        if any(not node.users for node in placeholders):
            raise ValueError(f'An unused input survived DCE: {directory.name}')
        matmuls = Counter(node.target._schema.name.split('::', 1)[1] for node in traced.graph.nodes if is_matmul_node(node))
        if dict(matmuls) != meta['gemm_ops'] or sum(matmuls.values()) != entry['gemm_count']:
            raise ValueError(f'Serialized matmul operators differ from metadata: {directory.name}')
        if sum(matmuls.values()) > manifest['selection']['max_matmul_ops']:
            raise ValueError(f'A graph exceeds the matmul limit: {directory.name}')
        returned = next(node for node in traced.graph.nodes if node.op == 'output').all_input_nodes
        contains = {}
        for node in traced.graph.nodes:
            contains[node] = is_matmul_node(node) or any(contains[parent] for parent in node.all_input_nodes)
        if any(not contains[node] for node in returned):
            raise ValueError(f'An output has no internal matmul producer: {directory.name}')
        _, oracle, _ = _load_oracle_module(directory)
        configured_index = ExactGraphIndex()
        for position, (point, config) in enumerate(zip(points, configs)):
            settings = point["input_settings"]
            configured = configuration_graph(core, settings["inputs"], settings["aliases"])
            if graph_digest(configured) != point["config_sha256"] or point["shape_hash"] != point["config_sha256"][:8]:
                raise ValueError(f"Input configuration hash mismatch: {directory.name}")
            _, _, new = configured_index.add(configured)
            if not new:
                raise ValueError(f"Duplicate input configuration: {directory.name}")
            actual_settings = settings
            if layout is not None:
                actual_settings = replay_settings(settings, layout)
                if actual_settings != point['replay_input_settings']:
                    raise ValueError(f'Coalescing lost retained settings: {directory.name}')
                digest = graph_digest(configuration_graph(physical, actual_settings['inputs'], actual_settings['aliases']))
                if digest != point['replay_config_sha256']:
                    raise ValueError(f'Physical replay configuration hash mismatch: {directory.name}')
                logical = replay_config(settings, {'overrides': point['replay_overrides'], 'fidelity': point['fidelity']},
                                        {entry['node']: entry['node'] for entry in settings['inputs']})['inputs']
                positions = {node: i for i, node in enumerate(layout['logical_input_nodes'])}
                representatives = {entry['physical']: entry['representative'] for entry in layout['physical_ids']}
                expected_inputs = [logical[positions[representatives[node]]] for node in input_nodes]
                if expected_inputs != point['inputs']:
                    raise ValueError(f'Physical input generation changed retained replay inputs: {directory.name}')
            with FakeTensorMode(allow_fallback_kernels=False), torch.no_grad():
                inputs = module.make_inputs(config)
                if position == 0:
                    specs = {e["node"]: e["spec"] for e in actual_settings["inputs"]}
                    for node, value in zip(placeholders, inputs):
                        node.meta["val"] = value
                    metadata = {"boundary_specs": {n.name: specs[node_id]
                                                   for n, node_id in zip(placeholders, input_nodes)},
                                "input_roles": {n.name: json.loads(physical.nodes[node_id]["color"])[0]
                                                for n, node_id in zip(placeholders, input_nodes)}}
                    context = GraphContext(traced, metadata)
                    selected = tuple(i for i, n in enumerate(context.nodes) if n.op in CALL_OPS)
                    actual = RegionEncoding(context, selected, set(returned)).graph
                    check = ExactGraphIndex()
                    check.add(physical)
                    if check.add(actual)[2]:
                        raise ValueError(f"Serialized Python changed the exact FX structure: {directory.name}")
                module.Repro()(*inputs)
                oracle(module.make_inputs(config))
            points_checked += 1
        if progress is not None:
            progress({'structures': len(structures.graphs), 'input_configs': points_checked})
    if len(structures.graphs) != manifest["totals"]["structures"] or points_checked != manifest["totals"]["input_configs"]:
        raise ValueError("Canonical totals do not match the collection manifest")
    return {"device": "fake", "structures": len(structures.graphs), "input_configs": points_checked}


def publish(staged, canonical, manifest):
    """Publish verified artifacts, preserving an existing optimized oracle."""
    canonical.mkdir(parents=True, exist_ok=True)
    manifest_path = canonical.parent / MANIFEST_NAME
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        comparable = lambda data: {key: value for key, value in data.items() if key != "validation"}
        if comparable(previous) != comparable(manifest):
            raise FileExistsError("Existing matmul manifest describes a different collection; choose a separate canonical directory")
        manifest["validation"] = {**previous.get("validation", {}), **manifest.get("validation", {})}
    existing_hashes = {}
    for path in canonical.glob("*/meta.json"):
        existing_hashes[json.loads(path.read_text())["pattern_hash"]] = path.parent.name
    for entry in manifest["patterns"]:
        name = entry["name"]
        other = existing_hashes.get(entry["pattern_hash"])
        if other is not None and other != name:
            raise ValueError(f"Pattern hash already belongs to another canonical graph: {other}")
        target = canonical / name
        if target.exists():
            for filename in ("repro.py", "meta.json", "shapes.json"):
                if (target / filename).read_bytes() != (staged / name / filename).read_bytes():
                    raise FileExistsError(f"Refusing to overwrite changed canonical artifact: {target / filename}")
    with tempfile.TemporaryDirectory(prefix=".matmul_publish_", dir=canonical) as scratch:
        prepared = Path(scratch)
        for entry in manifest["patterns"]:
            name = entry["name"]
            if not (canonical / name).exists():
                shutil.copytree(staged / name, prepared / name)
        created = []
        try:
            for entry in manifest["patterns"]:
                name = entry["name"]
                target = canonical / name
                if (prepared / name).exists():
                    (prepared / name).rename(target)
                    created.append(target)
                elif not (target / "oracle.py").exists():
                    shutil.copyfile(staged / name / "oracle.py", target / "oracle.py")
            write_json(manifest_path, manifest)
        except Exception:
            for target in created:
                shutil.rmtree(target)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "repros")
    parser.add_argument("--canonical-dir", type=Path, help="Default: <corpus-root>/canonical")
    parser.add_argument("--graph", action="append", help="Exact graph path relative to corpus root; repeatable")
    parser.add_argument("--validate-only", action="store_true", help="Validate the saved matmul manifest and its canonical artifacts")
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--fanout-limit', type=int, default=8, help='Stop ordinary hub transit; preserve protected matmul routes')
    parser.add_argument('--similarity-threshold', type=Fraction, default=Fraction(1, 2),
                        help='Remove directed WL similarity strictly above this fraction (default: 0.5)')
    parser.add_argument('--max-matmul-ops', type=int, default=8, help='Matmul-call limit applied before similarity selection')
    args = parser.parse_args(argv)
    corpus = args.corpus_root.resolve()
    canonical = (args.canonical_dir or corpus / "canonical").resolve()
    if args.validate_only:
        manifest = json.loads((canonical.parent / MANIFEST_NAME).read_text())
        print(json.dumps(validate(canonical, manifest), indent=2))
        return 0
    try:
        collection = Collection(args.radius, args.fanout_limit, args.similarity_threshold, args.max_matmul_ops)
    except ValueError as error:
        parser.error(str(error))
    paths = sorted((corpus / "models").rglob("full_graph_*.py"))
    if args.graph:
        requested = set(args.graph)
        paths = [p for p in paths if str(p.relative_to(corpus)) in requested]
        if len(paths) != len(requested):
            parser.error("Unknown saved source graph")
    if not paths:
        parser.error("No saved full FX graphs")
    for position, path in enumerate(paths, 1):
        try:
            collection.collect(path, corpus)
        except Exception:
            print(f"FAILED {path.relative_to(corpus)}", file=sys.stderr, flush=True)
            raise
        print(f"[{position}/{len(paths)}] {path.relative_to(corpus)}: "
              f"{len(collection.patterns)} structures, "
              f"{sum(len(p['configs']) for p in collection.patterns)} configs", flush=True)
    with tempfile.TemporaryDirectory(prefix="bb_matmul_export_") as scratch:
        staged = Path(scratch) / "canonical"
        manifest = collection.write(staged)
        manifest["validation"] = {**validate(staged, manifest), "status": "passed", "gpu_kernels_executed": False}
        publish(staged, canonical, manifest)
    print(json.dumps({**manifest["totals"], "validation": manifest["validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
