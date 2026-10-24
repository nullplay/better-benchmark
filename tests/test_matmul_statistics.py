import csv
import hashlib
import json

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from input_codec import compact_from_spec
from scripts import report_matmul_corpus as statistics_module
from scripts.report_matmul_corpus import (RecordingInterpreter, analyze_flow,
                                         flow_from_fx, report_corpus, source_motifs)


def test_source_ancestry_rejects_horizontal_pair_hidden_by_effect_boundary():
    graph = torch.fx.Graph()
    activation, first_weight, second_weight, destination = [graph.placeholder(name)
        for name in ("activation", "first_weight", "second_weight", "destination")]
    first = graph.call_function(torch.ops.aten.mm.default, (activation, first_weight))
    effect = graph.call_function(torch.ops.aten.copy_.default, (destination, first))
    second = graph.call_function(torch.ops.aten.mm.default, (effect, second_weight))
    combined = graph.call_function(torch.ops.aten.add.Tensor, (first, second))
    graph.output((first, second, combined))
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    with torch._dynamo.config.patch(fake_tensor_cache_enabled=False), FakeTensorMode(allow_fallback_kernels=False):
        inputs = [torch.empty(4, 4) for _ in range(4)]
        interpreter = RecordingInterpreter(module)
        interpreter.run(*inputs)
        flow = flow_from_fx(module, interpreter.values)
    positions = {node: index for index, node in enumerate(module.graph.nodes)}
    counts, flags, _, _ = analyze_flow(flow, [positions[node] for node in (first, second, combined)])
    assert counts["vertical_pairs"] == 0
    assert not flags["horizontal_join"]
    assert positions[first] in flow[positions[second]]["matmul_ancestors"]


def test_actual_shared_activation_and_compute_join_are_distinct_motifs():
    graph = torch.fx.Graph()
    activation, first_weight, second_weight = [graph.placeholder(name)
        for name in ("activation", "first_weight", "second_weight")]
    first = graph.call_function(torch.ops.aten.mm.default, (activation, first_weight))
    second = graph.call_function(torch.ops.aten.mm.default, (activation, second_weight))
    combined = graph.call_function(torch.ops.aten.add.Tensor, (first, second))
    graph.output((first, second, combined))
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    with FakeTensorMode(allow_fallback_kernels=False):
        interpreter = RecordingInterpreter(module)
        interpreter.run(*[torch.empty(4, 4) for _ in range(3)])
        counts, flags, _, _ = analyze_flow(flow_from_fx(module, interpreter.values))
    assert counts["direct_shared_activation_pairs"] == 1
    assert counts["shared_activation_pairs"] == 1
    assert counts["horizontal_join_pairs"] == 1
    assert counts["horizontal_join_consumers"] == 1
    assert flags["horizontal"]
    assert not flags["vertical"]


def test_literal_factory_is_not_a_data_dependent_shared_activation():
    graph = torch.fx.Graph()
    first_weight, second_weight = [graph.placeholder(name) for name in ("first_weight", "second_weight")]
    constant = graph.call_function(torch.ops.aten.full.default, ([4, 4], 0.0), {"device": torch.device("cpu")})
    first = graph.call_function(torch.ops.aten.mm.default, (constant, first_weight))
    second = graph.call_function(torch.ops.aten.mm.default, (constant, second_weight))
    graph.output((first, second))
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    with FakeTensorMode(allow_fallback_kernels=False):
        interpreter = RecordingInterpreter(module)
        interpreter.run(torch.empty(4, 4), torch.empty(4, 4))
        counts, flags, _, _ = analyze_flow(flow_from_fx(module, interpreter.values))
    assert counts["shared_activation_pairs"] == 0
    assert counts["direct_shared_activation_pairs"] == 0
    assert not flags["horizontal_join"]


def test_source_cache_rejects_stale_effect_classifier_identity(tmp_path, monkeypatch):
    graph = torch.fx.Graph()
    activation, first_weight, second_weight, destination = [graph.placeholder(name)
        for name in ("activation", "first_weight", "second_weight", "destination")]
    first = graph.call_function(torch.ops.aten.mm.default, (activation, first_weight))
    effect = graph.call_function(torch.ops.aten.copy_.default, (destination, first))
    second = graph.call_function(torch.ops.aten.mm.default, (effect, second_weight))
    combined = graph.call_function(torch.ops.aten.add.Tensor, (first, second))
    graph.output((first, second, combined))
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    with torch._dynamo.config.patch(fake_tensor_cache_enabled=False), FakeTensorMode(allow_fallback_kernels=False):
        interpreter = RecordingInterpreter(module)
        interpreter.run(*[torch.empty(4, 4) for _ in range(4)])
        for node, value in interpreter.values.items():
            node.meta["val"] = value
        records = flow_from_fx(module, interpreter.values)
    positions = {node: index for index, node in enumerate(module.graph.nodes)}
    selected = [positions[node] for node in (first, second, combined)]
    stale = json.loads(json.dumps(records))
    stale[positions[second]]["matmul_ancestors"] = []
    stale[positions[effect]]["effect"] = False
    assert analyze_flow(stale, selected)[1]["horizontal_join"]
    source_path = tmp_path / "source.py"
    source_path.write_text("fixture source\n")
    source_path.with_suffix(".meta.json").write_text("{}\n")
    source = {"graph": source_path.name, "graph_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "sidecar_sha256": hashlib.sha256(source_path.with_suffix(".meta.json").read_bytes()).hexdigest(),
              "cuts": [{"pattern": "fixture", "shape_hash": "setting", "selected_positions": selected,
                        "anchors": [first.name], "anchor_ops": {first.name: "mm"}}]}
    manifest = {"source_graphs": [source]}
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / (hashlib.sha256(source_path.name.encode()).hexdigest() + ".json")
    identity = {key: source[key] for key in ("graph_sha256", "sidecar_sha256")}
    identity["torch_version"] = str(torch.__version__)
    cache_file.write_text(json.dumps({"schema_version": 1, "identity": identity, "records": stale}))
    calls = []

    def load_fixture(path):
        calls.append(path)
        return module

    monkeypatch.setattr(statistics_module, "load_matmul_graph", load_fixture)
    observed = source_motifs(manifest, {"fixture"}, tmp_path, cache_dir)
    assert len(calls) == 1
    assert not observed["fixture", "setting"][0]["flags"]["horizontal_join"]
    saved = json.loads(cache_file.read_text())
    assert saved["identity"]["effect_classifier"] == {
        "version": 1, "rng_schemas": sorted(statistics_module.RNG_EFFECT_SCHEMAS)}
    source_motifs(manifest, {"fixture"}, tmp_path, cache_dir)
    assert len(calls) == 1
    monkeypatch.setattr(statistics_module, "RNG_EFFECT_SCHEMAS",
                        statistics_module.RNG_EFFECT_SCHEMAS | {"prims::future_rng"})
    source_motifs(manifest, {"fixture"}, tmp_path, cache_dir)
    assert len(calls) == 2
    assert "prims::future_rng" in json.loads(cache_file.read_text())["identity"]["effect_classifier"]["rng_schemas"]


def test_report_consumes_every_setting_and_preserves_shape_stride_variants(tmp_path):
    directory = tmp_path / "canonical" / "mm_fixture"
    directory.mkdir(parents=True)
    (directory / "repro.py").write_text('''import torch
from repro_harness import default_shape_config, make_inputs_from_config

_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, a, b):
        return (torch.ops.aten.mm.default(a, b),)

def make_inputs(config=None):
    return make_inputs_from_config(default_shape_config(__file__) if config is None else config)
''')
    points = []
    for index, (k, right_stride) in enumerate(((64, [1, 1]), (512, [1, 1]), (64, [1, 64]))):
        specs = [{"kind": "tensor", "shape": [4, k], "stride": [k, 1], "dtype": "float32", "device": "cpu"},
                 {"kind": "tensor", "shape": [k, 1], "stride": right_stride, "dtype": "float32", "device": "cpu"}]
        shape_hash = str(index + 1) * 8
        points.append({"shape_hash": shape_hash, "config_sha256": str(index + 1) * 64,
                       "inputs": [compact_from_spec(spec) for spec in specs], "models": {},
                       "input_settings": {"inputs": [{"node": node, "spec": spec} for node, spec in zip((1, 2), specs)], "aliases": []}})
    (directory / "shapes.json").write_text(json.dumps({"points": points}))
    (directory / "meta.json").write_text(json.dumps({"exact_structure": {"input_nodes": [1, 2], "input_roles": {"a": "input", "b": "input"}}}))
    manifest = {"patterns": [{"name": directory.name, "gemm_count": 1, "input_configs": 3}], "source_graphs": []}
    output = tmp_path / "statistics"
    report = report_corpus(directory.parent, manifest, output, tmp_path)
    assert (report["structures"], report["input_configs"]) == (1, 3)
    assert report["motifs"]["small_reduction"] == {"structures": 1, "settings": 2}
    assert report["motifs"]["rank2_mv_like"] == {"structures": 1, "settings": 3}
    assert report["source_verified_input_configs"] == 0
    assert report["tensor_dtypes"]["input"]["torch.float32"] == {"structures": 1, "settings": 3, "leaves": 6}
    details = [json.loads(line) for line in (output / "configuration_details.jsonl").read_text().splitlines()]
    assert [detail["inputs"][1]["value"]["stride"] for detail in details] == [[1, 1], [1, 1], [1, 64]]
    assert all(detail["output_tree"]["items"][0]["shape"] == [4, 1] for detail in details)
    with (output / "graph_stats.csv").open(newline="") as handle:
        graph = next(csv.DictReader(handle))
    for key, expected in (("input_shape_variants", "2"), ("input_shape_stride_variants", "3"), ("input_geometry_variants", "3")):
        assert graph[key] == expected
