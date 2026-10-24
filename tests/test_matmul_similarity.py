import copy
import json
from collections import Counter
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from matmul_similarity import (intersection_matrix, pair_components,
                               select_representatives, structural_features)


def color(target, arguments, escaping=True, effect=False):
    return json.dumps(["call_function", target, arguments, escaping, effect], separators=(",", ":"))


def graph_fixture():
    graph = nx.DiGraph()
    graph.add_node(0, color='["input","tensor"]')
    graph.add_node(1, color='["attribute","tensor"]')
    graph.add_node(2, color=color("aten.mm.default", [["self", ["ref"]], ["mat2", ["ref"]]]))
    graph.add_node(3, color=color("aten.mul.Scalar", [["self", ["ref"]], ["other", ["int", 1]]]))
    graph.add_edge(0, 2, ports='[["argument","self"]]')
    graph.add_edge(1, 2, ports='[["argument","mat2"]]')
    graph.add_edge(2, 3, ports='[["argument","self"]]')
    return graph


def test_features_are_relabel_invariant_and_do_not_mutate_metadata():
    graph = graph_fixture()
    graph.nodes[0].update(shape=[128, 16], stride=[16, 1], dtype="torch.bfloat16")
    snapshot = copy.deepcopy(graph)
    relabeled = nx.relabel_nodes(graph, {0: 93, 1: 18, 2: 41, 3: 7})
    relabeled.nodes[93].update(shape=[1, 4096], stride=[8192, 2], dtype="torch.float32")
    first, second = structural_features(graph), structural_features(relabeled)
    assert first == second
    assert sum(first.values()) == 6
    assert {round_index for round_index, _ in first} == {0, 1, 2}
    assert all(len(label) == 64 for _, label in first)
    assert dict(graph.nodes(data=True)) == dict(snapshot.nodes(data=True))
    assert dict(((s, t), d) for s, t, d in graph.edges(data=True)) == dict(
        ((s, t), d) for s, t, d in snapshot.edges(data=True))


def test_same_operations_with_different_directed_wiring_are_distinguished():
    graph = graph_fixture()
    reversed_path = graph.copy()
    reversed_path.remove_edges_from([(0, 2), (2, 3)])
    reversed_path.add_edge(0, 3, ports='[["argument","self"]]')
    reversed_path.add_edge(3, 2, ports='[["argument","self"]]')
    first, second = structural_features(graph), structural_features(reversed_path)
    intersection, union, per_round = pair_components(first, second)
    assert per_round[0] == 2
    assert intersection < union


@pytest.mark.parametrize("kind", ["call_function", "call_method", "call_module"])
def test_all_compute_call_kinds_are_counted_and_boundary_nodes_are_not(kind):
    graph = graph_fixture()
    label = json.loads(graph.nodes[2]["color"])
    label[0] = kind
    graph.nodes[2]["color"] = json.dumps(label, separators=(",", ":"))
    graph.add_node(4, color='["shape_parameter","shape"]')
    assert sum(structural_features(graph).values()) == 6


@pytest.mark.parametrize("change", ["ports", "shared_operand", "escape", "effect", "overload", "constant", "input_role"])
def test_semantic_labels_and_complete_ports_affect_similarity(change):
    graph = graph_fixture()
    other = copy.deepcopy(graph)
    if change == "ports":
        other.edges[0, 2]["ports"], other.edges[1, 2]["ports"] = other.edges[1, 2]["ports"], other.edges[0, 2]["ports"]
    elif change == "shared_operand":
        other.remove_node(1)
        other.edges[0, 2]["ports"] = '[["argument","self"],["argument","mat2"]]'
    elif change == "input_role":
        other.nodes[1]["color"] = '["input","tensor"]'
    elif change == "constant":
        label = json.loads(other.nodes[3]["color"])
        label[2][1][1][1] = 2
        other.nodes[3]["color"] = json.dumps(label, separators=(",", ":"))
    else:
        label = json.loads(other.nodes[2]["color"])
        if change == "escape":
            label[3] = False
        elif change == "effect":
            label[4] = True
        elif change == "overload":
            label[1] = "aten.mm.out"
        other.nodes[2]["color"] = json.dumps(label, separators=(",", ":"))
    first, second = structural_features(graph), structural_features(other)
    intersection, union, _ = pair_components(first, second)
    assert intersection < union


def test_boundary_sharing_is_seen_by_refining_noncompute_nodes():
    shared = nx.DiGraph()
    shared.add_node(0, color='["input","tensor"]')
    for node in (1, 2):
        shared.add_node(node, color=color("aten.neg.default", [["self", ["ref"]]]))
        shared.add_edge(0, node, ports='[["argument","self"]]')
    separate = shared.copy()
    separate.add_node(3, color='["input","tensor"]')
    separate.remove_edge(0, 2)
    separate.add_edge(3, 2, ports='[["argument","self"]]')
    assert structural_features(shared, rounds=1) == structural_features(separate, rounds=1)
    assert structural_features(shared) != structural_features(separate)


def test_effect_order_direction_is_retained_separately_from_data_edges():
    graph = nx.DiGraph()
    graph.add_node(0, color='["input","tensor"]')
    for node, target in [(1, "aten.uniform_.default"), (2, "aten.normal_.default")]:
        graph.add_node(node, color=color(target, [["self", ["ref"]]], effect=True))
        graph.add_edge(0, node, ports='[["argument","self"]]')
    graph.add_edge(1, 2, ports='[["effect_order"]]')
    other = graph.copy()
    other.remove_edge(1, 2)
    other.add_edge(2, 1, ports='[["effect_order"]]')
    first, second = structural_features(graph), structural_features(other)
    assert pair_components(first, second)[2][0] == 2
    assert first != second


def test_repeated_patterns_retain_multiplicity_and_size_penalty():
    graph = nx.DiGraph()
    graph.add_node(0, color='["input","tensor"]')
    graph.add_node(1, color=color("aten.neg.default", [["self", ["ref"]]]))
    graph.add_edge(0, 1, ports='[["argument","self"]]')
    repeated = nx.disjoint_union(graph, graph)
    first, second = structural_features(graph), structural_features(repeated)
    assert second == Counter({key: 2 * count for key, count in first.items()})
    assert pair_components(first, second) == (3, 6, {0: 1, 1: 1, 2: 1})
    assert intersection_matrix([first, second]).tolist() == [[3, 3], [3, 6]]


def test_inverted_posting_matrix_matches_direct_multiset_minima():
    generator = np.random.default_rng(193)
    features = [Counter({(round_index, str(label)): int(generator.integers(0, 7))
                         for round_index in range(3) for label in range(8)}) for _ in range(19)]
    matrix = intersection_matrix(features)
    assert matrix.dtype == np.uint32
    for i, first in enumerate(features):
        for j, second in enumerate(features):
            assert int(matrix[i, j]) == pair_components(first, second)[0]
    assert intersection_matrix([]).shape == (0, 0)


def test_shared_posting_blocks_cover_first_boundary_and_last_rows():
    features = [Counter({(0, "shared"): i + 1, (1, str(i % 3)): 2}) for i in range(301)]
    matrix = intersection_matrix(features)
    for i in (0, 254, 255, 256, 257, 300):
        for j in range(len(features)):
            assert int(matrix[i, j]) == pair_components(features[i], features[j])[0]
            assert int(matrix[j, i]) == int(matrix[i, j])


@pytest.mark.parametrize("shared,extra,threshold", [
    (2, 2, Fraction(1, 2)), (7, 3, Fraction(7, 10)), (3, 7, Fraction(3, 10)),
])
def test_threshold_boundaries_are_strict_and_exact(shared, extra, threshold):
    features = [Counter({(0, "shared"): shared, (0, "extra"): extra}), Counter({(0, "shared"): shared})]
    matrix = intersection_matrix(features)
    masses = [sum(counter.values()) for counter in features]
    selected = select_representatives(matrix, masses, [0, 1], threshold)
    assert selected == {"kept": [0, 1], "removed": []}
    assert select_representatives(matrix, masses, [0, 1], threshold - Fraction(1, 100)) == {
        "kept": [0], "removed": [{"index": 1, "representative": 0,
                                  "intersection": shared, "union": shared + extra}]}


@pytest.mark.parametrize("values,threshold", [
    ([{"a": 3, "x": 7}, {"a": 3}], Fraction(3, 10)),
    ([{"a": 3, "x": 3}, {"a": 3, "y": 4}], Fraction(3, 10)),
    ([{"a": 9}, {"a": 9}], Fraction(1)),
])
def test_strict_threshold_retains_exact_boundary_at_upper_bound_and_actual_score(values, threshold):
    features = [Counter({(0, key): count for key, count in value.items()}) for value in values]
    matrix, masses = intersection_matrix(features), [sum(counter.values()) for counter in features]
    intersection, union, _ = pair_components(*features)
    assert Fraction(intersection, union) == threshold
    assert select_representatives(matrix, masses, [0, 1], threshold) == {
        "kept": [0, 1], "removed": []}


@pytest.mark.parametrize("shared,kept", [(49, [0, 1]), (50, [0, 1]), (51, [0])])
def test_strict_fifty_percent_filters_only_above_threshold(shared, kept):
    features = [Counter({(0, "shared"): shared, (0, "extra"): 100 - shared}),
                Counter({(0, "shared"): shared})]
    result = select_representatives(intersection_matrix(features), [100, shared], [0, 1],
                                    Fraction(1, 2))
    assert result["kept"] == kept


@pytest.mark.parametrize("last,order,representative", [
    ({"a": 4, "b": 5}, [0, 1, 2], 1),
    ({"a": 4, "b": 4}, [0, 1, 2], 0),
    ({"a": 4, "b": 4}, [1, 0, 2], 1),
])
def test_best_direct_witness_and_priority_ties(last, order, representative):
    values = [{"a": 6, "z": 4}, {"b": 6, "z": 4}, last]
    features = [Counter({(0, key): count for key, count in value.items()}) for value in values]
    selected = select_representatives(intersection_matrix(features), [sum(c.values()) for c in features],
                                     order, Fraction(13, 50))
    assert selected["kept"] == order[:2]
    assert selected["removed"][0]["representative"] == representative


@pytest.mark.parametrize("order", [[0, 1, 2], [1, 0, 2]])
def test_strict_threshold_uses_qualifying_best_witness(order):
    values = [{"a": 3, "c": 3}, {"b": 4, "c": 2}, {"a": 3, "b": 4, "c": 2}]
    features = [Counter({(0, key): count for key, count in value.items()}) for value in values]
    assert Fraction(*pair_components(features[0], features[2])[:2]) == Fraction(1, 2)
    result = select_representatives(intersection_matrix(features), [6, 6, 9], order,
                                    Fraction(1, 2))
    assert result["kept"] == order[:2]
    assert result["removed"] == [{"index": 2, "representative": 1, "intersection": 6, "union": 9}]


def test_independent_thresholds_keep_direct_witnesses_without_transitive_merges():
    base = set(range(100))
    sets = [set(range(75)) | set(range(100, 125)), base,
            (base - set(range(17))) | set(range(125, 142)),
            (base - set(range(17, 34))) | set(range(142, 159))]
    features = [Counter((0, str(token)) for token in values) for values in sets]
    matrix, masses = intersection_matrix(features), [100] * 4
    outputs = [select_representatives(matrix, masses, range(4), threshold)
               for threshold in (Fraction(1, 2), Fraction(7, 10))]
    assert outputs[0]["kept"] == [0, 2, 3]
    assert outputs[1]["kept"] == [0, 1]
    for threshold, result in zip((Fraction(1, 2), Fraction(7, 10)), outputs):
        assert sorted(result["kept"] + [r["index"] for r in result["removed"]]) == list(range(4))
        for record in result["removed"]:
            assert record["representative"] in result["kept"]
            score = Fraction(record["intersection"], record["union"])
            assert score > threshold
        for i, left in enumerate(result["kept"]):
            for right in result["kept"][i + 1:]:
                intersection, union, _ = pair_components(features[left], features[right])
                score = Fraction(intersection, union)
                assert score <= threshold


def manifest_fixture(populations):
    from matmul_dedup import graph_digest
    from scripts.extract_matmuls import SELECTION

    graphs, entries, cuts = [], [], []
    for copies, configs in populations:
        graph = nx.disjoint_union_all([graph_fixture() for _ in range(copies)])
        digest = graph_digest(graph)
        name = "mm_" + digest[:12]
        graphs.append(graph)
        entries.append({"name": name, "pattern_hash": digest[:12], "gemm_count": copies,
                        "input_configs": configs})
        for config in range(configs):
            anchor = f"anchor_{len(cuts)}"
            cuts.append({"pattern": name, "shape_hash": f"{config:08x}", "anchors": [anchor],
                         "anchor_ops": {anchor: "mm"}, "selected_positions": [0],
                         "escaping_positions": [0], "boundary_sources": {"1": "activation"},
                         "raw_selected_positions": [0, 1], "output_pruning": {"removed_calls": 1}})
    manifest = {"schema_version": 1, "collection": "matmul_effect_barrier_8hop_v1",
                "selection": copy.deepcopy(SELECTION), "patterns": entries,
                "source_graphs": [{"graph": "models/test/train/model/full_graph_000.py", "cuts": cuts,
                                   "anchors": len(cuts), "graph_sha256": "a" * 64, "sidecar_sha256": "b" * 64,
                                   "traversal": {"debug": True}, "output_pruning": {"removed_calls": len(cuts)}}],
                "totals": {"source_graphs": 1, "graphs_with_anchors": 1, "anchors": len(cuts),
                           "source_cuts": len(cuts), "structures": len(entries),
                           "input_configs": sum(configs for _, configs in populations)}}
    return graphs, manifest


def test_sampling_keeps_exact_fifty_percent_boundary_and_every_retained_configuration():
    from scripts.extract_matmuls import filter_manifest, sample_manifest

    graphs, original = manifest_fixture([(8, 1), (4, 2)])
    before = copy.deepcopy(original)
    eligible = filter_manifest(original, {entry["name"] for entry in original["patterns"]}, max_matmul_ops=8)
    sampled = sample_manifest(eligible, graphs)
    assert sampled["totals"]["structures"] == 2 and sampled["totals"]["input_configs"] == 3
    assert original == before
    assert sampled["patterns"] == original["patterns"]
    assert sampled["anchor_counts"] == {"mm": 3}
    assert sampled["matmul_limit"]["source_totals"] == original["totals"]
    assert sampled["matmul_limit"]["removed_structures"] == 0
    assert sampled["matmul_limit"]["removed_input_configs"] == 0
    assert sampled["matmul_limit"]["stage"] == "before_similarity"
    source = sampled["source_graphs"][0]
    assert "traversal" not in source and "output_pruning" not in source
    assert {cut["shape_hash"] for cut in source["cuts"]} == {"00000000", "00000001"}
    assert all(cut["boundary_sources"] == {"1": "activation"} for cut in source["cuts"])
    assert all("raw_selected_positions" not in cut and "output_pruning" not in cut for cut in source["cuts"])
    assert sampled["sampling"]["comparison"] == "gt"
    assert sampled["sampling"]["threshold_percent"] == 50
    assert sampled["sampling"]["source_totals"] == eligible["totals"]
    assert sampled["selection"]["max_matmul_ops"] == 8


def test_oversized_graph_cannot_discard_an_eligible_graph_during_similarity():
    from scripts.extract_matmuls import filter_manifest, sample_manifest

    graphs, original = manifest_fixture([(9, 1), (8, 2)])
    eligible = filter_manifest(original, {original["patterns"][1]["name"]}, max_matmul_ops=8)
    sampled = sample_manifest(eligible, graphs[1:])
    assert sampled["patterns"] == original["patterns"][1:]
    assert sampled["totals"]["input_configs"] == 2
    assert sampled["anchor_counts"] == {"mm": 2}
    assert sampled["sampling"]["source_totals"] == eligible["totals"]
    assert sampled["matmul_limit"]["source_totals"] == original["totals"]
    assert sampled["matmul_limit"]["removed_structures"] == 1
    assert sampled["matmul_limit"]["removed_input_configs"] == 1


def test_limit_rejects_oversized_selected_graphs_before_similarity():
    from scripts.extract_matmuls import filter_manifest

    _, manifest = manifest_fixture([(9, 1), (8, 1)])
    with pytest.raises(ValueError, match="exceeds the matmul limit"):
        filter_manifest(manifest, {manifest["patterns"][0]["name"]}, max_matmul_ops=8)
    eligible = filter_manifest(manifest, {manifest["patterns"][1]["name"]}, max_matmul_ops=8)
    assert "sampling" not in eligible and eligible["matmul_limit"]["stage"] == "before_similarity"


def test_invalid_matrix_counts_are_rejected_before_uint32_overflow():
    with pytest.raises(ValueError, match="multiplicities"):
        intersection_matrix([Counter({(0, "a"): -1})])
    with pytest.raises(OverflowError, match="uint32"):
        intersection_matrix([Counter({(0, "a"): 2 ** 32})])
    counts = Counter({(0, "a"): np.uint32(2 ** 31), (1, "b"): np.uint32(2 ** 31)})
    with pytest.raises(OverflowError, match="uint32"):
        intersection_matrix([counts])
    assert pair_components(counts, counts) == (2 ** 32, 2 ** 32, {0: 2 ** 31, 1: 2 ** 31})
