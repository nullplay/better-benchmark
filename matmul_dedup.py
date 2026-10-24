"""Exact equivalence classes for directed FX structure and configuration graphs.

Only node ``color`` and edge ``ports`` participate in equivalence. Node IDs
are representative labels, not canonical labels. Hashes select candidates;
every merge verifies serialized equality or exact graph isomorphism.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import networkx as nx


DIGEST_POLICY = "SHA-256 of ID-ordered graph data; node IDs are not canonical labels"
MATCHER_POLICY = "Exact joint incidence color refinement, then VF2 on original directed graphs; verify original colors and ports"


def graph_data(graph: nx.DiGraph) -> dict:
    if not graph.is_directed() or graph.is_multigraph():
        raise TypeError("expected a simple directed graph")
    if any(type(node) is not int for node in graph):
        raise TypeError("graph node IDs must be integers")
    nodes = []
    for node in sorted(graph):
        color = graph.nodes[node]["color"]
        if not isinstance(color, str):
            raise TypeError("graph node colors must be strings")
        nodes.append({"id": node, "color": color})
    edges = []
    for source, target in sorted(graph.edges):
        ports = graph.edges[source, target]["ports"]
        if not isinstance(ports, str):
            raise TypeError("graph edge ports must be strings")
        edges.append({"source": source, "target": target, "ports": ports})
    return {"nodes": nodes, "edges": edges}


def graph_from_data(data: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in data["nodes"]:
        node_id = node["id"]
        if node_id in graph:
            raise ValueError(f"duplicate graph node ID: {node_id}")
        graph.add_node(node_id, color=node["color"])
    for edge in data["edges"]:
        source, target = edge["source"], edge["target"]
        if source not in graph or target not in graph:
            raise ValueError("graph edge references an absent node")
        if graph.has_edge(source, target):
            raise ValueError(f"duplicate directed graph edge: {(source, target)}")
        graph.add_edge(source, target, ports=edge["ports"])
    graph_data(graph)
    return graph


def _serialize(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def graph_digest(graph: nx.DiGraph) -> str:
    return hashlib.sha256(_serialize(graph_data(graph)).encode()).hexdigest()


def _fingerprint(graph: nx.DiGraph) -> tuple:
    colors = tuple(sorted(Counter(data["color"] for _, data in graph.nodes(data=True)).items()))
    ports = tuple(sorted(Counter(data["ports"] for _, _, data in graph.edges(data=True)).items()))
    refined = nx.weisfeiler_lehman_graph_hash(
        graph, node_attr="color", edge_attr="ports", iterations=4, digest_size=32)
    return colors, ports, refined


def _incidence_graph(graph: nx.DiGraph) -> nx.DiGraph:
    result = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        result.add_node(("node", node), color=("node", data["color"]))
    for source, target, data in graph.edges(data=True):
        edge = ("edge", source, target)
        result.add_node(edge, color=("edge", data["ports"]))
        result.add_edge(("node", source), edge)
        result.add_edge(edge, ("node", target))
    return result


def _refine_colors(first: nx.DiGraph, second: nx.DiGraph) -> bool:
    graphs = (first, second)
    palette = {color: i for i, color in enumerate(sorted({data["color"] for graph in graphs
                                                         for _, data in graph.nodes(data=True)}))}
    colors = [{node: palette[data["color"]] for node, data in graph.nodes(data=True)} for graph in graphs]
    previous = len(palette)
    while True:
        signatures = []
        for graph, labels in zip(graphs, colors):
            signatures.append({node: (labels[node],
                                     tuple(sorted(Counter(labels[v] for v in graph.pred[node]).items())),
                                     tuple(sorted(Counter(labels[v] for v in graph.succ[node]).items())))
                               for node in graph})
        palette = {key: i for i, key in enumerate(sorted(set().union(*(set(s.values()) for s in signatures))))}
        colors = [{node: palette[key] for node, key in values.items()} for values in signatures]
        if Counter(colors[0].values()) != Counter(colors[1].values()):
            return False
        if len(palette) == previous:
            break
        previous = len(palette)
    for graph, labels in zip(graphs, colors):
        nx.set_node_attributes(graph, labels, "refined_color")
    return True


def _verify_mapping(graph: nx.DiGraph, representative: nx.DiGraph, mapping: dict) -> None:
    if set(mapping) != set(graph) or set(mapping.values()) != set(representative):
        raise AssertionError("isomorphism did not preserve the node sets")
    if graph.number_of_edges() != representative.number_of_edges():
        raise AssertionError("isomorphism did not preserve the edge count")
    for node, target in mapping.items():
        if graph.nodes[node]["color"] != representative.nodes[target]["color"]:
            raise AssertionError("isomorphism did not preserve a node color")
    for source, target, data in graph.edges(data=True):
        other = representative.get_edge_data(mapping[source], mapping[target])
        if other is None or data["ports"] != other["ports"]:
            raise AssertionError("isomorphism did not preserve a directed argument port")


class ExactGraphIndex:
    def __init__(self):
        self.graphs: list[nx.DiGraph] = []
        self.stats = {"add_calls": 0, "new_graphs": 0, "exact_cache_hits": 0,
                      "isomorphism_checks": 0, "isomorphism_matches": 0,
                      "isomorphism_rejections": 0, "digest_collisions": 0,
                      "incidence_builds": 0, "refinement_rejections": 0, "vf2_checks": 0}
        self._buckets: dict[tuple, list[int]] = defaultdict(list)
        self._cache: dict[str, list[tuple[str, int, dict[int, int]]]] = defaultdict(list)
        self._incidence_graphs: dict[int, nx.DiGraph] = {}

    def add(self, graph: nx.DiGraph) -> tuple[int, dict[int, int], bool]:
        self.stats["add_calls"] += 1
        data = graph_data(graph)
        serialized = _serialize(data)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        cached = self._cache[digest]
        for previous, index, mapping in cached:
            if previous == serialized:
                self.stats["exact_cache_hits"] += 1
                return index, dict(mapping), False
        if cached:
            self.stats["digest_collisions"] += 1
        fingerprint = _fingerprint(graph)
        incidence = None
        for index in self._buckets[fingerprint]:
            self.stats["isomorphism_checks"] += 1
            if incidence is None:
                incidence = _incidence_graph(graph)
                self.stats["incidence_builds"] += 1
            if index not in self._incidence_graphs:
                self._incidence_graphs[index] = _incidence_graph(self.graphs[index])
                self.stats["incidence_builds"] += 1
            if not _refine_colors(incidence, self._incidence_graphs[index]):
                self.stats["refinement_rejections"] += 1
                self.stats["isomorphism_rejections"] += 1
                continue
            self.stats["vf2_checks"] += 1
            originals = graph.copy(), self.graphs[index].copy()
            for original, refined in zip(originals, (incidence, self._incidence_graphs[index])):
                nx.set_node_attributes(original, {node: refined.nodes[("node", node)]["refined_color"]
                                                  for node in original}, "refined_color")
            matcher = nx.algorithms.isomorphism.DiGraphMatcher(
                originals[0], originals[1],
                node_match=lambda left, right: left["refined_color"] == right["refined_color"]
                and left["color"] == right["color"],
                edge_match=lambda left, right: left["ports"] == right["ports"])
            if matcher.is_isomorphic():
                self.stats["isomorphism_matches"] += 1
                mapping = dict(matcher.mapping)
                _verify_mapping(graph, self.graphs[index], mapping)
                cached.append((serialized, index, mapping))
                return index, dict(mapping), False
            self.stats["isomorphism_rejections"] += 1
        index = len(self.graphs)
        self.graphs.append(graph_from_data(data))
        if incidence is not None:
            self._incidence_graphs[index] = incidence
        self._buckets[fingerprint].append(index)
        mapping = {node: node for node in graph}
        cached.append((serialized, index, mapping))
        self.stats["new_graphs"] += 1
        return index, dict(mapping), True
