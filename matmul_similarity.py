"""Shape-independent feature overlap for exact directed matmul structure graphs.

This is a multiset Jaccard similarity of local structural features, not an
exact common-subgraph percentage or a graph equivalence test.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from numbers import Integral

import networkx as nx
import numpy as np


CALL_OPS = {"call_function", "call_method", "call_module"}
METRIC = {
    "name": "directed_wl_multiset_jaccard_v1",
    "rounds": [0, 1, 2],
    "centers": sorted(CALL_OPS),
    "refinement": "All nodes; full node colors and separate directed incoming/outgoing full argument-port bundles",
    "score": "sum(min(feature multiplicities)) / sum(max(feature multiplicities))",
    "boundary": "Input roles and containers participate in refinement; boundary nodes are not feature centers",
    "settings": "Concrete input shape, stride, dtype, device, generators and aliases are excluded",
    "interpretation": "Local structural feature overlap, not common FX node percentage, global isomorphism or semantic equivalence",
}
REPRESENTATIVE_ORDER = "Descending compute-call count, descending matmul count, ascending full structure SHA-256"
SETTINGS_POLICY = "Keep every own config of retained structures; discarded nonidentical structures and their settings are not transplanted"


def _label(value) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def structural_features(graph: nx.DiGraph, rounds: int = 2) -> Counter:
    """Refine every node; count one feature per compute node at each round."""
    if not graph.is_directed() or graph.is_multigraph():
        raise TypeError("expected a simple directed graph")
    if type(rounds) is not int or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")
    calls, labels = [], {}
    for node, data in graph.nodes(data=True):
        color = data["color"]
        if not isinstance(color, str):
            raise TypeError("node colors must be strings")
        if json.loads(color)[0] in CALL_OPS:
            calls.append(node)
        labels[node] = _label(["color", color])
    if any(not isinstance(data["ports"], str) for _, _, data in graph.edges(data=True)):
        raise TypeError("edge ports must be strings")
    features = Counter((0, labels[node]) for node in calls)
    for round_index in range(1, rounds + 1):
        updated = {}
        for node in graph:
            incoming = sorted((data["ports"], labels[source]) for source, data in graph.pred[node].items())
            outgoing = sorted((data["ports"], labels[target]) for target, data in graph.succ[node].items())
            updated[node] = _label([labels[node], incoming, outgoing])
        labels = updated
        features.update((round_index, labels[node]) for node in calls)
    return features


def pair_components(first: Counter, second: Counter) -> tuple[int, int, dict[int, int]]:
    """Return multiset intersection, union and intersections grouped by round."""
    per_round = {round_index: 0 for round_index, _ in first.keys() | second.keys()}
    for feature, count in first.items():
        per_round[feature[0]] += min(int(count), int(second.get(feature, 0)))
    intersection = sum(per_round.values())
    union = sum(int(count) for count in first.values()) + sum(int(count) for count in second.values()) - intersection
    return intersection, union, dict(sorted(per_round.items()))


def intersection_matrix(features: list[Counter]) -> np.ndarray:
    """Compute all exact multiset-min intersections as a uint32 matrix."""
    postings = defaultdict(list)
    maximum = np.iinfo(np.uint32).max
    for index, counter in enumerate(features):
        if any(not isinstance(count, Integral) or isinstance(count, bool) or count < 0
               for count in counter.values()):
            raise ValueError("feature multiplicities must be nonnegative integers")
        if sum(int(count) for count in counter.values()) > maximum:
            raise OverflowError("feature mass exceeds uint32 capacity")
        for feature, count in counter.items():
            if count:
                postings[feature].append((index, count))
    result = np.zeros((len(features), len(features)), dtype=np.uint32)
    for posting in postings.values():
        if len(posting) == 1:
            index, count = posting[0]
            result[index, index] += count
            continue
        indices = np.array([index for index, _ in posting], dtype=np.intp)
        counts = np.array([count for _, count in posting], dtype=np.uint32)
        # Bound temporary dense blocks for labels shared by many graphs.
        for start in range(0, len(posting), 256):
            stop = start + 256
            result[np.ix_(indices[start:stop], indices)] += np.minimum(
                counts[start:stop, None], counts[None, :])
    return result


def select_representatives(intersections: np.ndarray, masses, order, threshold: Fraction) -> dict:
    """Greedily keep graphs, assigning removals to their closest retained graph.

    Removal requires similarity strictly greater than the threshold.
    Admission and best-witness comparisons use exact integer arithmetic.
    Ties keep the representative encountered earlier in the supplied order.
    """
    if not isinstance(threshold, Fraction) or not 0 < threshold <= 1:
        raise ValueError("threshold must be a Fraction in (0, 1]")
    if any(not isinstance(mass, Integral) or isinstance(mass, bool) or mass <= 0 for mass in masses):
        raise ValueError("feature masses must be positive integers")
    masses = [int(mass) for mass in masses]
    order = list(order)
    if len(order) != len(masses) or set(order) != set(range(len(masses))):
        raise ValueError("order must be a permutation of all graph indices")
    if intersections.shape != (len(masses), len(masses)) or intersections.dtype.kind not in "iu":
        raise ValueError("intersections must be a square integer matrix matching masses")
    if any(int(intersections[i, i]) != mass for i, mass in enumerate(masses)):
        raise ValueError("intersection diagonal must equal feature masses")
    kept, removed = [], []
    numerator, denominator = threshold.numerator, threshold.denominator
    for index in order:
        index = int(index)
        best = None
        for representative in kept:
            upper_bound_difference = min(masses[index], masses[representative]) * denominator - max(
                masses[index], masses[representative]) * numerator
            if upper_bound_difference <= 0:
                continue
            intersection = int(intersections[index, representative])
            if not 0 <= intersection <= min(masses[index], masses[representative]):
                raise ValueError("pair intersection exceeds its feature masses")
            union = masses[index] + masses[representative] - intersection
            threshold_difference = intersection * denominator - union * numerator
            if threshold_difference <= 0:
                continue
            if best is None or intersection * best["union"] > best["intersection"] * union:
                best = {"index": index, "representative": representative,
                        "intersection": intersection, "union": union}
        if best is None:
            kept.append(index)
        else:
            removed.append(best)
    return {"kept": kept, "removed": removed}
