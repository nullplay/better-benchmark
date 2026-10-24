"""Coalesce replay-only shape arguments across a finite captured point family.

The original structure and full settings remain authoritative for sampling.
Equal captured values do not establish a symbolic shape relationship.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict

import networkx as nx
import torch.fx as fx

from matmul_capture import CALL_OPS, add_port, dumps


def _input_entries(settings, logical_nodes):
    entries = {entry['node']: entry for entry in settings['inputs']}
    if len(entries) != len(settings['inputs']) or set(entries) != set(logical_nodes):
        raise ValueError('Input settings do not match the logical input nodes')
    return entries


def _layout_maps(layout):
    logical = layout['logical_input_nodes']
    physical = layout['physical_input_nodes']
    mapping = {entry['logical']: entry['physical'] for entry in layout['logical_to_physical']}
    representatives = {entry['physical']: entry['representative'] for entry in layout['physical_ids']}
    if (layout['schema_version'] != 1 or len(set(logical)) != len(logical)
            or len(set(physical)) != len(physical) or len(mapping) != len(logical)
            or len(layout['logical_to_physical']) != len(logical)
            or set(mapping) != set(logical) or set(mapping.values()) != set(physical)
            or len(layout['physical_ids']) != len(physical) or len(representatives) != len(physical)
            or len(set(representatives.values())) != len(physical) or set(representatives) != set(physical)):
        raise ValueError('Invalid coalesced replay input layout')
    return mapping, representatives


def coalesced_layout(core, points):
    if not points:
        raise ValueError('Shape coalescing requires every retained captured point')
    logical = sorted(entry['node'] for entry in points[0]['input_settings']['inputs'])
    settings = [_input_entries(point['input_settings'], logical) for point in points]
    groups = defaultdict(list)
    mapping = {node: node for node in logical}
    for node in logical:
        color = json.loads(core.nodes[node]['color'])
        if color[0] == 'shape_parameter':
            if any(point[node]['spec']['kind'] != 'shape' for point in settings):
                raise ValueError('A lifted shape parameter has a non-shape specification')
            groups[tuple(dumps(point[node]['spec']) for point in settings)].append(node)
    shape_groups = []
    for nodes in groups.values():
        representative = min(nodes)
        mapping.update((node, representative) for node in nodes)
        shape_groups.append({'representative': representative, 'logical_nodes': nodes,
                             'physical': representative})
    physical = sorted(set(mapping.values()))
    return {
        'schema_version': 1,
        'policy': 'typed shape specs equal across all retained captured points; no symbolic relation inferred',
        'observed_input_configs': len(points),
        'logical_input_nodes': logical, 'physical_input_nodes': physical,
        'logical_to_physical': [{'logical': node, 'physical': mapping[node]} for node in logical],
        'physical_ids': [{'representative': node, 'physical': node} for node in physical],
        'shape_groups': sorted(shape_groups, key=lambda group: group['representative']),
    }


def replay_structure(core, layout):
    mapping, representatives = _layout_maps(layout)
    logical_to_rep = {node: representatives[physical] for node, physical in mapping.items()}
    calls = sorted(node for node, data in core.nodes(data=True) if json.loads(data['color'])[0] in CALL_OPS)
    if calls != list(range(len(calls))) or set(core) != set(calls) | set(mapping):
        raise ValueError('Expected ordered compute IDs followed by logical input IDs')
    for node, representative in logical_to_rep.items():
        if representative not in mapping:
            raise ValueError('A replay representative is not a logical input')
        if node != representative and (json.loads(core.nodes[node]['color'])[0] != 'shape_parameter'
                                       or core.nodes[node]['color'] != core.nodes[representative]['color']):
            raise ValueError('Only identical lifted shape input roles may be coalesced')
    graph = nx.DiGraph()
    for node, data in core.nodes(data=True):
        representative = logical_to_rep.get(node, node)
        if representative == node:
            graph.add_node(node, **copy.deepcopy(data))
    for source, target, data in core.edges(data=True):
        source = logical_to_rep.get(source, source)
        target = logical_to_rep.get(target, target)
        for port in json.loads(data['ports']):
            add_port(graph, source, target, port)
    compact = {node: position for position, node in enumerate(sorted(graph))}
    graph = nx.relabel_nodes(graph, compact)
    result = copy.deepcopy(layout)
    result['physical_input_nodes'] = sorted(compact[node] for node in representatives.values())
    result['logical_to_physical'] = [{'logical': node, 'physical': compact[logical_to_rep[node]]}
                                     for node in layout['logical_input_nodes']]
    result['physical_ids'] = [{'representative': node, 'physical': compact[node]}
                              for node in sorted(representatives.values())]
    for group in result['shape_groups']:
        group['physical'] = compact[group['representative']]
    return graph, result


def replay_settings(logical_settings, layout):
    mapping, _ = _layout_maps(layout)
    logical = _input_entries(logical_settings, layout['logical_input_nodes'])
    inputs = {}
    for node in layout['logical_input_nodes']:
        physical = mapping[node]
        if physical in inputs and dumps(inputs[physical]['spec']) != dumps(logical[node]['spec']):
            raise ValueError(f'Captured shape equality does not hold for replay input {physical}')
        inputs.setdefault(physical, {**copy.deepcopy(logical[node]), 'node': physical})
    result = copy.deepcopy(logical_settings)
    result['inputs'] = [inputs[node] for node in layout['physical_input_nodes']]
    for alias in result['aliases']:
        for member in alias['members']:
            member['node'] = mapping[member['node']]
        alias['members'].sort(key=dumps)
    result['aliases'].sort(key=dumps)
    return result


def coalesce_module(traced, layout):
    mapping, representatives = _layout_maps(layout)
    placeholders = [node for node in traced.graph.nodes if node.op == 'placeholder']
    if len(placeholders) != len(layout['logical_input_nodes']):
        raise ValueError('Traced module input arity differs from the logical input layout')
    originals = dict(zip(layout['logical_input_nodes'], placeholders))
    graph, values, physical_values = fx.Graph(), {}, {}
    for physical in layout['physical_input_nodes']:
        original = originals[representatives[physical]]
        node = graph.create_node('placeholder', f'input_{physical}', original.args,
                                 original.kwargs, type_expr=original.type)
        node.meta = dict(original.meta)
        physical_values[physical] = node
    for logical, original in originals.items():
        values[original] = physical_values[mapping[logical]]
    for node in traced.graph.nodes:
        if node.op != 'placeholder':
            values[node] = graph.node_copy(node, values.__getitem__)
    graph.lint()
    return fx.GraphModule(traced, graph, class_name='Repro')
