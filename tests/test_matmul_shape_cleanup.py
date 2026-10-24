import copy
import json
import textwrap

import pytest
import torch

from matmul_capture import (GraphContext, RegionEncoding, dumps, input_kind, literal,
                            trace_region)
from matmul_dedup import ExactGraphIndex, graph_data
from matmul_shape_cleanup import (coalesced_layout, coalesce_module, replay_settings,
                                  replay_structure)


class ValueInterpreter(torch.fx.Interpreter):
    def run_node(self, node):
        value = super().run_node(node)
        node.meta['val'] = value
        return value


def region(shapes):
    graph = torch.fx.Graph()
    left, right = graph.placeholder('left'), graph.placeholder('right')
    result = graph.call_function(torch.ops.aten.mm.default, (left, right))
    views = [graph.call_function(torch.ops.aten.view.default, (result, shape)) for shape in shapes]
    graph.output(tuple(views))
    source = torch.fx.GraphModule(torch.nn.Module(), graph)
    ValueInterpreter(source).run(torch.randn(2, 2), torch.randn(2, 2))
    context = GraphContext(source)
    encoding = RegionEncoding(context, tuple(range(len(context.nodes))))
    settings = {'inputs': copy.deepcopy(encoding.inputs), 'aliases': copy.deepcopy(encoding.aliases)}
    return encoding, {'input_settings': settings}


def test_all_points_must_agree_and_literal_types_remain_distinct():
    first, first_point = region(([2, 2], [2, 2], (2, 2), [2, 2]))
    second, second_point = region(([1, 4], [1, 4], (1, 4), [4, 1]))
    if graph_data(first.graph) != graph_data(second.graph):
        raise AssertionError('Shape settings unexpectedly changed the logical structure')
    layout = coalesced_layout(first.graph, [first_point, second_point])
    assert len(layout['logical_input_nodes']) == 6
    assert len(layout['physical_input_nodes']) == 5
    groups = [group['logical_nodes'] for group in layout['shape_groups']]
    assert sorted(map(len, groups)) == [1, 1, 2]
    assert layout['observed_input_configs'] == 2


def test_equal_tensor_scalar_and_nonlifted_shape_inputs_are_never_coalesced():
    encoding, point = region(([2, 2], [2, 2]))
    core = encoding.graph.copy()
    for role, spec in [('input', {'kind': 'scalar', 'value': literal(7)}),
                       ('input', {'kind': 'scalar', 'value': literal(7)}),
                       ('input', {'kind': 'shape', 'value': literal([2, 2])})]:
        node = len(core)
        core.add_node(node, color=dumps([role, spec['kind']]))
        point['input_settings']['inputs'].append({'node': node, 'spec': spec})
    layout = coalesced_layout(core, [point])
    assert len(layout['logical_input_nodes']) - len(layout['physical_input_nodes']) == 1
    mapping = {entry['logical']: entry['physical'] for entry in layout['logical_to_physical']}
    for entry in point['input_settings']['inputs']:
        if json.loads(core.nodes[entry['node']]['color'])[0] != 'shape_parameter':
            assert mapping[entry['node']] == entry['node']


def test_replay_graph_preserves_all_ports_and_logical_artifacts():
    encoding, point = region(([2, 2], [2, 2], [2, 2]))
    before_core, before_point = graph_data(encoding.graph), copy.deepcopy(point)
    physical, layout = replay_structure(encoding.graph, coalesced_layout(encoding.graph, [point]))
    assert graph_data(encoding.graph) == before_core and point == before_point
    assert sorted(physical) == list(range(len(physical)))
    calls = list(range(len(encoding.calls)))
    assert [physical.nodes[node] for node in calls] == [encoding.graph.nodes[node] for node in calls]
    original_ports = sum(len(json.loads(data['ports'])) for _, _, data in encoding.graph.edges(data=True))
    actual_ports = sum(len(json.loads(data['ports'])) for _, _, data in physical.edges(data=True))
    assert actual_ports == original_ports
    again, again_layout = replay_structure(encoding.graph, layout)
    assert graph_data(again) == graph_data(physical) and again_layout == layout


def test_size_and_stride_ports_into_one_consumer_are_both_retained():
    graph = torch.fx.Graph()
    left, right = graph.placeholder('left'), graph.placeholder('right')
    product = graph.call_function(torch.ops.aten.mm.default, (left, right))
    result = graph.call_function(torch.ops.aten.as_strided.default, (product, [1, 1], [1, 1], 0))
    graph.output(result)
    source = torch.fx.GraphModule(torch.nn.Module(), graph)
    ValueInterpreter(source).run(torch.randn(2, 2), torch.randn(2, 2))
    context = GraphContext(source)
    encoding = RegionEncoding(context, tuple(range(len(context.nodes))))
    point = {'input_settings': {'inputs': encoding.inputs, 'aliases': encoding.aliases}}
    physical, layout = replay_structure(encoding.graph, coalesced_layout(encoding.graph, [point]))
    bundled = [json.loads(data['ports']) for _, _, data in physical.edges(data=True)
               if len(json.loads(data['ports'])) == 2]
    assert bundled == [[['argument', 'size'], ['argument', 'stride']]]


def test_replay_settings_preserve_tensor_specs_alias_paths_and_metadata():
    encoding, point = region(([2, 2], [2, 2]))
    settings = point['input_settings']
    tensor_nodes = [entry['node'] for entry in settings['inputs'] if entry['spec']['kind'] == 'tensor']
    for entry in settings['inputs']:
        if entry['node'] in tensor_nodes:
            entry['spec'].update(device='cuda:3', stride=[1, 2], storage_offset=4,
                                 requires_grad=True, gen={'kind': 'randn'}, observed={'n_unique': 4})
    late = len(encoding.graph)
    compound = {'kind': 'tuple', 'items': [{'kind': 'scalar', 'value': literal(None)},
                                          copy.deepcopy(settings['inputs'][0]['spec'])]}
    encoding.graph.add_node(late, color=dumps(['input', input_kind(compound)]))
    settings['inputs'].append({'node': late, 'spec': compound})
    settings['aliases'] = [{'nbytes': 96, 'size_source': 'retained',
                           'members': [{'node': node, 'path': []} for node in tensor_nodes]
                                      + [{'node': late, 'path': [1]}]}]
    settings['provenance'] = {'captured': True}
    before = copy.deepcopy(settings)
    _, layout = replay_structure(encoding.graph, coalesced_layout(encoding.graph, [point]))
    replay = replay_settings(settings, layout)
    assert settings == before and replay['provenance'] == settings['provenance']
    original_tensors = [entry['spec'] for entry in settings['inputs'] if entry['spec']['kind'] == 'tensor']
    physical_tensors = [entry['spec'] for entry in replay['inputs'] if entry['spec']['kind'] == 'tensor']
    assert physical_tensors == original_tensors
    assert replay['inputs'][-1]['spec'] == compound
    expected_aliases = copy.deepcopy(settings['aliases'])
    expected_aliases[0]['members'][-1]['node'] -= 1
    assert replay['aliases'] == expected_aliases


def test_future_point_with_broken_shape_equality_is_rejected():
    encoding, point = region(([2, 2], [2, 2]))
    _, layout = replay_structure(encoding.graph, coalesced_layout(encoding.graph, [point]))
    future = copy.deepcopy(point['input_settings'])
    shape_inputs = [entry for entry in future['inputs'] if entry['spec']['kind'] == 'shape']
    shape_inputs[-1]['spec']['value'] = literal([1, 4])
    with pytest.raises(ValueError, match='equality does not hold'):
        replay_settings(future, layout)


def test_coalesced_module_executes_with_no_added_or_reordered_operations():
    encoding, point = region(([2, 2], [2, 2], [2, 2]))
    _, other_point = region(([1, 4], [1, 4], [1, 4]))
    logical = encoding.build_module()
    physical_core, layout = replay_structure(encoding.graph, coalesced_layout(encoding.graph, [point, other_point]))
    physical_settings = replay_settings(point['input_settings'], layout)
    module = coalesce_module(logical, layout)
    old_calls = [(node.op, node.target, node.name) for node in logical.graph.nodes if node.op != 'placeholder']
    new_calls = [(node.op, node.target, node.name) for node in module.graph.nodes if node.op != 'placeholder']
    assert new_calls == old_calls
    logical_inputs = [node.meta['val'] for node in logical.graph.nodes if node.op == 'placeholder']
    by_logical = dict(zip(layout['logical_input_nodes'], logical_inputs))
    physical_inputs = [by_logical[entry['representative']] for entry in layout['physical_ids']]
    expected, actual = logical(*logical_inputs), module(*physical_inputs)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)
    code = module.graph.python_code('self')
    namespace = dict(code.globals)
    exec('class Repro(torch.nn.Module):\n' + textwrap.indent(code.src, '    '), namespace)
    loaded = namespace['Repro']()
    traced = trace_region(loaded)
    ValueInterpreter(traced).run(*physical_inputs)
    placeholders = [node for node in traced.graph.nodes if node.op == 'placeholder']
    metadata = {'boundary_specs': {}, 'input_roles': {}}
    for node, entry in zip(placeholders, physical_settings['inputs']):
        metadata['boundary_specs'][node.name] = entry['spec']
        metadata['input_roles'][node.name] = json.loads(physical_core.nodes[entry['node']]['color'])[0]
    context = GraphContext(traced, metadata)
    restored = RegionEncoding(context, tuple(range(len(context.nodes))))
    index = ExactGraphIndex()
    index.add(physical_core)
    assert not index.add(restored.graph)[2]
    compatible = replay_settings(other_point['input_settings'], layout)
    compatible_inputs = []
    by_physical = dict(zip(layout['physical_input_nodes'], physical_inputs))
    for entry in compatible['inputs']:
        compatible_inputs.append([1, 4] if entry['spec']['kind'] == 'shape' else by_physical[entry['node']])
    assert all(value.shape == (1, 4) for value in loaded(*compatible_inputs))


def test_empty_point_family_and_missing_settings_are_rejected():
    encoding, point = region(([2, 2], [2, 2]))
    with pytest.raises(ValueError, match='every retained captured point'):
        coalesced_layout(encoding.graph, [])
    other = copy.deepcopy(point)
    other['input_settings']['inputs'].pop()
    with pytest.raises(ValueError, match='logical input nodes'):
        coalesced_layout(encoding.graph, [point, other])
