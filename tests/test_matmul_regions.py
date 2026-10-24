import json
import operator

import pytest
import torch
import torch._inductor.inductor_prims

from matmul_capture import (GraphContext, MatmulTraversal, RegionEncoding,
                            is_effectful_node, prune_matmul_outputs, trace_region)
from matmul_dedup import ExactGraphIndex


def source_graph():
    graph = torch.fx.Graph()
    return graph, graph.placeholder('activation'), graph.placeholder('weight')


def context_with_values(graph, *, scalar_nodes=()):
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    for node in gm.graph.nodes:
        if node.op != 'output':
            node.meta['val'] = torch.empty(()) if node in scalar_nodes else torch.empty(4, 4)
    return GraphContext(gm)


def selected_nodes(context, anchor, radius=8):
    policy = MatmulTraversal(context, 8)
    cuts = dict(context.neighborhoods(radius=radius, traversal=policy))
    return {context.nodes[position] for position in cuts[anchor]}, policy


def test_shared_constant_cannot_connect_unrelated_mm_consumers():
    graph, activation, weight = source_graph()
    zero = graph.call_function(torch.ops.aten.full.default, ([], 0.0))
    mask = graph.placeholder('mask')
    mm = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    result = graph.call_function(torch.ops.aten.where.self, (mask, mm, zero))
    unrelated = []
    for index in range(8):
        value = graph.placeholder(f'other_{index}')
        where = graph.call_function(torch.ops.aten.where.self, (mask, value, zero))
        unrelated.append(graph.call_function(torch.ops.aten.mm.default, (where, weight)))
    graph.output((result, *unrelated))
    context = context_with_values(graph, scalar_nodes={zero})
    mask.meta['val'] = torch.empty(4, 4, dtype=torch.bool)
    chosen, policy = selected_nodes(context, mm)
    assert zero in chosen
    assert not set(unrelated) & chosen
    assert context.positions[zero] in policy.constants
    assert len(zero.users) == 9


def test_activation_fanout_preserves_multiple_primary_mm_routes():
    graph, activation, weight = source_graph()
    outputs = [graph.call_function(torch.ops.aten.mm.default, (activation, weight)) for _ in range(8)]
    graph.output(tuple(outputs))
    context = context_with_values(graph)
    chosen, policy = selected_nodes(context, outputs[0], radius=2)
    assert set(outputs) <= chosen
    assert context.positions[activation] in policy.forks
    assert context.positions[weight] not in policy.forks


def test_weight_fanout_alone_does_not_connect_mm_anchors():
    graph, activation, weight = source_graph()
    outputs = [graph.call_function(torch.ops.aten.mm.default, (activation, weight))]
    for index in range(7):
        other = graph.placeholder(f'other_{index}')
        outputs.append(graph.call_function(torch.ops.aten.mm.default, (other, weight)))
    graph.output(tuple(outputs))
    context = context_with_values(graph)
    chosen, _ = selected_nodes(context, outputs[0], radius=2)
    assert not set(outputs[1:]) & chosen


def test_vertical_path_survives_fanout_and_unrelated_branches_stop():
    graph, activation, weight = source_graph()
    first = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    middle = graph.call_function(torch.ops.aten.neg.default, (first,))
    second = graph.call_function(torch.ops.aten.mm.default, (middle, weight))
    branches = [graph.call_function(torch.ops.aten.sin.default, (middle,)) for _ in range(7)]
    graph.output((second, *branches))
    context = context_with_values(graph)
    chosen, policy = selected_nodes(context, first)
    assert second in chosen
    assert not set(branches) & chosen
    assert (context.positions[middle], context.positions[second]) in policy.vertical_edges


def test_horizontal_result_join_survives_consumer_fanout():
    graph, activation, weight = source_graph()
    other = graph.placeholder('other')
    first = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    second = graph.call_function(torch.ops.aten.mm.default, (other, weight))
    joined = graph.call_function(torch.ops.aten.add.Tensor, (first, second))
    branches = [graph.call_function(torch.ops.aten.sin.default, (joined,)) for _ in range(8)]
    graph.output(tuple(branches))
    context = context_with_values(graph)
    chosen, policy = selected_nodes(context, first, radius=2)
    assert second in chosen and joined in chosen
    assert context.positions[joined] in policy.joins
    assert not set(branches) & chosen


def test_data_dependent_scalar_reduction_remains_a_vertical_route():
    graph, activation, weight = source_graph()
    first = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    reduction = graph.call_function(torch.ops.aten.sum.default, (first,))
    product = graph.call_function(torch.ops.aten.mul.Tensor, (activation, reduction))
    second = graph.call_function(torch.ops.aten.mm.default, (product, weight))
    branches = [graph.call_function(torch.ops.aten.neg.default, (reduction,)) for _ in range(7)]
    graph.output((second, *branches))
    context = context_with_values(graph, scalar_nodes={reduction})
    chosen, policy = selected_nodes(context, first)
    assert second in chosen and reduction in chosen
    assert context.positions[reduction] not in policy.constants


def test_shape_argument_cannot_connect_a_value_only_mm_region():
    graph, activation, weight = source_graph()
    first = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    size = graph.call_function(torch.ops.aten.sym_size.int, (first, 0))
    other = graph.placeholder('other')
    view = graph.call_function(torch.ops.aten.view.default, (other, [size, 4]))
    second_weight = graph.placeholder('second_weight')
    second = graph.call_function(torch.ops.aten.mm.default, (view, second_weight))
    graph.output((first, second))
    context = context_with_values(graph)
    size.meta['val'] = 4
    chosen, _ = selected_nodes(context, first)
    assert second not in chosen


def test_output_pruning_keeps_live_producers_without_returning_them():
    graph, activation, weight = source_graph()
    producer = graph.call_function(torch.ops.aten.neg.default, (activation,))
    mm = graph.call_function(torch.ops.aten.mm.default, (producer, weight))
    result = graph.call_function(torch.ops.aten.neg.default, (mm,))
    unused = graph.call_function(torch.ops.aten.sin.default, (producer,))
    graph.output((producer, result, unused))
    context = context_with_values(graph)
    selected, outputs, stats = prune_matmul_outputs(context, tuple(range(len(context.nodes))))
    assert outputs == {result}
    assert context.positions[producer] in selected and context.positions[unused] not in selected
    assert stats['removed_outputs'] == 2 and stats['removed_calls'] == 1
    encoding = RegionEncoding(context, selected, outputs)
    assert encoding.escape_ids == [encoding.call_ids[result]]
    assert not json.loads(encoding.graph.nodes[encoding.call_ids[producer]]['color'])[3]
    traced = trace_region(encoding.build_module())
    for node in traced.graph.nodes:
        if node.op != 'output':
            node.meta['val'] = torch.empty(4, 4)
    restored_context = GraphContext(traced)
    returned = set(next(node for node in traced.graph.nodes if node.op == 'output').all_input_nodes)
    restored = RegionEncoding(restored_context, tuple(range(len(restored_context.nodes))), returned)
    index = ExactGraphIndex()
    index.add(encoding.graph)
    assert not index.add(restored.graph)[2]


def test_effect_calls_require_the_traversal_barrier_before_pruning_or_encoding():
    graph, activation, weight = source_graph()
    mm = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    destination = graph.placeholder('destination')
    copied = graph.call_function(torch.ops.aten.copy_.default, (destination, activation))
    graph.output((mm, copied))
    context = context_with_values(graph)
    all_nodes = tuple(range(len(context.nodes)))
    with pytest.raises(ValueError, match='Effect calls must be excluded'):
        prune_matmul_outputs(context, all_nodes)
    with pytest.raises(ValueError, match='Effect calls must be excluded'):
        RegionEncoding(context, all_nodes)
    selected = dict(context.neighborhoods())[mm]
    retained, outputs, stats = prune_matmul_outputs(context, selected)
    encoding = RegionEncoding(context, retained, outputs)
    assert outputs == {mm} and not encoding.effects
    assert stats['required_effects'] == 0
    assert copied not in encoding.calls


def test_protected_paths_do_not_extend_the_radius():
    graph, activation, weight = source_graph()
    first = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    value = first
    for _ in range(9):
        value = graph.call_function(torch.ops.aten.neg.default, (value,))
    second_weight = graph.placeholder('second_weight')
    second = graph.call_function(torch.ops.aten.mm.default, (value, second_weight))
    graph.output(second)
    context = context_with_values(graph)
    chosen, _ = selected_nodes(context, first)
    assert second not in chosen


def test_effect_barrier_excludes_rng_and_its_shared_mm_connection():
    graph, _, weight = source_graph()
    other_weight = graph.placeholder('other_weight')
    random = graph.call_function(torch.ops.aten.rand.default, ([4, 4],))
    first = graph.call_function(torch.ops.aten.mm.default, (random, weight))
    second = graph.call_function(torch.ops.aten.mm.default, (random, other_weight))
    graph.output((first, second))
    context = context_with_values(graph)
    chosen, policy = selected_nodes(context, first)
    assert is_effectful_node(random) and not is_effectful_node(weight)
    assert random not in chosen and second not in chosen
    assert context.positions[random] in policy.excluded
    selected, outputs, stats = prune_matmul_outputs(context, tuple(context.positions[n] for n in chosen))
    encoding = RegionEncoding(context, selected, outputs)
    assert not encoding.effects and stats['required_effects'] == 0
    assert set(encoding.input_sources.values()) == {random.name, weight.name}
    assert all(node.users for node in encoding.build_module().graph.nodes if node.op == 'placeholder')


@pytest.mark.parametrize('mode', ['rand', 'randn', 'randint'])
def test_compiler_rng_consumes_eager_state_and_cannot_connect_mm_anchors(mode):
    graph, _, weight = source_graph()
    other_weight = graph.placeholder('other_weight')
    seeds = graph.placeholder('seeds')
    lookup = graph.call_function(torch.ops.prims.inductor_lookup_seed.default, (seeds, 0))
    target = torch.ops.prims.inductor_randint.default if mode == 'randint' else torch.ops.prims.inductor_random.default
    arguments = (0, 100, [4, 4], lookup) if mode == 'randint' else ([4, 4], lookup, mode)
    random = graph.call_function(target, arguments)
    first = graph.call_function(torch.ops.aten.mm.default, (random, weight))
    second = graph.call_function(torch.ops.aten.mm.default, (random, other_weight))
    graph.output((first, second))
    seed_values = torch.tensor([42], dtype=torch.int64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        seed = torch.ops.prims.inductor_lookup_seed.default(seed_values, 0)
        before = torch.get_rng_state()
        eager_arguments = tuple(seed if value is lookup else value for value in arguments)
        random_value = target(*eager_arguments)
        after = torch.get_rng_state()
        repeated = target(*eager_arguments)
        assert not torch.equal(before, after)
        assert not torch.equal(random_value, repeated)
    context = context_with_values(graph)
    seeds.meta['val'], lookup.meta['val'], random.meta['val'] = seed_values, seed, random_value
    weight.meta['val'] = other_weight.meta['val'] = torch.eye(4, dtype=random_value.dtype)
    chosen, policy = selected_nodes(context, first)
    assert is_effectful_node(random) and not is_effectful_node(lookup)
    assert random not in chosen and second not in chosen
    assert context.positions[random] in policy.excluded
    selected, outputs, _ = prune_matmul_outputs(context, tuple(context.positions[node] for node in chosen))
    encoding = RegionEncoding(context, selected, outputs)
    assert set(encoding.input_sources.values()) == {random.name, weight.name}
    assert all(not is_effectful_node(node) for node in encoding.calls)
    inputs = [random_value if encoding.input_sources[str(entry['node'])] == random.name else weight.meta['val']
              for entry in encoding.inputs]
    torch.testing.assert_close(encoding.build_module()(*inputs)[0], random_value)


def test_rng_state_reservation_becomes_a_compound_boundary_without_generator_input():
    graph, _, weight = source_graph()
    generator = graph.placeholder('generator')
    reservation = graph.call_function(torch.ops.inductor_prims.inductor_reserve_rng_state.default, (generator, 4))
    seed = graph.call_function(operator.getitem, (reservation, 0))
    converted = graph.call_function(torch.ops.prims.convert_element_type.default, (seed, torch.float32))
    expanded = graph.call_function(torch.ops.aten.expand.default, (converted, [4, 4]))
    mm = graph.call_function(torch.ops.aten.mm.default, (expanded, weight))
    graph.output(mm)
    context = context_with_values(graph)
    generator.meta['val'] = None
    reservation.meta['val'] = tuple(torch.tensor([value], dtype=torch.int64) for value in (42, 0, 0))
    seed.meta['val'] = reservation.meta['val'][0]
    converted.meta['val'], expanded.meta['val'] = torch.ones(1), torch.ones(4, 4)
    chosen, policy = selected_nodes(context, mm)
    assert is_effectful_node(reservation) and reservation not in chosen
    assert context.positions[reservation] in policy.excluded
    selected, outputs, _ = prune_matmul_outputs(context, tuple(context.positions[node] for node in chosen))
    encoding = RegionEncoding(context, selected, outputs)
    assert reservation.name in encoding.input_sources.values()
    assert generator.name not in encoding.input_sources.values()
    compound = [entry for entry in encoding.inputs if entry['spec']['kind'] == 'tuple']
    assert len(compound) == 1 and len(compound[0]['spec']['items']) == 3
    assert all(not is_effectful_node(node) for node in encoding.calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA generator state reservation')
def test_rng_state_reservation_advances_the_supplied_cuda_generator():
    generator = torch.Generator(device='cuda')
    generator.manual_seed(1234)
    before = generator.get_offset()
    seed, offset, intra = torch.ops.inductor_prims.inductor_reserve_rng_state.default(generator, 4)
    assert generator.get_offset() == before + 4
    assert seed.shape == offset.shape == intra.shape == (1,)


def test_pure_seed_lookup_remains_in_an_eligible_mm_region():
    graph, _, weight = source_graph()
    seeds = graph.placeholder('seeds')
    lookup = graph.call_function(torch.ops.prims.inductor_lookup_seed.default, (seeds, 0))
    converted = graph.call_function(torch.ops.prims.convert_element_type.default, (lookup, torch.float32))
    expanded = graph.call_function(torch.ops.aten.expand.default, (converted, [4, 4]))
    mm = graph.call_function(torch.ops.aten.mm.default, (expanded, weight))
    graph.output(mm)
    context = context_with_values(graph)
    seeds.meta['val'] = torch.tensor([42], dtype=torch.int64)
    before = torch.get_rng_state()
    lookup.meta['val'] = torch.ops.prims.inductor_lookup_seed.default(seeds.meta['val'], 0)
    assert torch.equal(before, torch.get_rng_state())
    converted.meta['val'], expanded.meta['val'] = torch.tensor(42.0), torch.full((4, 4), 42.0)
    chosen, policy = selected_nodes(context, mm)
    assert lookup in chosen and not is_effectful_node(lookup)
    assert context.positions[lookup] not in policy.excluded


def test_effect_barrier_cannot_be_overridden_by_a_vertical_mm_route():
    graph, activation, weight = source_graph()
    other_weight = graph.placeholder('other_weight')
    destination = graph.placeholder('destination')
    first = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    copied = graph.call_function(torch.ops.aten.copy_.default, (destination, first))
    second = graph.call_function(torch.ops.aten.mm.default, (copied, other_weight))
    graph.output(second)
    context = context_with_values(graph)
    chosen, policy = selected_nodes(context, first)
    assert first in chosen and copied not in chosen and second not in chosen
    assert not any(context.positions[copied] in edge for edge in policy.vertical_edges)
    chosen_second, _ = selected_nodes(context, second)
    selected, outputs, _ = prune_matmul_outputs(context, tuple(context.positions[n] for n in chosen_second))
    encoding = RegionEncoding(context, selected, outputs)
    assert first not in chosen_second and not encoding.effects
    assert set(encoding.input_sources.values()) == {copied.name, other_weight.name}


def test_effect_barrier_removes_private_tensor_and_shape_inputs():
    graph, activation, weight = source_graph()
    private = graph.placeholder('private')
    private_shape = graph.placeholder('private_shape')
    expanded = graph.call_function(torch.ops.aten.expand.default, (private, private_shape))
    mm = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    copied = graph.call_function(torch.ops.aten.copy_.default, (expanded, mm))
    graph.output((mm, copied))
    context = context_with_values(graph)
    private_shape.meta['val'] = [4, 4]
    chosen, _ = selected_nodes(context, mm)
    selected, outputs, _ = prune_matmul_outputs(context, tuple(context.positions[n] for n in chosen))
    encoding = RegionEncoding(context, selected, outputs)
    assert outputs == {mm} and encoding.calls == [mm]
    assert set(encoding.input_sources.values()) == {activation.name, weight.name}
    assert not {private, private_shape, expanded, copied} & chosen


def test_effect_barrier_lifts_compound_dropout_results():
    graph, activation, weight = source_graph()
    dropout = graph.call_function(torch.ops.aten.native_dropout.default, (activation, 0.1, True))
    getitem = graph.call_function(operator.getitem, (dropout, 0))
    mm = graph.call_function(torch.ops.aten.mm.default, (getitem, weight))
    graph.output(mm)
    context = context_with_values(graph)
    dropout.meta['val'] = (torch.empty(4, 4), torch.empty(4, 4, dtype=torch.bool))
    getitem.meta['val'] = dropout.meta['val'][0]
    chosen, _ = selected_nodes(context, mm)
    selected, outputs, _ = prune_matmul_outputs(context, tuple(context.positions[n] for n in chosen))
    encoding = RegionEncoding(context, selected, outputs)
    compound = [entry for entry in encoding.inputs if entry['spec']['kind'] == 'tuple']
    assert dropout not in chosen and getitem in chosen
    assert len(compound) == 1 and not encoding.effects
    assert len(compound[0]['spec']['items']) == 2
    inputs = [dropout.meta['val'] if encoding.input_sources[str(entry['node'])] == dropout.name
              else torch.eye(4) for entry in encoding.inputs]
    assert encoding.build_module()(*inputs)[0].shape == (4, 4)


def test_effectful_matmul_anchor_is_explicitly_skipped():
    graph, activation, weight = source_graph()
    destination = graph.placeholder('destination')
    mutable = graph.call_function(torch.ops.aten.mm.out, (activation, weight), {'out': destination})
    pure = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    graph.output((mutable, pure))
    context = context_with_values(graph)
    policy = MatmulTraversal(context, 8)
    cuts = dict(context.neighborhoods(radius=8, traversal=policy))
    assert mutable not in cuts and pure in cuts
    assert context.positions[mutable] in policy.excluded
    assert context.positions[mutable] not in cuts[pure]


def test_effect_cut_does_not_make_source_vertical_pair_a_horizontal_join():
    graph, activation, weight = source_graph()
    other_weight = graph.placeholder('other_weight')
    destination = graph.placeholder('destination')
    first = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    copied = graph.call_function(torch.ops.aten.copy_.default, (destination, first))
    second = graph.call_function(torch.ops.aten.mm.default, (copied, other_weight))
    joined = graph.call_function(torch.ops.aten.add.Tensor, (first, second))
    graph.output(joined)
    context = context_with_values(graph)
    chosen, policy = selected_nodes(context, first)
    assert first in chosen and second in chosen and joined in chosen
    assert copied not in chosen and context.positions[joined] not in policy.joins


def test_explicit_seed_attention_backward_is_also_an_effect_barrier():
    graph, activation, weight = source_graph()
    other_weight = graph.placeholder('other_weight')
    seed = graph.placeholder('seed')
    offset = graph.placeholder('offset')
    first = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
    backward = graph.call_function(torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
                                   (first, activation, activation, activation, activation, activation,
                                    activation, seed, offset, 0.1, [True, True, True, False], False))
    gradient = graph.call_function(operator.getitem, (backward, 0))
    second = graph.call_function(torch.ops.aten.mm.default, (gradient, other_weight))
    graph.output(second)
    context = context_with_values(graph, scalar_nodes={seed, offset})
    backward.meta['val'] = (torch.empty(4, 4), torch.empty(4, 4), torch.empty(4, 4), None)
    chosen, _ = selected_nodes(context, first)
    assert is_effectful_node(backward)
    assert not {backward, gradient, second} & chosen
