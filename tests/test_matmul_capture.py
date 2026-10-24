import copy
import json
import textwrap

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode

from matmul_capture import (MATMUL_ATEN_NAMES, GraphContext, RegionEncoding, configuration_graph,
                            decode_literal, is_matmul_node, literal,
                            load_matmul_graph, native_input_spec, trace_region)
from matmul_dedup import ExactGraphIndex, graph_digest


class ValueInterpreter(torch.fx.Interpreter):
    def run_node(self, node):
        result = super().run_node(node)
        node.meta['val'] = result
        return result


class NoTensorExecution(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        raise AssertionError(f'Tracing executed a tensor operator: {function}')


def reencode_serialized(encoding):
    module = encoding.build_module()
    inputs = [node.meta['val'] for node in module.graph.nodes if node.op == 'placeholder']
    code = module.graph.python_code('self')
    namespace = dict(code.globals)
    exec('class Repro(torch.nn.Module):\n' + textwrap.indent(code.src, '    '), namespace)
    loaded = namespace['Repro']()
    with NoTensorExecution():
        traced = trace_region(loaded)
    ValueInterpreter(traced).run(*inputs)
    placeholders = [node for node in traced.graph.nodes if node.op == 'placeholder']
    metadata = {'boundary_specs': {}, 'input_roles': {}}
    for node, entry in zip(placeholders, sorted(encoding.inputs, key=lambda entry: entry['node'])):
        metadata['boundary_specs'][node.name] = entry['spec']
        metadata['input_roles'][node.name] = json.loads(encoding.graph.nodes[entry['node']]['color'])[0]
    context = GraphContext(traced, metadata)
    restored = RegionEncoding(context, tuple(range(len(context.nodes))))
    index = ExactGraphIndex()
    index.add(encoding.graph)
    assert not index.add(restored.graph)[2]
    return loaded, inputs


def test_serialized_nested_inputs_shapes_and_schema_defaults():
    class Source(torch.nn.Module):
        def forward(self, pair, bias):
            result = torch.ops.aten.addmm.default(bias, pair[0], pair[1], beta=2, alpha=0.5)
            return torch.ops.aten.view.default(result, [1, -1])

    pair = (torch.randn(2, 3), torch.randn(3, 4))
    bias = torch.randn(4)
    gm = torch.fx.symbolic_trace(Source())
    expected = ValueInterpreter(gm).run(pair, bias)
    context = GraphContext(gm)
    encoding = RegionEncoding(context, next(context.neighborhoods())[1])
    loaded, inputs = reencode_serialized(encoding)
    torch.testing.assert_close(loaded(*inputs)[0], expected)
    assert any(entry['spec']['kind'] == 'tuple' for entry in encoding.inputs)


def test_shape_literals_and_storage_settings_do_not_enter_core_identity():
    index, configs = ExactGraphIndex(), ExactGraphIndex()
    for shape, offset, rows in [([0, 1], 0, 0), ([1, 1], 1, 1), ([2, -1], 0, 2)]:
        graph = torch.fx.Graph()
        a, b = graph.placeholder('a'), graph.placeholder('b')
        result = graph.call_function(torch.ops.aten.mm.default, (a, b))
        view = graph.call_function(torch.ops.aten.view.default, (result, shape))
        graph.output(view)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        ValueInterpreter(gm).run(torch.empty(8).as_strided((rows, 3), (3, 1), offset), torch.empty(3, 1))
        context = GraphContext(gm)
        encoding = RegionEncoding(context, next(context.neighborhoods())[1])
        index.add(encoding.graph)
        configs.add(configuration_graph(encoding.graph, encoding.inputs, encoding.aliases))
        assert any(entry['spec'].get('value') == literal(shape) for entry in encoding.inputs)
    assert len(index.graphs) == 1
    assert len(configs.graphs) == 3


def test_aliased_index_add_replay_infers_bounds_without_changing_captured_identity():
    from input_codec import spec_from_compact
    from repro_harness import make_inputs_from_config
    from scripts.extract_matmuls import replay_config

    class Source(torch.nn.Module):
        def forward(self, a, b, indices, aliased_indices):
            values = torch.ops.aten.view.default(torch.ops.aten.mm.default(a, b), [-1])
            first_indices = torch.ops.aten.view.default(indices, [-1])
            second_indices = torch.ops.aten.view.default(aliased_indices, [-1])
            destination = torch.ops.aten.full.default([11], 0.0, dtype=torch.float32, device='cpu')
            first = torch.ops.aten.index_add.default(destination, 0, first_indices, values)
            return torch.ops.aten.index_add.default(first, 0, second_indices, values)

    a, b = torch.randn(2, 2), torch.randn(2, 3)
    storage = torch.arange(8, dtype=torch.int64)
    indices = storage.as_strided((2, 3), (3, 1), 1)
    aliased_indices = storage.as_strided((6,), (1,), 0)
    gm = trace_region(Source())
    ValueInterpreter(gm).run(a, b, indices, aliased_indices)
    context = GraphContext(gm)
    encoding = RegionEncoding(context, next(context.neighborhoods())[1])
    settings = {'inputs': copy.deepcopy(encoding.inputs), 'aliases': copy.deepcopy(encoding.aliases)}
    structure_hash = graph_digest(encoding.graph)
    config_hash = graph_digest(configuration_graph(encoding.graph, encoding.inputs, encoding.aliases))
    constraints = encoding.replay_constraints()
    integer_ids = {entry['node'] for entry in encoding.inputs if entry['spec'].get('dtype') == 'torch.int64'}
    assert len(integer_ids) == 2
    assert {entry['node'] for entry in constraints['overrides']} == integer_ids
    assert all(entry['gen'] == {'kind': 'index', 'low': 0, 'high': 11} for entry in constraints['overrides'])
    assert encoding.inputs == settings['inputs'] and encoding.aliases == settings['aliases']
    assert graph_digest(encoding.graph) == structure_hash
    assert graph_digest(configuration_graph(encoding.graph, encoding.inputs, encoding.aliases)) == config_hash
    replay = replay_config(settings, constraints, {entry['node']: entry['node'] for entry in encoding.inputs})
    inputs = make_inputs_from_config({'inputs': [spec_from_compact(entry) for entry in replay['inputs']],
                                     'alias_group_nbytes': replay['alias_group_nbytes']})
    by_source = {encoding.input_sources[str(entry['node'])]: value for entry, value in zip(encoding.inputs, inputs)
                 if entry['spec']['kind'] == 'tensor'}
    for entry, value in zip(encoding.inputs, inputs):
        if entry['spec']['kind'] != 'tensor':
            continue
        spec = entry['spec']
        assert list(value.shape) == spec['shape'] and list(value.stride()) == spec['stride']
        assert value.storage_offset() == spec['storage_offset'] and str(value.dtype) == spec['dtype']
    replayed_indices, replayed_alias = by_source['indices'], by_source['aliased_indices']
    assert replayed_indices.untyped_storage()._cdata == replayed_alias.untyped_storage()._cdata
    assert torch.equal(replayed_indices.reshape(-1)[:-1], replayed_alias[1:])
    assert replayed_indices.min().item() >= 0 and replayed_indices.max().item() < 11
    loaded, _ = reencode_serialized(encoding)
    expected = Source()(by_source['a'], by_source['b'], replayed_indices, replayed_alias)
    torch.testing.assert_close(loaded(*inputs)[0], expected)


def test_index_add_source_values_do_not_receive_an_index_generator():
    class Source(torch.nn.Module):
        def forward(self, values, indices, weight):
            destination = torch.ops.aten.full.default([2, 3], 0.0, dtype=torch.float32, device='cpu')
            scaled = torch.ops.aten.mul.Tensor(values, 0.125)
            combined = torch.ops.aten.index_add.default(destination, 1, indices, scaled)
            return torch.ops.aten.mm.default(combined, weight)

    gm = trace_region(Source())
    ValueInterpreter(gm).run(torch.randn(2, 3), torch.arange(3), torch.randn(3, 2))
    context = GraphContext(gm)
    encoding = RegionEncoding(context, next(context.neighborhoods())[1])
    constraints = encoding.replay_constraints()
    index_node = next(entry['node'] for entry in encoding.inputs if entry['spec'].get('dtype') == 'torch.int64')
    assert len(constraints['overrides']) == 1
    assert constraints['overrides'][0]['node'] == index_node
    assert constraints['overrides'][0]['gen'] == {'kind': 'index', 'low': 0, 'high': 3}


def test_output_is_terminal_and_shared_inputs_are_transit_nodes():
    graph = torch.fx.Graph()
    a, b = graph.placeholder('a'), graph.placeholder('b')
    left = graph.call_function(torch.ops.aten.mm.default, (a, b))
    right = graph.call_function(torch.ops.aten.neg.default, (a,))
    other = graph.placeholder('other')
    remote = graph.call_function(torch.ops.aten.neg.default, (other,))
    graph.output((left, remote))
    context = GraphContext(torch.fx.GraphModule(torch.nn.Module(), graph))
    anchor, positions = next(context.neighborhoods(radius=2))
    assert anchor is left
    assert context.positions[right] in positions
    assert context.positions[remote] not in positions
    assert context.positions[a] in positions


def test_attribute_boundary_is_lifted_and_role_survives_serialization():
    instance = torch.nn.Module()
    instance.register_buffer('weight', torch.randn(3, 4))
    graph = torch.fx.Graph()
    a, weight = graph.placeholder('a'), graph.get_attr('weight')
    result = graph.call_function(torch.ops.aten.mm.default, (a, weight))
    graph.output(result)
    gm = torch.fx.GraphModule(instance, graph)
    ValueInterpreter(gm).run(torch.randn(2, 3))
    context = GraphContext(gm)
    encoding = RegionEncoding(context, next(context.neighborhoods())[1])
    assert not any(node.op == 'get_attr' for node in encoding.build_module().graph.nodes)
    reencode_serialized(encoding)
    assert any(json.loads(data['color'])[0] == 'attribute' for _, data in encoding.graph.nodes(data=True))


def test_grouped_histogram_routing_and_alias_constraints_preserve_raw_specs():
    class Source(torch.nn.Module):
        def forward(self, pair, a, b, indices):
            counts = torch.ops.aten.histc.default(pair[0], bins=2, min=0, max=1)
            offsets = torch.ops.aten.cumsum.default(counts, 0, dtype=torch.int32)
            result = torch.ops.aten._grouped_mm.default(a, b, offsets)
            return result, torch.ops.aten.index_select.default(a, 0, indices), pair[1]

    gm = torch.fx.symbolic_trace(Source())
    with FakeTensorMode(allow_fallback_kernels=False):
        indices = torch.empty(16, device='cuda', dtype=torch.int64)
        pair = (torch.empty(16, device='cuda'), indices)
        a = torch.empty(16, 8, device='cuda', dtype=torch.bfloat16)
        b = torch.empty(2, 8, 16, device='cuda', dtype=torch.bfloat16)
        ValueInterpreter(gm).run(pair, a, b, indices)
        context = GraphContext(gm)
        encoding = RegionEncoding(context, tuple(range(len(context.nodes))))
        reencode_serialized(encoding)
    original, digest = copy.deepcopy(encoding.inputs), graph_digest(encoding.graph)
    replay = encoding.replay_constraints()
    assert encoding.inputs == original and graph_digest(encoding.graph) == digest
    recipes = {(entry['node'], tuple(entry['path'])): entry['gen'] for entry in replay['overrides']}
    compound = next(entry['node'] for entry in encoding.inputs if entry['spec']['kind'] == 'tuple')
    flat = next(entry['node'] for entry in encoding.inputs if entry['spec'].get('dtype') == 'torch.int64')
    assert recipes[compound, (0,)] == {'kind': 'index', 'low': 0, 'high': 2}
    assert recipes[compound, (1,)] == recipes[flat, ()] == {'kind': 'index', 'low': 0, 'high': 16}
    assert replay['fidelity']['grouped_offsets'] == 'histogram_of_synthetic_expert_ids'


def test_direct_grouped_offsets_and_unrelated_matmul_inputs():
    with FakeTensorMode(allow_fallback_kernels=False):
        gm = make_fx(torch.ops.aten._grouped_mm.default, tracing_mode='fake')(
            torch.empty(16, 8, device='cuda', dtype=torch.bfloat16),
            torch.empty(2, 8, 16, device='cuda', dtype=torch.bfloat16),
            torch.empty(2, device='cuda', dtype=torch.int32))
    context = GraphContext(gm)
    encoding = RegionEncoding(context, next(context.neighborhoods())[1])
    replay = encoding.replay_constraints()
    assert len(replay['overrides']) == 1
    assert replay['overrides'][0]['gen'] == {'kind': 'grouped_mm_offsets', 'high': 16}
    context = GraphContext(make_fx(torch.mm)(torch.randn(2, 3), torch.randn(3, 4)))
    encoding = RegionEncoding(context, next(context.neighborhoods())[1])
    assert encoding.replay_constraints() == {'overrides': [], 'fidelity': {}}


def grouped_topk_region(*, dim=-1, bins=3, topk_position=1, sort_position=0, boundary_pair=False):
    class Source(torch.nn.Module):
        def forward(self, routing, a, b):
            pair = routing if boundary_pair else torch.ops.aten.topk.default(routing, 2, dim)
            flat = torch.ops.aten.view.default(pair[topk_position], [-1])
            sorted_ids = torch.ops.aten.sort.default(flat)[sort_position]
            ids = torch.ops.prims.convert_element_type.default(sorted_ids, torch.int32)
            counts = torch.ops.aten.histc.default(ids, bins=bins, min=0, max=bins - 1)
            offsets = torch.ops.aten.cumsum.default(counts, 0, dtype=torch.int32)
            return torch.ops.aten._grouped_mm.default(a, b, offsets)

    gm = torch.fx.symbolic_trace(Source())
    with FakeTensorMode(allow_fallback_kernels=False):
        routing = (torch.empty(4, 2, device='cuda', dtype=torch.bfloat16),
                   torch.empty(4, 2, device='cuda', dtype=torch.int64)) if boundary_pair else torch.empty(
                       (4, 3) if dim == -1 else (3, 4), device='cuda', dtype=torch.bfloat16)
        a = torch.empty(8, 8, device='cuda', dtype=torch.bfloat16)
        b = torch.empty(bins, 8, 16, device='cuda', dtype=torch.bfloat16)
        ValueInterpreter(gm).run(routing, a, b)
        context = GraphContext(gm)
        encoding = RegionEncoding(context, tuple(range(len(context.nodes))))
        reencode_serialized(encoding)
    return encoding


@pytest.mark.parametrize('dim', [-1, 0])
def test_grouped_offsets_recompute_sorted_topk_indices_without_changing_scores(dim):
    encoding = grouped_topk_region(dim=dim)
    original = copy.deepcopy(encoding.inputs)
    core_hash = graph_digest(encoding.graph)
    config_hash = graph_digest(configuration_graph(encoding.graph, encoding.inputs, encoding.aliases))
    replay = encoding.replay_constraints()
    assert replay['overrides'] == []
    assert replay['fidelity']['grouped_offsets'] == 'histogram_of_recomputed_topk_indices'
    assert encoding.inputs == original and graph_digest(encoding.graph) == core_hash
    assert graph_digest(configuration_graph(encoding.graph, encoding.inputs, encoding.aliases)) == config_hash
    assert any(node.target is torch.ops.aten.topk.default for node in encoding.build_module().graph.nodes)


def test_grouped_offsets_sorted_boundary_topk_indices_keep_existing_synthetic_recipe():
    encoding = grouped_topk_region(boundary_pair=True)
    replay = encoding.replay_constraints()
    pair = next(entry['node'] for entry in encoding.inputs if entry['spec']['kind'] == 'tuple')
    assert len(replay['overrides']) == 1
    override = replay['overrides'][0]
    assert (override['node'], override['path'], override['gen']) == (
        pair, [1], {'kind': 'index', 'low': 0, 'high': 3})
    assert 'grouped_histogram_expert_ids' in override['reason']
    assert replay['fidelity']['grouped_offsets'] == 'histogram_of_synthetic_expert_ids'


@pytest.mark.parametrize('settings,message', [
    ({'bins': 2}, 'expert dimension does not match'),
    ({'topk_position': 0}, 'Unsupported grouped histogram tuple projection'),
    ({'sort_position': 1}, 'Unsupported grouped histogram tuple projection'),
])
def test_grouped_offsets_reject_unproven_computed_expert_ids(settings, message):
    encoding = grouped_topk_region(**settings)
    with pytest.raises(ValueError, match=message):
        encoding.replay_constraints()


def test_scaled_mm_selection_serialized_core_and_fake_execution():
    with FakeTensorMode(allow_fallback_kernels=False):
        gm = make_fx(lambda a, b, sa, sb: torch.ops.aten._scaled_mm.default(
            a, b, sa, sb, out_dtype=torch.bfloat16), tracing_mode='fake')(
                torch.empty(16, 32, device='cuda', dtype=torch.float8_e4m3fn),
                torch.empty_strided((32, 16), (1, 32), device='cuda', dtype=torch.float8_e4m3fn),
                torch.empty((), device='cuda'), torch.empty((), device='cuda'))
        context = GraphContext(gm)
        anchor, positions = next(context.neighborhoods(names=MATMUL_ATEN_NAMES))
        assert next(context.neighborhoods()) == (anchor, positions)
        assert is_matmul_node(anchor)
        encoding = RegionEncoding(context, positions)
        loaded, inputs = reencode_serialized(encoding)
        assert loaded(*inputs)[0].dtype == torch.bfloat16


def test_nullary_cuda_factory_is_retained_without_tensor_execution():
    with FakeTensorMode(allow_fallback_kernels=False):
        gm = make_fx(lambda a, b: torch.ops.aten.add.Tensor(
            torch.ops.aten.mm.default(a, b), torch.ops.aten.scalar_tensor.default(
                0.5, dtype=torch.float32, device=torch.device('cuda'))), tracing_mode='fake')(
                    torch.empty(2, 3, device='cuda'), torch.empty(3, 4, device='cuda'))
        context = GraphContext(gm)
        encoding = RegionEncoding(context, next(context.neighborhoods())[1])
        factory = next(node for node in encoding.calls if node.target is torch.ops.aten.scalar_tensor.default)
        assert not factory.all_input_nodes
        loaded, inputs = reencode_serialized(encoding)
        assert loaded(*inputs)[0].shape == (2, 4)


def test_native_shape_scalar_and_typed_literals_preserve_containers():
    for value in [None, 0, 1, -1, 0.5, float('inf'), [0, 1, -1], (0, 1, -1),
                  torch.bfloat16, torch.device('cuda:1'), torch.strided, torch.contiguous_format]:
        assert decode_literal(literal(value)) == value
    assert native_input_spec({'kind': 'shape', 'value': literal((0, 1, -1))}) == {
        'kind': 'shape', 'dims': [0, 1, -1], 'container': 'tuple'}
    assert native_input_spec({'kind': 'shape', 'value': literal(0)}) == {'kind': 'scalar', 'value': 0}
    assert native_input_spec({'kind': 'scalar', 'value': literal(None)}) == {'kind': 'scalar', 'value': None}


@pytest.mark.parametrize('target,left_shape,right_shape', [
    (torch.ops.aten.mv.default, (2, 3), (3,)),
    (torch.ops.aten.dot.default, (3,), (3,)),
])
def test_vector_contractions_are_real_anchors_and_replay_after_serialization(target, left_shape, right_shape):
    left, right = torch.randn(left_shape), torch.randn(right_shape)
    source = torch.fx.symbolic_trace(lambda a, b: target(a, b))
    ValueInterpreter(source).run(left, right)
    context = GraphContext(source)
    anchor, selected = next(context.neighborhoods())
    assert anchor.target is target and is_matmul_node(anchor)
    encoding = RegionEncoding(context, selected)
    loaded, inputs = reencode_serialized(encoding)
    torch.testing.assert_close(loaded(*inputs)[0], target(left, right))


@pytest.fixture
def copy_offset_source(tmp_path):
    corpus = tmp_path / 'corpus'
    path = corpus / 'models' / 'test' / 'train' / 'CopyOffsets' / 'full_graph_000.py'
    path.parent.mkdir(parents=True)
    path.write_text(textwrap.dedent('''\
        import torch

        class CapturedGraph(torch.nn.Module):
            def forward(self, a: "bf16[2, 1][1, 1]cuda:0", b: "bf16[1, 3][3, 1]cuda:0",
                        target0: "bf16[2, 3][3, 1]cuda:0", target3: "bf16[2, 3][3, 1]cuda:0"):
                result = torch.ops.aten.mm.default(a, b)
                first = torch.ops.aten.copy.default(target0, result)
                second = torch.ops.aten.copy.default(target3, result)
                return first, second
    '''))
    inputs = [{'kind': 'tensor', 'name': name, 'shape': shape, 'stride': stride,
               'storage_offset': offset, 'dtype': 'bfloat16', 'device': 'cuda:0'}
              for name, shape, stride, offset in [('a', [2, 1], [1, 1], 0), ('b', [1, 3], [3, 1], 0),
                                                 ('target0', [2, 3], [3, 1], 0), ('target3', [2, 3], [3, 1], 3)]]
    path.with_suffix('.meta.json').write_text(json.dumps({'schema_version': 1, 'inputs': inputs}))
    return corpus, path


def test_source_tracing_preserves_copy_offsets_with_warm_fake_cache(copy_offset_source):
    _, path = copy_offset_source
    with torch._dynamo.config.patch(fake_tensor_cache_enabled=True):
        for _ in range(3):
            gm, _ = load_matmul_graph(path, with_metadata=True)
            copies = [node.meta['val'] for node in gm.graph.nodes if node.target is torch.ops.aten.copy.default]
            assert [value.storage_offset() for value in copies] == [0, 3]
            assert torch._dynamo.config.fake_tensor_cache_enabled


def test_persisted_validation_preserves_copy_offsets_on_repeated_oracle_calls(copy_offset_source, tmp_path, monkeypatch):
    import oracle_harness
    from scripts import extract_matmuls

    corpus, path = copy_offset_source
    collection = extract_matmuls.Collection()
    collection.collect(path, corpus)
    canonical = tmp_path / 'output' / 'canonical'
    manifest = collection.write(canonical)
    original_loader = oracle_harness._load_oracle_module
    observed = []

    def checked_loader(directory):
        module, oracle, metadata = original_loader(directory)

        def checked_oracle(inputs):
            outputs = oracle(inputs)
            offsets = [value.storage_offset() for value in outputs]
            assert offsets == [0, 3]
            observed.append(offsets)
            return outputs

        return module, checked_oracle, metadata

    monkeypatch.setattr(oracle_harness, '_load_oracle_module', checked_loader)
    with torch._dynamo.config.patch(fake_tensor_cache_enabled=True):
        for _ in range(2):
            result = extract_matmuls.validate(canonical, manifest)
            assert result == {'device': 'fake', 'structures': 1, 'input_configs': 1}
            assert torch._dynamo.config.fake_tensor_cache_enabled
    assert observed == [[0, 3], [0, 3]]


def test_publication_stages_in_destination_and_preserves_edited_oracle(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts import extract_matmuls

    staged, canonical = tmp_path / 'staged', tmp_path / 'destination' / 'canonical'
    old, new = 'mm_existing', 'mm_new'
    for name in (old, new):
        directory = staged / name
        directory.mkdir(parents=True)
        (directory / 'meta.json').write_text(json.dumps({'pattern_hash': name}))
        (directory / 'shapes.json').write_text(json.dumps({'points': []}))
        (directory / 'repro.py').write_text('class Repro: pass\n')
        (directory / 'oracle.py').write_text('placeholder\n')
    import shutil
    shutil.copytree(staged / old, canonical / old)
    (canonical / old / 'oracle.py').write_text('edited oracle\n')
    manifest = {'patterns': [{'name': name, 'pattern_hash': name} for name in (old, new)]}
    rename = Path.rename
    calls = []

    def same_filesystem_rename(source, target):
        assert source.parent.parent == canonical
        calls.append((source, target))
        return rename(source, target)

    monkeypatch.setattr(Path, 'rename', same_filesystem_rename)
    extract_matmuls.publish(staged, canonical, manifest)
    assert len(calls) == 1 and calls[0][1] == canonical / new
    assert (canonical / old / 'oracle.py').read_text() == 'edited oracle\n'
    assert (canonical / new / 'repro.py').read_bytes() == (staged / new / 'repro.py').read_bytes()
    assert json.loads((canonical.parent / extract_matmuls.MANIFEST_NAME).read_text()) == manifest


def test_publication_rolls_back_new_directories_when_manifest_write_fails(tmp_path, monkeypatch):
    from scripts import extract_matmuls

    staged, canonical = tmp_path / 'staged', tmp_path / 'canonical'
    name = 'mm_new'
    (staged / name).mkdir(parents=True)
    (staged / name / 'meta.json').write_text(json.dumps({'pattern_hash': name}))
    manifest = {'patterns': [{'name': name, 'pattern_hash': name}]}

    def fail_manifest_write(path, value):
        raise OSError('manifest write failed')

    monkeypatch.setattr(extract_matmuls, 'write_json', fail_manifest_write)
    with pytest.raises(OSError, match='manifest write failed'):
        extract_matmuls.publish(staged, canonical, manifest)
    assert not (canonical / name).exists()
    assert not list(canonical.iterdir())
