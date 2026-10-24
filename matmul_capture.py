"""Capture and replay exact matmul-centered FX neighborhoods.

Distances use undirected producer-consumer edges with fanout, effect and metadata
barriers. The graph output is terminal.
Structure records retain directed argument ports, constants and escapes;
concrete input settings and storage memberships are recorded separately.
"""
from __future__ import annotations

import copy
import json
import math
import operator
from collections import defaultdict
from pathlib import Path

import networkx as nx
import torch
import torch.fx as fx

from full_graph_harness import _fetch_attr, load_full_graph_definition


CALL_OPS = frozenset({"call_function", "call_method", "call_module"})

# These eager RNG/state operations lack nondeterministic tags.
RNG_EFFECT_SCHEMAS = frozenset({
    "prims::inductor_random", "prims::inductor_randint",
    "inductor_prims::inductor_reserve_rng_state",
})


# Schema names, rather than substring matching: e.g. clamp/scatter are not GEMMs.
# Match all overloads, including out variants, and tolerate absent newer ops.
MATMUL_ATEN_NAMES = frozenset({
    "mm", "bmm", "matmul", "addmm", "addmm_", "baddbmm", "baddbmm_",
    "addbmm", "addbmm_", "mv", "mv_", "addmv", "addmv_", "dot", "vdot",
    "linear", "einsum", "tensordot", "_int_mm", "_scaled_mm", "_scaled_mm_v2", "_grouped_mm",
    "_scaled_grouped_mm", "_scaled_grouped_mm_v2", "_sparse_mm",
    "_sparse_addmm", "_sparse_sparse_matmul", "_sparse_semi_structured_mm",
    "_sparse_semi_structured_addmm", "_weight_int4pack_mm",
    "_weight_int4pack_mm_for_cpu", "_weight_int8pack_mm", "_dyn_quant_matmul_4bit",
})


def is_matmul_node(node: fx.Node, names=MATMUL_ATEN_NAMES) -> bool:
    target = node.target
    return (node.op == "call_function" and isinstance(target, torch._ops.OpOverload)
            and target._schema.name.startswith("aten::")
            and target._schema.name.split("::", 1)[1] in names)


def trace_region(repro):
    """Trace serialized regions without evaluating no-Proxy operator calls.

    Normal symbolic tracing eagerly folds tensor factories with literal-only
    arguments. The function mode records every overload, retaining those calls
    and effects while leaving compound inputs and shape parameters intact.
    """
    from torch.overrides import TorchFunctionMode

    tracer = fx.Tracer()

    class RecordOverloads(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if isinstance(function, torch._ops.OpOverload):
                return tracer.create_proxy('call_function', function, args, kwargs)
            return function(*args, **kwargs)

    with RecordOverloads():
        graph = tracer.trace(repro)
    graph.lint()
    return fx.GraphModule(repro, graph)


def _fake_tensor(spec: dict):
    """Fabricate metadata only, retaining device, strides and storage offset.

    Do not use the real full-graph input generator here: it falls back to CPU
    without a GPU, and tensor(data, device='cuda') can initialize CUDA even
    inside FakeTensorMode. Empty factories need neither a driver nor VRAM.
    """
    shape = spec["shape"]
    stride = spec.get("stride")
    offset = int(spec.get("storage_offset", 0) or 0)
    if stride:
        size = offset + (0 if 0 in shape else 1 + sum(
            (d - 1) * s for d, s in zip(shape, stride)))
    else:
        size = offset + math.prod(shape)
    storage = torch.empty(size, dtype=getattr(torch, spec["dtype"].removeprefix("torch.")),
                          device=spec.get("device") or "cuda")
    if stride:
        return storage.as_strided(shape, stride, offset)
    return storage[offset:offset + math.prod(shape)].reshape(shape)


@torch._dynamo.config.patch(fake_tensor_cache_enabled=False)
def load_matmul_graph(path: Path, *, with_metadata: bool = False):
    """Read the existing structured sidecar and trace without real execution.

    No real-mode fallback: missing metadata or unsupported fake operators are
    reported as failures, rather than silently changing the benchmark device.
    with_metadata also returns retained input/attribute specs and input alias
    groups mapped to the retraced FX node names.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    definition = load_full_graph_definition(path)
    with FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False):
        instance = definition.graph_cls()
        for name, spec in definition.tensor_attrs.items():
            if not hasattr(instance, name):
                instance.register_buffer(name, _fake_tensor(spec))
        original_attrs = []
        if with_metadata:
            original_attrs = [(name, _fetch_attr(instance, name), spec)
                              for name, spec in definition.tensor_attrs.items()]
        inputs = [_fake_tensor(spec) if spec["kind"] == "tensor"
                  else spec.get("value", 1) for spec in definition.input_specs]
        with torch.no_grad():
            gm = make_fx(instance, tracing_mode="fake")(*inputs)
    if with_metadata:
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        if len(placeholders) != len(definition.input_specs):
            raise ValueError(f"retrace changed input count for {path}: "
                             f"{len(definition.input_specs)} -> {len(placeholders)}")
        names = {spec["name"]: node.name for spec, node in zip(definition.input_specs, placeholders)
                 if spec.get("name") is not None}
        storage_groups = definition.metadata.get("sidecar", {}).get("storage_groups", [])
        for group in storage_groups:
            missing = [name for name in group if name not in names]
            if missing:
                raise ValueError(f"unknown retained storage-group inputs for {path}: {missing}")
        tensor_attrs = {}
        matched = set()
        for node in gm.graph.nodes:
            if node.op != "get_attr":
                continue
            value = _fetch_attr(gm, node.target)
            candidates = [(name, spec) for name, original, spec in original_attrs
                          if value is original or isinstance(value, torch.Tensor)
                          and isinstance(original, torch.Tensor)
                          and value._cdata == original._cdata
                          and value.shape == original.shape
                          and value.stride() == original.stride()
                          and value.storage_offset() == original.storage_offset()]
            if not candidates:
                continue
            identities = {json.dumps({k: v for k, v in spec.items() if k != "name"}, sort_keys=True)
                          for _, spec in candidates}
            if len(identities) > 1:
                raise ValueError(f"ambiguous retained tensor attribute for {path}: {node.target}")
            tensor_attrs[node.name] = copy.deepcopy(candidates[0][1])
            matched.update(name for name, _ in candidates)
        return gm, {
            "input_specs": {node.name: copy.deepcopy(spec)
                            for node, spec in zip(placeholders, definition.input_specs)},
            "tensor_attrs": tensor_attrs,
            "storage_groups": [[names[name] for name in group] for group in storage_groups],
            "unmatched_tensor_attrs": sorted(set(definition.tensor_attrs) - matched),
        }
    return gm



SHAPE_NAMES = frozenset({'size', 'shape', 'sizes', 'output_size', 'normalized_shape'})
STRIDED_NAMES = frozenset({'as_strided', 'as_strided_', 'as_strided_copy',
                           'as_strided_scatter', 'empty_strided', 'new_empty_strided'})


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def concrete(value):
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return value.node.hint
    return value


def literal(value):
    value = concrete(value)
    if isinstance(value, float):
        return ['float', value.hex()]
    if isinstance(value, (list, tuple)):
        return ['list' if isinstance(value, list) else 'tuple', [literal(v) for v in value]]
    if isinstance(value, dict):
        return ['dict', [[literal(k), literal(v)] for k, v in sorted(value.items(), key=lambda p: repr(p[0]))]]
    if value is None or isinstance(value, (bool, int, str)):
        return [type(value).__name__, value]
    return [f'{type(value).__module__}.{type(value).__qualname__}', str(value)]


def decode_literal(encoded):
    kind, value = encoded
    if kind in {'int', 'bool', 'str', 'NoneType'}:
        return value
    if kind == 'float':
        return float.fromhex(value)
    if kind in {'list', 'tuple'}:
        items = [decode_literal(child) for child in value]
        return items if kind == 'list' else tuple(items)
    if kind == 'dict':
        return {decode_literal(key): decode_literal(child) for key, child in value}
    if kind == 'torch.device':
        return torch.device(value)
    if kind in {'torch.dtype', 'torch.layout', 'torch.memory_format'}:
        decoded = getattr(torch, value.removeprefix('torch.'))
        if f'{type(decoded).__module__}.{type(decoded).__qualname__}' != kind:
            raise ValueError(f'Literal type mismatch: {encoded}')
        return decoded
    raise ValueError(f'Unsupported retained literal type: {kind}')


def native_input_spec(spec):
    """Decode a replay spec without modifying its retained exact descriptor."""
    result = copy.deepcopy(spec)
    kind = spec['kind']
    if kind in {'list', 'tuple'}:
        result['items'] = [native_input_spec(child) for child in spec['items']]
    elif kind == 'dict':
        result['items'] = [[decode_literal(key), native_input_spec(child)] for key, child in spec['items']]
    elif kind in {'shape', 'scalar'}:
        value = decode_literal(spec['value'])
        result.pop('value')
        if kind == 'shape' and isinstance(value, (list, tuple)):
            result['dims'] = list(value)
            if isinstance(value, tuple):
                result['container'] = 'tuple'
        else:
            result.update(kind='scalar', value=value)
    return result


def input_kind(spec):
    kind = spec['kind']
    if kind in ('list', 'tuple'):
        return [kind, [input_kind(v) for v in spec['items']]]
    if kind == 'dict':
        return [kind, [[k, input_kind(v)] for k, v in spec['items']]]
    return kind


class GraphContext:
    def __init__(self, gm, metadata=None):
        metadata = metadata or {}
        self.gm = gm
        self.nodes = tuple(gm.graph.nodes)
        self.positions = {node: i for i, node in enumerate(self.nodes)}
        self.metadata = metadata
        self.source_aliases = {}
        by_name = {node.name: node for node in self.nodes}
        for group, names in enumerate(metadata.get('storage_groups', [])):
            for name in names:
                val = by_name[name].meta.get('val')
                if isinstance(val, torch.Tensor):
                    self.source_aliases[val.untyped_storage()._cdata] = ('source', group)

    def neighborhoods(self, radius=8, names=MATMUL_ATEN_NAMES, traversal=None):
        if type(radius) is not int or radius < 0:
            raise ValueError('Neighborhood radius must be a nonnegative integer')
        if traversal is None:
            traversal = MatmulTraversal(self)
        for seed, anchor in enumerate(self.nodes):
            if not is_matmul_node(anchor, names):
                continue
            if seed in traversal.excluded:
                continue
            distances = {seed: 0}
            queue = [seed]
            for i in queue:
                if distances[i] == radius or self.nodes[i].op == 'output':
                    continue
                for j in traversal.adjacency[i]:
                    if j not in distances:
                        distances[j] = distances[i] + 1
                        queue.append(j)
            yield anchor, tuple(sorted(distances))

    def value_spec(self, value, path=()):
        value = concrete(value)
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = self.source_aliases.get(storage._cdata, ('fake', storage._cdata))
            spec = {'kind': 'tensor', 'shape': [int(concrete(d)) for d in value.shape],
                    'stride': [int(concrete(d)) for d in value.stride()], 'dtype': str(value.dtype),
                    'device': str(value.device), 'layout': str(value.layout),
                    'storage_offset': int(concrete(value.storage_offset())),
                    'requires_grad': bool(value.requires_grad),
                    'is_conj': bool(value.is_conj()), 'is_neg': bool(value.is_neg())}
            size = None if key[0] == 'source' else int(concrete(storage.nbytes()))
            source = 'captured_membership_size_unretained' if key[0] == 'source' else 'fake_tensor_metadata'
            return spec, [(path, key, size, source)]
        if isinstance(value, (list, tuple)):
            specs, storages = [], []
            for i, child in enumerate(value):
                spec, leaves = self.value_spec(child, (*path, i))
                specs.append(spec)
                storages.extend(leaves)
            return {'kind': 'list' if isinstance(value, list) else 'tuple', 'items': specs}, storages
        if isinstance(value, dict):
            specs, storages = [], []
            for key, child in sorted(value.items(), key=lambda p: repr(p[0])):
                spec, leaves = self.value_spec(child, (*path, dumps(literal(key))))
                specs.append([literal(key), spec])
                storages.extend(leaves)
            return {'kind': 'dict', 'items': specs}, storages
        return {'kind': 'scalar', 'value': literal(value)}, []

    def boundary_spec(self, node):
        if 'val' not in node.meta:
            raise ValueError(f'Missing boundary metadata: {node.name}')
        spec, storage = self.value_spec(node.meta['val'])
        override = self.metadata.get('boundary_specs', {}).get(node.name)
        if override is not None:
            return copy.deepcopy(override), storage
        retained = self.metadata.get('input_specs', {}).get(node.name)
        if node.op == 'get_attr':
            retained = self.metadata.get('tensor_attrs', {}).get(node.name)
        if retained:
            for key, value in retained.items():
                if key not in {'name', 'kind', 'shape', 'stride', 'dtype', 'device', 'storage_offset'}:
                    if key == 'value' and spec['kind'] == 'scalar':
                        continue
                    spec[key] = copy.deepcopy(value)
        return spec, storage


METADATA_QUERIES = frozenset({
    'size', 'stride', 'numel', 'dim', 'storage_offset', 'is_contiguous',
    'sym_size', 'sym_stride', 'sym_numel', 'sym_storage_offset',
})
CONSTANT_FACTORIES = frozenset({'full', 'zeros', 'ones', 'scalar_tensor'})
PRIMARY_OPERANDS = {
    'addmm': 'mat1', 'addmm_': 'mat1', 'baddbmm': 'batch1',
    'baddbmm_': 'batch1', 'addbmm': 'batch1', 'addbmm_': 'batch1',
    'addmv': 'vec', 'addmv_': 'vec', 'linear': 'input', 'einsum': 'tensors',
}


def schema_name(node):
    schema = getattr(node.target, '_schema', None)
    return schema.name.split('::', 1)[-1] if schema is not None else None


def is_effectful_node(node):
    if node.op not in CALL_OPS:
        return False
    schema = getattr(node.target, '_schema', None)
    mutable = bool(getattr(schema, 'is_mutable', False))
    rng = getattr(schema, 'name', None) in RNG_EFFECT_SCHEMAS
    return mutable or rng or node.is_impure() or any(
        'nondeterministic' in str(tag) for tag in getattr(node.target, 'tags', ()))


def _activation_value(value):
    if isinstance(value, torch.Tensor):
        return value.ndim > 0 and (value.is_floating_point() or value.is_complex())
    if isinstance(value, (tuple, list)):
        return any(_activation_value(child) for child in value)
    if isinstance(value, dict):
        return any(_activation_value(child) for child in value.values())
    return False


def primary_matmul_inputs(node):
    if not is_matmul_node(node):
        return set()
    arguments = _bind_arguments(node)
    name = PRIMARY_OPERANDS.get(schema_name(node), 'self')
    if name not in arguments:
        name = next((key for key in ('input', 'mat1', 'a', 'A') if key in arguments), None)
    value = arguments.get(name)
    if schema_name(node) == 'einsum' and isinstance(value, (list, tuple)):
        value = value[0] if value else None
    nodes = []
    fx.map_arg(value, nodes.append)
    return set(nodes)


class MatmulTraversal:
    """Stop broad transit at hubs while retaining genuine contraction routes.

    Masks use the full source graph, before lifting inputs. A constant can be
    reached and retained, but never links its otherwise unrelated consumers.
    High-fanout values expand only along protected value-data edges. Ordinary
    nodes still expand in both directions; the radius is never extended.
    Effect barriers are excluded. MM paths obey barriers, while independence
    still uses source ancestry, including paths through excluded effects.
    """

    def __init__(self, context, fanout_limit=8):
        if type(fanout_limit) is not int or fanout_limit < 1:
            raise ValueError('Fanout limit must be a positive integer')
        self.context = context
        self.fanout_limit = fanout_limit
        nodes, positions = context.nodes, context.positions
        count = len(nodes)
        self.excluded = {i for i, node in enumerate(nodes) if is_effectful_node(node)}
        self.constants, self.metadata_nodes = set(), set()
        self.parents = [set() for _ in nodes]
        self.children = [set() for _ in nodes]
        source_parents, source_children = [set() for _ in nodes], [set() for _ in nodes]
        self.adjacency = [set() for _ in nodes]
        for i, node in enumerate(nodes):
            name, value = schema_name(node), node.meta.get('val')
            if name in CONSTANT_FACTORIES and not node.all_input_nodes and isinstance(value, torch.Tensor):
                self.constants.add(i)
            if name in METADATA_QUERIES or node.op == 'call_method' and node.target in METADATA_QUERIES:
                self.metadata_nodes.add(i)
            refs = []
            schema = getattr(node.target, '_schema', None)
            if schema is None:
                refs.extend(node.all_input_nodes)
            else:
                arguments = _bind_arguments(node)
                for argument in schema.arguments:
                    if argument.name in SHAPE_NAMES or name in STRIDED_NAMES and argument.name in {'stride', 'storage_offset'}:
                        continue
                    fx.map_arg(arguments[argument.name], refs.append)
            for parent in set(refs):
                j = positions[parent]
                if node.op != 'output':
                    source_parents[i].add(j)
                    source_children[j].add(i)
                if i in self.excluded or j in self.excluded:
                    continue
                self.adjacency[i].add(j)
                self.adjacency[j].add(i)
                if node.op != 'output':
                    self.parents[i].add(j)
                    self.children[j].add(i)
        terminals = self.constants | self.metadata_nodes
        for i in terminals:
            for child in self.children[i]:
                self.parents[child].discard(i)
            self.children[i].clear()
            for child in source_children[i]:
                source_parents[child].discard(i)
            source_children[i].clear()

        anchors = [i for i, node in enumerate(nodes) if is_matmul_node(node) and i not in self.excluded]
        bits = {i: 1 << rank for rank, i in enumerate(anchors)}
        nearest_up, nearest_down = [0] * count, [0] * count
        all_up, all_down = [0] * count, [0] * count
        for i in range(count):
            upstream, every = 0, 0
            for parent in self.parents[i]:
                upstream |= nearest_up[parent]
            for parent in source_parents[i]:
                every |= all_up[parent]
            nearest_up[i] = bits.get(i, upstream)
            all_up[i] = every | bits.get(i, 0)
        for i in reversed(range(count)):
            downstream, every = 0, 0
            for child in self.children[i]:
                downstream |= nearest_down[child]
            for child in source_children[i]:
                every |= all_down[child]
            nearest_down[i] = bits.get(i, downstream)
            all_down[i] = every | bits.get(i, 0)
        self.vertical_edges = {(parent, child) for parent in range(count)
                               for child in self.children[parent]
                               if nearest_up[parent] and nearest_down[child]}

        primary = {i: {positions[parent] for parent in primary_matmul_inputs(nodes[i])} for i in anchors}
        left_routes = [0] * count
        route_edges = [set() for _ in nodes]
        for i in reversed(range(count)):
            if i in bits or i in terminals:
                continue
            for child in self.children[i]:
                route = bits[child] if child in bits and i in primary[child] else (
                    left_routes[child] if child not in bits else 0)
                if route:
                    left_routes[i] |= route
                    route_edges[i].add(child)
        self.forks = [i for i in range(count) if i not in terminals
                      and (_activation_value(nodes[i].meta.get('val')) or any(
                          child in bits and i in primary[child] for child in route_edges[i]))
                      and len(route_edges[i]) >= 2 and left_routes[i].bit_count() >= 2]
        active = set(self.forks)
        self.fork_edges = set()
        for i in range(count):
            if i not in active:
                continue
            for child in route_edges[i]:
                self.fork_edges.add((i, child))
                if child not in bits:
                    active.add(child)

        self.joins, self.join_edges = [], set()
        for i, node in enumerate(nodes):
            if node.op not in CALL_OPS or i in terminals:
                continue
            parents = sorted(parent for parent in self.parents[i] if nearest_up[parent])
            horizontal = False
            for offset, left in enumerate(parents):
                for right in parents[offset + 1:]:
                    left_only = nearest_up[left] & ~nearest_up[right]
                    right_only = nearest_up[right] & ~nearest_up[left]
                    while left_only and right_only:
                        bit = left_only & -left_only
                        anchor = anchors[bit.bit_length() - 1]
                        if right_only & ~(all_up[anchor] | all_down[anchor]):
                            horizontal = True
                            break
                        left_only ^= bit
                    if horizontal:
                        break
                if horizontal:
                    break
            if not horizontal:
                continue
            self.joins.append(i)
            pending, visited = [i], {i}
            for child in pending:
                if child in bits and child != i:
                    continue
                for parent in self.parents[child]:
                    if not nearest_up[parent]:
                        continue
                    self.join_edges.add((parent, child))
                    if parent not in visited:
                        visited.add(parent)
                        pending.append(parent)

        protected = self.vertical_edges | self.fork_edges | self.join_edges
        undirected = [set() for _ in nodes]
        for parent, child in protected:
            undirected[parent].add(child)
            undirected[child].add(parent)
        self.hubs = [i for i, node in enumerate(nodes) if len(node.users) >= fanout_limit]
        for i in self.hubs:
            self.adjacency[i].intersection_update(undirected[i])
        for i in terminals:
            self.adjacency[i].clear()


def add_port(graph, source, target, path):
    ports = json.loads(graph.edges[source, target]['ports']) if graph.has_edge(source, target) else []
    ports.append(path)
    graph.add_edge(source, target, ports=dumps(sorted(ports, key=dumps)))


def configuration_graph(structure, inputs, aliases):
    graph = structure.copy()
    for entry in inputs:
        node = entry['node']
        spec = dict(entry['spec'])
        spec.pop('constraint_source', None)
        graph.nodes[node]['color'] = dumps([json.loads(structure.nodes[node]['color']), spec])
    for alias in aliases:
        node = len(graph)
        graph.add_node(node, color=dumps(['storage', alias['nbytes']]))
        for member in alias['members']:
            add_port(graph, node, member['node'], ['alias', *member['path']])
    return graph


class RegionEncoding:
    def __init__(self, context, selected, escaping_nodes=None):
        self.context = context
        self.calls = [context.nodes[i] for i in selected if context.nodes[i].op in CALL_OPS]
        self.call_ids = {node: i for i, node in enumerate(self.calls)}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.call_ids.values())
        self.input_ids = {}
        self.inputs = []
        self.alias_members = defaultdict(list)
        self.alias_sizes = {}
        self.input_sources = {}
        self.escape_ids = []
        self.effects = []
        self.literal_values = {}
        for node in self.calls:
            if is_effectful_node(node):
                raise ValueError(f'Effect calls must be excluded before encoding: {node.name}')
            i = self.call_ids[node]
            template = self.arguments(node, i)
            escaping = any(user not in self.call_ids for user in node.users) if escaping_nodes is None else node in escaping_nodes
            target = str(node.target)
            if node.op == 'call_function' and not hasattr(node.target, '_schema'):
                target = f'{getattr(node.target, "__module__", "")}.{getattr(node.target, "__qualname__", target)}'
            self.graph.nodes[i]['color'] = dumps([node.op, target, template, escaping, False])
            if escaping:
                self.escape_ids.append(i)
        self.aliases = [{'nbytes': self.alias_sizes[key][0], 'size_source': self.alias_sizes[key][1],
                         'members': members} for key, members in self.alias_members.items()]

    def new_input(self, spec, source=None, storage=(), role='input'):
        i = len(self.graph)
        self.graph.add_node(i, color=dumps([role, input_kind(spec)]))
        self.inputs.append({'node': i, 'spec': spec})
        if source is not None:
            self.input_sources[str(i)] = source.name
        for path, key, size, size_source in storage:
            self.alias_members[key].append({'node': i, 'path': list(path)})
            if key in self.alias_sizes and self.alias_sizes[key] != (size, size_source):
                raise AssertionError('Inconsistent storage allocation metadata')
            self.alias_sizes[key] = (size, size_source)
        return i

    def encode(self, value, consumer, path, shape=False):
        if isinstance(value, fx.Node):
            if value in self.call_ids:
                i = self.call_ids[value]
            else:
                if value not in self.input_ids:
                    spec, storage = self.context.boundary_spec(value)
                    role = self.context.metadata.get('input_roles', {}).get(
                        value.name, 'attribute' if value.op == 'get_attr' else 'input')
                    self.input_ids[value] = self.new_input(spec, value, storage, role)
                i = self.input_ids[value]
            add_port(self.graph, i, consumer, path)
            return ['ref']
        refs = []
        if shape:
            fx.map_arg(value, refs.append)
        if shape and not refs:
            spec = {'kind': 'shape', 'value': literal(value)}
            i = self.new_input(spec, role='shape_parameter')
            add_port(self.graph, i, consumer, path)
            return ['ref']
        if isinstance(value, (list, tuple)):
            kind = 'list' if isinstance(value, list) else 'tuple'
            return [kind, [self.encode(v, consumer, [*path, kind, i], shape=shape)
                           for i, v in enumerate(value)]]
        if isinstance(value, dict):
            return ['dict', [[literal(k), self.encode(v, consumer, [*path, 'dict', dumps(literal(k))], shape=shape)]
                             for k, v in sorted(value.items(), key=lambda p: repr(p[0]))]]
        encoded = literal(value)
        self.literal_values[dumps(encoded)] = concrete(value)
        return encoded

    def arguments(self, node, consumer):
        schema = getattr(node.target, '_schema', None)
        if schema is None:
            return [self.encode(node.args, consumer, ['args']), self.encode(node.kwargs, consumer, ['kwargs'])]
        positional = iter(node.args)
        bound = []
        name = schema.name.split('::', 1)[-1]
        known = set()
        for argument in schema.arguments:
            known.add(argument.name)
            value = node.kwargs.get(argument.name, argument.default_value) if argument.kwarg_only else (
                node.kwargs[argument.name] if argument.name in node.kwargs else next(positional, argument.default_value))
            spelling = str(argument.type_with_alias()) if hasattr(argument, 'type_with_alias') else str(argument.type)
            shape = argument.name in SHAPE_NAMES and ('SymInt' in spelling or 'List[int]' in spelling)
            shape |= name in STRIDED_NAMES and argument.name in {'stride', 'storage_offset'} and value is not None
            bound.append([argument.name, self.encode(value, consumer, ['argument', argument.name], shape=shape)])
        extra = tuple(positional)
        if extra:
            bound.append(['extra_args', self.encode(extra, consumer, ['extra_args'])])
        for key in sorted(set(node.kwargs) - known):
            bound.append([key, self.encode(node.kwargs[key], consumer, ['extra_kwargs', key])])
        return bound

    def build_module(self):
        """Lift boundary values and every shape literal into ordered inputs.

        Argument templates and labeled edges reconstruct shared operands and
        nested arguments. Original call order is preserved.
        The returned graph's python_code('self').globals supplies its globals.
        """
        graph = fx.Graph()
        values = {}
        by_name = {node.name: node for node in self.context.nodes}
        for entry in sorted(self.inputs, key=lambda entry: entry['node']):
            node_id = entry['node']
            node = graph.placeholder(f'input_{node_id}')
            source = self.input_sources.get(str(node_id))
            if source is not None:
                node.meta = dict(by_name[source].meta)
            else:
                node.meta['val'] = decode_literal(entry['spec']['value'])
            values[node_id] = node

        def decode(template, path, references):
            if template == ['ref']:
                try:
                    return values[references[dumps(path)]]
                except KeyError as error:
                    raise ValueError(f'Unresolved argument port: {path}') from error
            kind, children = template
            if kind in {'list', 'tuple'}:
                result = [decode(child, [*path, kind, i], references) for i, child in enumerate(children)]
                return result if kind == 'list' else tuple(result)
            if kind == 'dict':
                return {decode_literal(key): decode(child, [*path, 'dict', dumps(key)], references)
                        for key, child in children}
            key = dumps(template)
            return self.literal_values[key] if key in self.literal_values else decode_literal(template)

        for original in self.calls:
            node_id = self.call_ids[original]
            references = {}
            for source in self.graph.predecessors(node_id):
                for path in json.loads(self.graph.edges[source, node_id]['ports']):
                    key = dumps(path)
                    if key in references:
                        raise ValueError(f'Duplicate argument port: {path}')
                    references[key] = source
            template = json.loads(self.graph.nodes[node_id]['color'])[2]
            schema = getattr(original.target, '_schema', None)
            if schema is None:
                args = decode(template[0], ['args'], references)
                kwargs = decode(template[1], ['kwargs'], references)
            else:
                encoded = dict(template)
                args, kwargs = [], {}
                known = {argument.name for argument in schema.arguments}
                for argument in schema.arguments:
                    value = decode(encoded[argument.name], ['argument', argument.name], references)
                    if argument.kwarg_only:
                        kwargs[argument.name] = value
                    else:
                        args.append(value)
                if 'extra_args' in encoded:
                    args.extend(decode(encoded['extra_args'], ['extra_args'], references))
                for key in encoded.keys() - known - {'extra_args'}:
                    kwargs[key] = decode(encoded[key], ['extra_kwargs', key], references)
                args = tuple(args)
            node = graph.create_node(original.op, original.target, args, kwargs, name=original.name)
            node.meta = dict(original.meta)
            values[node_id] = node
        graph.output(tuple(values[node_id] for node_id in self.escape_ids))
        graph.lint()
        return fx.GraphModule(self.context.gm, graph, class_name='Repro')

    def replay_constraints(self):
        return infer_replay_constraints(self.build_module(), self.inputs, self.aliases)


def prune_matmul_outputs(context, selected):
    """Keep MM-dependent outputs, then only their live internal producers.

    Ancestry is internal to the selected region. The explicit output set must
    survive DCE: re-inferring escapes from full-source users would resurrect
    outputs that the user requested removing.
    """
    calls = {context.nodes[i] for i in selected if context.nodes[i].op in CALL_OPS}
    ordered = sorted(calls, key=context.positions.__getitem__)
    contains_matmul, escaped = {}, []
    for node in ordered:
        if is_effectful_node(node):
            raise ValueError(f'Effect calls must be excluded before output pruning: {node.name}')
        contains_matmul[node] = is_matmul_node(node) or any(
            contains_matmul.get(parent, False) for parent in node.all_input_nodes)
        if any(user not in calls for user in node.users):
            escaped.append(node)
    outputs = {node for node in escaped if contains_matmul[node]}
    live = set(outputs)
    pending = sorted(live, key=context.positions.__getitem__)
    for node in pending:
        for parent in node.all_input_nodes:
            if parent in calls and parent not in live:
                live.add(parent)
                pending.append(parent)
    retained = tuple(sorted(context.positions[node] for node in live))
    statistics = {
        'selected_calls': len(calls), 'selected_outputs': len(escaped),
        'retained_calls': len(live), 'retained_outputs': len(outputs),
        'removed_calls': len(calls) - len(live), 'removed_outputs': len(escaped) - len(outputs),
        'required_effects': 0,
        'removed_matmuls': sum(is_matmul_node(node) for node in calls - live),
    }
    return retained, outputs, statistics


def _bind_arguments(node):
    positional = iter(node.args)
    return {argument.name: node.kwargs.get(argument.name, argument.default_value) if argument.kwarg_only else (
        node.kwargs[argument.name] if argument.name in node.kwargs else next(positional, argument.default_value))
        for argument in node.target._schema.arguments}


def infer_replay_constraints(module, inputs, aliases):
    """Return necessary surrogate generators, separate from captured settings.

    An analysis copy exposes tensor leaves of compound inputs to BB's existing
    index-domain inference. The executable region and retained specs stay intact.
    Call metadata must describe the concrete configuration being analyzed.
    """
    from full_graph_harness import infer_index_bounds_from_gm, infer_permutation_indices_from_gm

    graph = fx.Graph()
    values, origins, specs, info = {}, {}, {}, {}
    placeholders = [node for node in module.graph.nodes if node.op == 'placeholder']
    ordered_inputs = sorted(inputs, key=lambda entry: entry['node'])
    if len(placeholders) != len(ordered_inputs):
        raise ValueError('Replay module and retained input count differ')

    def expose(spec, value, node_id, path=()):
        kind = spec['kind']
        if kind in {'list', 'tuple'}:
            children = [expose(child, value[i], node_id, (*path, i)) for i, child in enumerate(spec['items'])]
            return children if kind == 'list' else tuple(children)
        if kind == 'dict':
            return {decode_literal(key): expose(child, value[decode_literal(key)], node_id,
                                                (*path, dumps(key))) for key, child in spec['items']}
        if kind == 'scalar':
            return decode_literal(spec['value'])
        node = graph.placeholder(f'leaf_{node_id}_{len(origins)}')
        node.meta['val'] = value if kind == 'tensor' else decode_literal(spec['value'])
        if kind == 'tensor':
            origin = (node_id, path)
            origins[node] = origin
            specs[origin] = spec
            info[node.name] = copy.deepcopy(spec)
        return node

    for node, entry in zip(placeholders, ordered_inputs):
        if 'val' not in node.meta:
            raise ValueError(f'Missing replay boundary metadata: {node.name}')
        values[node] = expose(entry['spec'], node.meta['val'], entry['node'])
    for node in module.graph.nodes:
        if node.op == 'placeholder':
            continue
        args = fx.map_arg(node.args, values.__getitem__)
        kwargs = fx.map_arg(node.kwargs, values.__getitem__)
        if node.op == 'call_function' and node.target is operator.getitem and isinstance(args[0], (tuple, list, dict)):
            values[node] = args[0][args[1]]
            continue
        new = graph.create_node(node.op, node.target, args, kwargs, name=node.name)
        new.meta = dict(node.meta)
        values[node] = new
    graph.lint()
    analysis = fx.GraphModule(module, graph)
    constants = {}
    bounds = infer_index_bounds_from_gm(analysis, info, constants)
    permutations = infer_permutation_indices_from_gm(analysis, info)
    requested, reasons, fidelity = {}, defaultdict(set), {}

    def require(origin, recipe, reason):
        previous = requested.get(origin)
        if previous is not None and previous != recipe:
            if previous['kind'] == recipe['kind'] == 'index':
                recipe = {'kind': 'index', 'low': max(previous.get('low', 0), recipe.get('low', 0)),
                          'high': min(previous['high'], recipe['high'])}
                if recipe['low'] >= recipe['high']:
                    raise ValueError(f'Incompatible replay index domains: {origin}')
            elif {previous['kind'], recipe['kind']} == {'index', 'permutation'}:
                permutation = previous if previous['kind'] == 'permutation' else recipe
                index = previous if previous['kind'] == 'index' else recipe
                if index.get('low', 0) > 0 or index['high'] < permutation['size']:
                    raise ValueError(f'Permutation exceeds replay index domain: {origin}')
                recipe = permutation
            else:
                raise ValueError(f'Incompatible replay generators: {origin}: {previous} / {recipe}')
        requested[origin] = recipe
        reasons[origin].add(reason)

    for node, origin in origins.items():
        if node.name in bounds:
            require(origin, {'kind': 'index', 'low': 0, 'high': max(bounds[node.name], 1)}, 'boundary_index_domain')
        if node.name in constants and any('max_pool_offsets_to_indices' in str(user.target) for user in node.users):
            require(origin, {'kind': 'constant', 'value': constants[node.name]}, 'pooling_index_offsets')
        if node.name in permutations:
            require(origin, {'kind': 'permutation', 'size': permutations[node.name]}, 'boundary_permutation')

    passthrough = {'view', 'reshape', 'clone', 'contiguous', '_to_copy', 'convert_element_type',
                   'flatten', 'alias', 'detach', 'squeeze', 'unsqueeze', 't', 'transpose', 'permute'}
    for node in analysis.graph.nodes:
        schema = getattr(node.target, '_schema', None)
        if node.op != 'call_function' or schema is None or 'grouped_mm' not in schema.name:
            continue
        bound = _bind_arguments(node)
        offsets = bound.get('offs')
        if offsets is None:
            continue
        a, b = bound['self'].meta['val'], bound['mat2'].meta['val']
        if a.dim() == 2 and b.dim() == 3:
            high = int(concrete(a.shape[0]))
        elif a.dim() == 3 and b.dim() == 2:
            high = int(concrete(b.shape[1]))
        elif a.dim() == b.dim() == 2:
            high = int(concrete(a.shape[1]))
        else:
            raise ValueError(f'Unsupported grouped GEMM routing ranks: {a.dim()} x {b.dim()}')
        if offsets in origins:
            require(origins[offsets], {'kind': 'grouped_mm_offsets', 'high': high}, 'grouped_end_offsets')
            fidelity['grouped_offsets'] = 'synthetic_even_partition'
        elif offsets.op == 'call_function' and offsets.target is torch.ops.aten.cumsum.default:
            cumulative = _bind_arguments(offsets)
            histogram = cumulative['self']
            if histogram.op != 'call_function' or histogram.target is not torch.ops.aten.histc.default:
                raise ValueError(f'Unsupported computed grouped GEMM offsets: {histogram.target}')
            routing = _bind_arguments(histogram)
            bins, low, last = int(routing['bins']), int(routing['min']), int(routing['max'])
            if cumulative['dim'] not in {0, -1} or low != 0 or last != bins - 1:
                raise ValueError('Grouped histogram routing must count nonnegative integer expert IDs')
            source = routing['self']
            if source.meta['val'].numel() != high:
                raise ValueError('Grouped histogram routing does not cover the partitioned dimension')
            computed_topk, transforms = False, []
            while source not in origins:
                transforms.append(source)
                if source.op == 'call_function' and source.target is operator.getitem:
                    producer, position = source.args
                    if producer.op == 'call_function' and producer.target is torch.ops.aten.sort.default and position in {0, -2}:
                        source = _bind_arguments(producer)['self']
                        continue
                    if producer.op == 'call_function' and producer.target is torch.ops.aten.topk.default and position in {1, -1}:
                        topk = _bind_arguments(producer)
                        scores = topk['self'].meta['val']
                        if int(concrete(scores.shape[topk['dim']])) != bins:
                            raise ValueError('Top-k expert dimension does not match grouped histogram bins')
                        for transform in transforms:
                            value = transform.meta['val']
                            if value.dtype is torch.bool:
                                maximum = 1
                            elif value.is_floating_point():
                                info = torch.finfo(value.dtype)
                                maximum = min(info.max, 2 / info.eps)
                            else:
                                maximum = torch.iinfo(value.dtype).max
                            if bins - 1 > maximum:
                                raise ValueError(f'Grouped expert IDs cannot be represented by {value.dtype}')
                        computed_topk = True
                        break
                    raise ValueError(f'Unsupported grouped histogram tuple projection: {producer.target}, {position}')
                source_schema = getattr(source.target, '_schema', None)
                if source_schema is None or source_schema.name.split('::')[-1] not in passthrough:
                    raise ValueError(f'Unsupported grouped histogram input transform: {source.target}')
                source = next((value for value in _bind_arguments(source).values() if isinstance(value, fx.Node)), None)
                if source is None:
                    raise ValueError('Missing boundary source for grouped histogram routing')
            if computed_topk:
                fidelity['grouped_offsets'] = 'histogram_of_recomputed_topk_indices'
            else:
                require(origins[source], {'kind': 'index', 'low': 0, 'high': bins}, 'grouped_histogram_expert_ids')
                fidelity['grouped_offsets'] = 'histogram_of_synthetic_expert_ids'
        else:
            raise ValueError(f'Unsupported computed grouped GEMM offsets: {offsets.target}')
        fidelity['original_routing_values'] = 'not_retained_in_full_graph'

    for alias in aliases:
        members = [(entry['node'], tuple(entry['path'])) for entry in alias['members']]
        constrained = [member for member in members if member in requested]
        if not constrained:
            continue
        for member in members:
            for original in constrained:
                require(member, requested[original], 'shared_storage_domain')
        recipes = {dumps(requested[member]) for member in members}
        if len(recipes) > 1:
            for member in members:
                for original in members:
                    require(member, requested[original], 'shared_storage_domain')

    overrides = []
    for origin, recipe in sorted(requested.items(), key=lambda item: dumps(item[0])):
        retained = specs[origin].get('gen', specs[origin].get('generator'))
        compatible = retained == recipe
        if recipe['kind'] == 'index' and retained and retained.get('kind') in {'index', 'randint'}:
            compatible = ('high' in retained and retained.get('low', 0) >= recipe['low']
                          and retained['high'] <= recipe['high'])
        if not compatible:
            overrides.append({'node': origin[0], 'path': list(origin[1]), 'gen': recipe,
                              'reason': '; '.join(sorted(reasons[origin]))})
    if overrides:
        fidelity['replay_generators'] = 'inferred_surrogates_captured_descriptors_preserved'
    return {'overrides': overrides, 'fidelity': fidelity}
