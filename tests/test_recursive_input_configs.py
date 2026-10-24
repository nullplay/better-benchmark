import json

import pytest
import torch

from input_codec import compact_from_spec, evaluate_symbolic_entry, spec_from_compact
from repro_harness import load_shape_configs, make_inputs_from_config


def test_nested_codec_preserves_leaf_specs_and_container_metadata():
    spec = {
        "kind": "tuple", "name": "packed", "source_role": "boundary",
        "items": [
            {"kind": "tensor", "name": "indices", "shape": [4], "dtype": "int64",
             "stride": [2], "storage_offset": 3, "device": "cpu", "alias_group": 0,
             "gen": {"kind": "index", "low": 0, "high": 32, "source": "histogram"},
             "requires_grad": False, "layout": "torch.strided", "is_conj": False, "is_neg": False},
            {"kind": "list", "items": [
                {"kind": "scalar", "name": "optional", "value": None},
                {"kind": "shape", "dims": [2, 4], "container": "tuple"},
                {"kind": "scalar", "value": 6},
            ]},
        ],
    }
    restored = spec_from_compact(json.loads(json.dumps(compact_from_spec(spec, include_name=True))))
    assert restored["name"] == "packed"
    assert restored["source_role"] == "boundary"
    tensor = restored["items"][0]
    for key, value in spec["items"][0].items():
        assert tensor[key] == value
    assert restored["items"][1]["kind"] == "list"
    assert restored["items"][1]["items"][0]["value"] is None
    assert restored["items"][1]["items"][0]["name"] == "optional"
    assert restored["items"][1]["items"][1]["container"] == "tuple"


def test_unknown_generator_fields_roundtrip_as_data():
    gen = {"kind": "custom", "bounds": {"low": 2, "high": 7}, "policy": "retained"}
    spec = {"kind": "tensor", "shape": [3], "dtype": "float32", "generator": gen}
    assert spec_from_compact(compact_from_spec(spec))["gen"] == gen
    ordinary = {**spec, "generator": {"kind": "index", "low": 0, "high": 32}}
    assert compact_from_spec(ordinary)[2]["gen"] == ["index", 0, 32]


def test_nested_alias_generation_uses_recipe_from_later_leaf():
    base = {"kind": "tensor", "shape": [4], "dtype": "int64", "stride": [2],
            "device": "cpu", "alias_group": 0}
    config = {"alias_group_nbytes": [12 * 8], "inputs": [
        {"kind": "tuple", "items": [
            {**base, "storage_offset": 2},
            {"kind": "list", "items": [{"kind": "scalar", "value": None}]},
        ]},
        {**base, "storage_offset": 4, "gen": {"kind": "index", "low": 0, "high": 32}},
    ]}
    packed, second = make_inputs_from_config(config)
    first, optional = packed
    assert isinstance(packed, tuple) and optional == [None]
    assert first.stride() == second.stride() == (2,)
    assert (first.storage_offset(), second.storage_offset()) == (2, 4)
    assert first.untyped_storage()._cdata == second.untyped_storage()._cdata
    assert first.untyped_storage().nbytes() == 12 * 8
    assert torch.all((first >= 0) & (first < 32))
    assert torch.all((second >= 0) & (second < 32))
    assert torch.equal(first[1:], second[:-1])
    second[0] = 19
    assert first[1].item() == 19


def test_nested_configs_load_through_shared_shapes_json_path(tmp_path):
    inputs = [{"kind": "list", "items": [
        {"kind": "tuple", "items": []},
        {"kind": "scalar", "value": None},
        {"kind": "scalar", "value": 8},
        {"kind": "shape", "dims": [2, 3], "container": "tuple"},
        {"kind": "shape", "dims": [1, 4]},
        {"kind": "tensor", "shape": [2], "dtype": "float32", "device": "cpu",
         "gen": {"kind": "constant", "value": 3.5}},
    ]}]
    (tmp_path / "shapes.json").write_text(json.dumps({
        "version": 3, "points": [{"shape_hash": "nested", "inputs": [compact_from_spec(s) for s in inputs]}],
    }))
    configs = load_shape_configs(str(tmp_path / "repro.py"))
    values = make_inputs_from_config(next(iter(configs.values())))[0]
    assert isinstance(values, list)
    assert values[:5] == [(), None, 8, (2, 3), [1, 4]]
    assert torch.equal(values[5], torch.full((2,), 3.5))


@pytest.mark.parametrize("gen", [None, {"kind": "constant", "value": 2.5}])
def test_nested_standalone_offset_preserves_expanded_layout(gen):
    spec = {"kind": "tensor", "shape": [3, 2], "stride": [0, 1], "storage_offset": 5,
            "dtype": "float32", "device": "cpu"}
    if gen is not None:
        spec["gen"] = gen
    (packed,) = make_inputs_from_config({"inputs": [{"kind": "tuple", "items": [spec]}]})
    tensor = packed[0]
    assert tensor.shape == (3, 2) and tensor.stride() == (0, 1)
    assert tensor.storage_offset() == 5
    assert torch.equal(tensor[0], tensor[1])
    if gen is not None:
        assert torch.equal(tensor, torch.full((3, 2), 2.5))


def test_exact_payload_and_flags_preserve_standalone_layout():
    spec = {"kind": "tensor", "shape": [2, 2], "stride": [1, 3], "storage_offset": 4,
            "dtype": "float32", "device": "cpu", "exact": True,
            "data": [[1.0, 2.0], [3.0, 4.0]], "requires_grad": True, "is_neg": True}
    (tensor,) = make_inputs_from_config({"inputs": [spec]})
    assert tensor.stride() == (1, 3) and tensor.storage_offset() == 4
    assert tensor.requires_grad and tensor.is_neg()
    assert torch.equal(tensor, -torch.tensor(spec["data"]))


def test_generator_field_and_bool_constant_are_honored():
    index = {"kind": "tensor", "shape": [32], "dtype": "float32", "device": "cpu",
             "generator": {"kind": "index", "low": 2, "high": 5}}
    flag = {"kind": "tensor", "shape": [4], "dtype": "bool", "device": "cpu",
            "gen": {"kind": "constant", "value": True}}
    tensor, boolean = make_inputs_from_config({"inputs": [index, flag]})
    assert torch.all((tensor >= 2) & (tensor < 5))
    assert torch.equal(boolean, torch.ones(4, dtype=torch.bool))


def test_bool_index_recipe_matches_saved_full_graph_generator():
    from full_graph_harness import make_tensor_from_spec

    spec = {"kind": "tensor", "shape": [256], "stride": [1], "storage_offset": 0,
            "dtype": "bool", "device": "cpu", "gen": {"kind": "index", "low": 0, "high": 30522}}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        expected = make_tensor_from_spec(spec, default_device="cpu")
        torch.manual_seed(17)
        (actual,) = make_inputs_from_config({"inputs": [spec]})
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("nbytes", [None, -1])
def test_unknown_alias_allocation_requires_explicit_replay_size(nbytes):
    spec = {"kind": "tensor", "shape": [2], "dtype": "float32", "device": "cpu", "alias_group": 0}
    with pytest.raises(ValueError, match="no recorded nbytes"):
        make_inputs_from_config({"inputs": [spec], "alias_group_nbytes": [nbytes]})


@pytest.mark.parametrize("size", [6, 8])
def test_zero_byte_alias_allocation_preserves_nested_empty_tensor(size):
    spec = {"kind": "tensor", "shape": [1, size, 0], "stride": [size, 1, 1],
            "storage_offset": 0, "dtype": "float32", "device": "cpu", "alias_group": 0}
    (packed,) = make_inputs_from_config({
        "inputs": [{"kind": "tuple", "items": [{"kind": "scalar", "value": None}, spec]}],
        "alias_group_nbytes": [0],
    })
    tensor = packed[1]
    assert packed[0] is None and tensor.shape == (1, size, 0)
    assert tensor.stride() == (size, 1, 1) and tensor.storage_offset() == 0
    assert tensor.untyped_storage().nbytes() == 0


def test_zero_byte_allocation_rejects_nonempty_alias_member():
    spec = {"kind": "tensor", "shape": [2], "dtype": "float32", "device": "cpu", "alias_group": 0}
    with pytest.raises(ValueError, match="zero-byte allocation for a nonempty tensor"):
        make_inputs_from_config({"inputs": [spec], "alias_group_nbytes": [0]})


def test_nested_symbolic_shapes_keep_container_options():
    entry = ["tuple", [
        [["n", 2], "f32", {"st": [2, 1], "dev": "cpu"}],
        ["S", ["n", 2], {"x": {"container": "tuple"}}],
        ["I", 3, "n"],
    ], {"n": "boundary"}]
    resolved = evaluate_symbolic_entry(entry, {"n": 3})
    assert resolved[1][0][0] == [3, 2]
    assert resolved[1][1] == ["S", [3, 2], {"x": {"container": "tuple"}}]
    assert resolved[1][2] == ["sym", 3]
    (packed,) = make_inputs_from_config({"inputs": [spec_from_compact(resolved)]})
    assert packed[0].shape == (3, 2) and packed[1:] == ((3, 2), 3)
