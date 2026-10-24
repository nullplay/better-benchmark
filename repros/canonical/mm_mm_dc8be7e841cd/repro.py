"""8-hop region from a captured real-world FX graph.

Exact structure and source provenance: meta.json and ../../matmul_manifest.json.
All retained input settings: shapes.json.
"""
import sys
from math import inf, nan
from pathlib import Path

import torch
import torch._inductor.inductor_prims
NoneType = type(None)
import torch as _fx_global_device
device = _fx_global_device.device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import default_shape_config, make_inputs_from_config, benchmark_repro

_repro_version = 3


class Repro(torch.nn.Module):
    def forward(self, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46):
        view_default = torch.ops.aten.view.default(input_31, input_32);  input_31 = input_32 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0])
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 7], 0);  permute_default = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default, input_33);  constant_pad_nd_default = input_33 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -7, 1);  mm_default = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(view_default, [0, 7, 0, 0], 0);  view_default = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(input_34, [0, 0, 0, 7], 0);  input_34 = None
        mm_default_1 = torch.ops.aten.mm.default(constant_pad_nd_default_1, constant_pad_nd_default_2);  constant_pad_nd_default_1 = constant_pad_nd_default_2 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_1, input_35);  mm_default_1 = input_35 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(input_36, convert_element_type_default, alpha = 1);  input_36 = convert_element_type_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_37, input_38, alpha = 1);  input_37 = input_38 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_39, input_40);  input_39 = input_40 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor, alpha = 1);  add_tensor_1 = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_41, input_42, alpha = 1);  input_41 = input_42 = None
        div_tensor = torch.ops.aten.div.Tensor(input_43, 768);  input_43 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor);  div_tensor = sub_tensor = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_1, alpha = 1);  add_tensor_2 = mul_tensor_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_44, torch.float32);  input_44 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_2);  add_tensor_3 = mul_tensor_2 = None
        ge_scalar = torch.ops.aten.ge.Scalar(input_45, 0)
        lt_scalar = torch.ops.aten.lt.Scalar(input_45, 50257)
        bitwise_and_tensor = torch.ops.aten.bitwise_and.Tensor(ge_scalar, lt_scalar);  ge_scalar = lt_scalar = None
        ne_scalar = torch.ops.aten.ne.Scalar(input_45, -1)
        bitwise_and_tensor_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, ne_scalar);  bitwise_and_tensor = ne_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(bitwise_and_tensor_1, -1);  bitwise_and_tensor_1 = None
        full_default = torch.ops.aten.full.default(input_46, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_46 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze_default, [input_45], mul_tensor_3);  full_default = unsqueeze_default = input_45 = mul_tensor_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(convert_element_type_default_1, _unsafe_masked_index_put_accumulate_default, alpha = 1);  convert_element_type_default_1 = _unsafe_masked_index_put_accumulate_default = None
        return (add_tensor, add_tensor_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
