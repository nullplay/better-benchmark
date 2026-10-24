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
    def forward(self, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_38, torch.float32);  input_38 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_39, input_40);  input_39 = input_40 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_tensor, alpha = 1);  convert_element_type_default = mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_41);  convert_element_type_default_1 = input_41 = None
        add_tensor = torch.ops.aten.add.Tensor(input_42, view_default, alpha = 1);  input_42 = view_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_43);  add_tensor = input_43 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default_1, [0, 4, 0, 0], 0)
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(input_44, [0, 0, 0, 4], 0);  input_44 = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 4], 0);  permute_default = None
        mm_default_1 = torch.ops.aten.mm.default(constant_pad_nd_default_2, input_45);  constant_pad_nd_default_2 = input_45 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -4, 1);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_46);  mm_default = input_46 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_47);  input_47 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_48);  input_48 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 1], False, dtype = None);  convert_element_type_default_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_49, 1536)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_49, [2], True, dtype = None);  input_49 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_50, input_51);  input_50 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(input_52, [2], True, dtype = None);  input_52 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_2);  mul_tensor_4 = sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list_1, alpha = 1);  mul_tensor_3 = sum_dim_int_list_1 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5, alpha = 1);  sub_tensor_1 = mul_tensor_5 = None
        div_tensor = torch.ops.aten.div.Tensor(input_51, 1536);  input_51 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        ge_scalar = torch.ops.aten.ge.Scalar(input_53, 0)
        lt_scalar = torch.ops.aten.lt.Scalar(input_53, 128100)
        bitwise_and_tensor = torch.ops.aten.bitwise_and.Tensor(ge_scalar, lt_scalar);  ge_scalar = lt_scalar = None
        ne_scalar = torch.ops.aten.ne.Scalar(input_53, 0)
        bitwise_and_tensor_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, ne_scalar);  bitwise_and_tensor = ne_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(bitwise_and_tensor_1, -1);  bitwise_and_tensor_1 = None
        full_default = torch.ops.aten.full.default(input_54, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_54 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze_default, [input_53], mul_tensor_6);  full_default = unsqueeze_default = input_53 = mul_tensor_6 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default_3, _unsafe_masked_index_put_accumulate_default, alpha = 1);  convert_element_type_default_3 = _unsafe_masked_index_put_accumulate_default = None
        return (mul_tensor_1, mul_tensor_2, sum_dim_int_list, add_tensor_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
