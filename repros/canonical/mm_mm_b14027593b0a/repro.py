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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48):
        mul_tensor = torch.ops.aten.mul.Tensor(input_30, input_31);  input_30 = input_31 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_32, input_33);  input_32 = input_33 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1, alpha = 1);  mul_tensor = mul_tensor_1 = None
        view_default = torch.ops.aten.view.default(sub_tensor, input_34);  sub_tensor = input_34 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        add_tensor = torch.ops.aten.add.Tensor(input_35, convert_element_type_default, alpha = 1);  input_35 = convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_36);  add_tensor = input_36 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default_1, [0, 7, 0, 0], 0)
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(input_37, [0, 0, 0, 7], 0);  input_37 = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 7], 0);  permute_default = None
        mm_default_1 = torch.ops.aten.mm.default(constant_pad_nd_default_2, input_38);  constant_pad_nd_default_2 = input_38 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -7, 1);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_39);  mm_default = input_39 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, input_40);  input_40 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, input_41);  input_41 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [0, 1], False, dtype = None);  convert_element_type_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_42, 768)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_42, [2], True, dtype = None);  input_42 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(input_43, [2], True, dtype = None);  input_43 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_44, sum_dim_int_list_2);  input_44 = sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list_1, alpha = 1);  mul_tensor_4 = sum_dim_int_list_1 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5, alpha = 1);  sub_tensor_1 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_45, sub_tensor_2);  input_45 = sub_tensor_2 = None
        full_default = torch.ops.aten.full.default(input_46, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_46 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, input_47, [input_48], mul_tensor_6);  full_default = input_47 = input_48 = mul_tensor_6 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default_2, _unsafe_masked_index_put_accumulate_default, alpha = 1);  convert_element_type_default_2 = _unsafe_masked_index_put_accumulate_default = None
        return (mul_tensor_2, mul_tensor_3, sum_dim_int_list, add_tensor_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
