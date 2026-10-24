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
    def forward(self, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_34, 1.1111111111111112);  input_34 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        view_default = torch.ops.aten.view.default(mul_tensor_1, input_35);  mul_tensor_1 = input_35 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_36);  input_36 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_37);  permute_default = input_37 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_38);  mm_default = input_38 = None
        full_default = torch.ops.aten.full.default(input_39, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_39 = None
        where_self = torch.ops.aten.where.self(input_40, full_default, view_default_1);  input_40 = full_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(where_self, input_41);  where_self = input_41 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_2, input_42);  input_42 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_43);  permute_default_1 = input_43 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True, dtype = torch.float32);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list, input_44);  sum_dim_int_list = input_44 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_3, torch.bfloat16);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_45);  mm_default_2 = input_45 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float32);  convert_element_type_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_46);  input_46 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1024)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(input_47, input_48, alpha = 1);  input_47 = input_48 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor, input_49);  sub_tensor = input_49 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = None);  mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list_1, alpha = 1);  mul_tensor_3 = sum_dim_int_list_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], False, dtype = None);  mul_tensor_7 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 1], False, dtype = None);  convert_element_type_default_2 = None
        return (mm_default_1, convert_element_type_default_3, convert_element_type_default_4, mul_tensor_6, sub_tensor_1, sum_dim_int_list_3, sum_dim_int_list_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
