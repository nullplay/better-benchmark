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
    def forward(self, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41):
        where_self = torch.ops.aten.where.self(input_27, input_28, input_29);  input_27 = input_28 = input_29 = None
        where_self_1 = torch.ops.aten.where.self(input_30, input_31, input_32);  input_30 = input_31 = input_32 = None
        mul_tensor = torch.ops.aten.mul.Tensor(where_self, where_self_1);  where_self = where_self_1 = None
        exp_default = torch.ops.aten.exp.default(input_33);  input_33 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True, dtype = None)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1, alpha = 1);  mul_tensor = mul_tensor_1 = None
        view_default = torch.ops.aten.view.default(sub_tensor, input_34);  sub_tensor = input_34 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        add_tensor = torch.ops.aten.add.Tensor(input_35, convert_element_type_default, alpha = 1);  input_35 = convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_36);  add_tensor = input_36 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_37);  input_37 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_38);  permute_default = input_38 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_39);  mm_default = input_39 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, input_40);  input_40 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 4096)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_41);  mul_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True, dtype = None);  mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_41, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list_1, alpha = 1);  mul_tensor_3 = sum_dim_int_list_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, input_41);  input_41 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], False, dtype = None);  mul_tensor_6 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [0, 1], False, dtype = None);  convert_element_type_default_1 = None
        return (convert_element_type_default_2, mul_tensor_5, sub_tensor_1, sum_dim_int_list_3, sum_dim_int_list_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
