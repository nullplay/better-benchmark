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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46):
        mul_tensor = torch.ops.aten.mul.Tensor(input_30, input_31);  input_30 = input_31 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_32, input_33, alpha = 1);  input_32 = input_33 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor, alpha = 1);  sub_tensor = mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_34, sub_tensor_1);  input_34 = sub_tensor_1 = None
        add_tensor = torch.ops.aten.add.Tensor(input_35, mul_tensor_1, alpha = 1);  input_35 = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_36, 1024)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_36, [2], True, dtype = None);  input_36 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_37, input_38);  input_37 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_39, [2], True, dtype = None);  input_39 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, sum_dim_int_list_1);  mul_tensor_3 = sum_dim_int_list_1 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list, alpha = 1);  mul_tensor_2 = sum_dim_int_list = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_4, alpha = 1);  sub_tensor_2 = mul_tensor_4 = None
        div_tensor = torch.ops.aten.div.Tensor(input_38, 1024);  input_38 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_3);  div_tensor = sub_tensor_3 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_5, alpha = 1);  add_tensor = mul_tensor_5 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_40, torch.bfloat16);  input_40 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_6);  convert_element_type_default = mul_tensor_6 = None
        view_default = torch.ops.aten.view.default(mul_tensor_7, input_41);  mul_tensor_7 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_42);  input_42 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        permute_default_1 = torch.ops.aten.permute.default(input_43, [0, 2, 1, 3]);  input_43 = None
        view_default_1 = torch.ops.aten.view.default(permute_default_1, input_44);  permute_default_1 = input_44 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_41);  view_default_1 = input_41 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, view_default_2);  permute_default = view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_45);  mm_default = input_45 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_46);  view_default_3 = input_46 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        return (convert_element_type_default_2, permute_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
