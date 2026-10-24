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
        view_default = torch.ops.aten.view.default(input_38, input_39);  input_38 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_40, torch.bfloat16);  input_40 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, convert_element_type_default, alpha = 1);  view_default = convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_41);  add_tensor = input_41 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_42, input_43, alpha = 1);  input_42 = input_43 = None
        div_tensor = torch.ops.aten.div.Tensor(input_44, 256);  input_44 = None
        mul_tensor = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor);  div_tensor = sub_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_1, convert_element_type_default_1, alpha = 1);  view_default_1 = convert_element_type_default_1 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_39);  add_tensor_1 = input_39 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_45, 256)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_45, [3], True, dtype = None);  input_45 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_46, input_47);  input_46 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_48, [3], True, dtype = None);  input_48 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_3, alpha = 1);  sub_tensor_1 = mul_tensor_3 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(input_47, 256);  input_47 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_2);  div_tensor_1 = sub_tensor_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_2, convert_element_type_default_2, alpha = 1);  view_default_2 = convert_element_type_default_2 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_2, input_49);  add_tensor_2 = input_49 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_50);  permute_default = input_50 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_51);  view_default_3 = input_51 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_52);  mm_default_1 = input_52 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_53);  input_53 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 512)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [3], True, dtype = None)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_54);  mul_tensor_5 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [3], True, dtype = None);  mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_54, sum_dim_int_list_3);  sum_dim_int_list_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, sum_dim_int_list_2, alpha = 1);  mul_tensor_6 = sum_dim_int_list_2 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_54);  input_54 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1, 2], False, dtype = None);  mul_tensor_9 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_3, [0, 1, 2], False, dtype = None);  convert_element_type_default_3 = None
        return (convert_element_type_default_4, mul_tensor_8, sub_tensor_3, sum_dim_int_list_4, sum_dim_int_list_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
