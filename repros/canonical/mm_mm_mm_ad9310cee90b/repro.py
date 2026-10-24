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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58):
        mul_tensor = torch.ops.aten.mul.Tensor(input_44, input_45);  input_44 = input_45 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_46);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_46, sum_dim_int_list_1);  input_46 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_47, sub_tensor_1);  input_47 = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_48, torch.bfloat16);  input_48 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_5);  convert_element_type_default = mul_tensor_5 = None
        view_default = torch.ops.aten.view.default(mul_tensor_6, input_49);  mul_tensor_6 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_50);  view_default = input_50 = None
        getitem = input_51[2];  input_51 = None
        permute_default = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default_1 = torch.ops.aten.view.default(permute_default, input_52);  permute_default = None
        view_default_2 = torch.ops.aten.view.default(input_53, input_52);  input_53 = None
        view_default_3 = torch.ops.aten.view.default(view_default_1, input_49);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_54);  input_54 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default_1, input_55);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_52);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_4, convert_element_type_default_2, alpha = 1);  mul_tensor_4 = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        view_default_5 = torch.ops.aten.view.default(view_default_2, input_49);  view_default_2 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_5, input_56);  input_56 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [1, 0]);  view_default_5 = None
        mm_default_4 = torch.ops.aten.mm.default(permute_default_2, input_55);  permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_3, input_52);  mm_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_4, alpha = 1);  add_tensor = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        view_default_7 = torch.ops.aten.view.default(input_57, input_52);  input_57 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_49);  view_default_7 = input_49 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_8, input_58);  input_58 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_8, [1, 0]);  view_default_8 = None
        mm_default_6 = torch.ops.aten.mm.default(permute_default_3, input_55);  permute_default_3 = input_55 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_5, input_52);  mm_default_5 = input_52 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_6, alpha = 1);  add_tensor_1 = convert_element_type_default_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mm_default_6, torch.float32);  mm_default_6 = None
        return (mm_default, convert_element_type_default_3, convert_element_type_default_5, add_tensor_2, convert_element_type_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
