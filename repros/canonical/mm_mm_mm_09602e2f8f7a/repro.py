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
    def forward(self, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66):
        mul_tensor = torch.ops.aten.mul.Tensor(input_49, input_50);  input_49 = input_50 = None
        view_default = torch.ops.aten.view.default(mul_tensor, input_51);  mul_tensor = input_51 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_52);  view_default = input_52 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_53);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(input_54, input_53);  input_54 = input_53 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.7071067811865476)
        erf_default = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default_1)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, -0.5);  mul_tensor_3 = None
        exp_default = torch.ops.aten.exp.default(mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mul_tensor_5);  convert_element_type_default_1 = mul_tensor_5 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_6, alpha = 1);  mul_tensor_2 = mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default, add_tensor_1);  convert_element_type_default = add_tensor_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_2, input_55);  convert_element_type_default_2 = input_55 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_56);  input_56 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_57);  permute_default = input_57 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True, dtype = torch.float32);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list, input_58);  sum_dim_int_list = input_58 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_4, torch.bfloat16);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_59);  mm_default_1 = input_59 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float32);  convert_element_type_default_3 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_60);  input_60 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, 2560)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True, dtype = None)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_61, input_62);  input_61 = input_62 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, 1.1111111111111112);  mul_tensor_10 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_63, mul_tensor_11, alpha = 1);  input_63 = mul_tensor_11 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, input_64, alpha = 1);  add_tensor_2 = input_64 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(sub_tensor, input_65);  sub_tensor = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_tensor_12);  mul_tensor_8 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [2], True, dtype = None);  mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_12, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_9, sum_dim_int_list_1, alpha = 1);  mul_tensor_9 = sum_dim_int_list_1 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_14, alpha = 1);  sub_tensor_1 = mul_tensor_14 = None
        div_tensor = torch.ops.aten.div.Tensor(input_65, 2560);  input_65 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, mul_tensor_12);  mul_tensor_12 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1], False, dtype = None);  mul_tensor_16 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0, 1], False, dtype = None);  convert_element_type_default_4 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_66, mul_tensor_15, alpha = 1);  input_66 = mul_tensor_15 = None
        return (convert_element_type_default_5, convert_element_type_default_6, sum_dim_int_list_3, sum_dim_int_list_4, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
