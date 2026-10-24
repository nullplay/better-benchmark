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
    def forward(self, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82):
        add_tensor = torch.ops.aten.add.Tensor(input_57, input_58, alpha = 1);  input_57 = input_58 = None
        view_default = torch.ops.aten.view.default(add_tensor, input_59);  add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_60, input_61);  input_60 = input_61 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default, convert_element_type_default, alpha = 1);  view_default = convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor_1, input_62);  add_tensor_1 = input_62 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_63, torch.bfloat16);  input_63 = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_1, 0.9043478220701218);  convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default_1, div_tensor);  div_tensor = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_1, input_64);  mul_tensor_1 = None
        mm_default = torch.ops.aten.mm.default(view_default_2, input_65);  view_default_2 = input_65 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_66, input_67);  input_66 = input_67 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_68, input_69, alpha = 1);  input_68 = input_69 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_2, alpha = 1);  sub_tensor = mul_tensor_2 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(input_70, 1024);  input_70 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_1);  div_tensor_1 = sub_tensor_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_1, convert_element_type_default_2, alpha = 1);  view_default_1 = convert_element_type_default_2 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_2, input_59);  add_tensor_2 = input_59 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_71, input_72);  input_71 = input_72 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1024)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [3], True, dtype = None)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(input_73, input_74, alpha = 1);  input_73 = input_74 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_2, input_75);  sub_tensor_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [3], True, dtype = None);  mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_6, sum_dim_int_list_1);  mul_tensor_6 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_5, sum_dim_int_list, alpha = 1);  mul_tensor_5 = sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_8, alpha = 1);  sub_tensor_3 = mul_tensor_8 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(input_75, 1024);  input_75 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(div_tensor_2, sub_tensor_4);  div_tensor_2 = sub_tensor_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_3, convert_element_type_default_3, alpha = 1);  view_default_3 = convert_element_type_default_3 = None
        view_default_4 = torch.ops.aten.view.default(add_tensor_3, input_64);  add_tensor_3 = input_64 = None
        permute_default = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_76);  permute_default = input_76 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, input_77);  view_default_4 = input_77 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_78);  mm_default_2 = input_78 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_79);  input_79 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, 2048)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [3], True, dtype = None)
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_80, torch.float32);  input_80 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_6, input_81, alpha = 1);  convert_element_type_default_6 = input_81 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(sub_tensor_5, input_82);  sub_tensor_5 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_12);  mul_tensor_10 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [3], True, dtype = None);  mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_12, sum_dim_int_list_3);  sum_dim_int_list_3 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(mul_tensor_11, sum_dim_int_list_2, alpha = 1);  mul_tensor_11 = sum_dim_int_list_2 = None
        sub_tensor_7 = torch.ops.aten.sub.Tensor(sub_tensor_6, mul_tensor_14, alpha = 1);  sub_tensor_6 = mul_tensor_14 = None
        div_tensor_3 = torch.ops.aten.div.Tensor(input_82, 2048);  input_82 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(div_tensor_3, sub_tensor_7);  div_tensor_3 = sub_tensor_7 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, mul_tensor_12);  mul_tensor_12 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1, 2], False, dtype = None);  mul_tensor_16 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0, 1, 2], False, dtype = None);  convert_element_type_default_4 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_15, torch.bfloat16);  mul_tensor_15 = None
        return (mm_default, convert_element_type_default_5, sum_dim_int_list_4, sum_dim_int_list_5, convert_element_type_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
