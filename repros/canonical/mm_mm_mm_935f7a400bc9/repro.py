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
    def forward(self, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76):
        mul_tensor = torch.ops.aten.mul.Tensor(input_59, input_60);  input_59 = input_60 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 256)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_61);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_61, sum_dim_int_list_1);  input_61 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_62, sub_tensor_1);  input_62 = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_63, torch.bfloat16);  input_63 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_5);  convert_element_type_default = mul_tensor_5 = None
        view_default = torch.ops.aten.view.default(mul_tensor_6, input_64);  mul_tensor_6 = input_64 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_65);  view_default = input_65 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_66);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(input_67, input_66);  input_67 = input_66 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 0.7071067811865476)
        erf_default = torch.ops.aten.erf.default(mul_tensor_7);  mul_tensor_7 = None
        add_tensor = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, convert_element_type_default_3)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, -0.5);  mul_tensor_9 = None
        exp_default = torch.ops.aten.exp.default(mul_tensor_10);  mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, mul_tensor_11);  convert_element_type_default_3 = mul_tensor_11 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_12, alpha = 1);  mul_tensor_8 = mul_tensor_12 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, add_tensor_1);  convert_element_type_default_2 = add_tensor_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_13, torch.bfloat16);  mul_tensor_13 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_4, input_68);  convert_element_type_default_4 = input_68 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_69);  input_69 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_70);  permute_default = input_70 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True, dtype = torch.float32);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_2, input_71);  sum_dim_int_list_2 = input_71 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_4, torch.bfloat16);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_72);  mm_default_1 = input_72 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_4, convert_element_type_default_6, alpha = 1);  mul_tensor_4 = convert_element_type_default_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float32);  convert_element_type_default_5 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(add_tensor_2, input_73);  input_73 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_14, 256)
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [2], True, dtype = None)
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_74, torch.float32);  input_74 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_9, input_75, alpha = 1);  convert_element_type_default_9 = input_75 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(sub_tensor_2, input_76);  sub_tensor_2 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(mul_tensor_14, mul_tensor_16);  mul_tensor_14 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [2], True, dtype = None);  mul_tensor_17 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(mul_tensor_16, sum_dim_int_list_4);  sum_dim_int_list_4 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_15, sum_dim_int_list_3, alpha = 1);  mul_tensor_15 = sum_dim_int_list_3 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_18, alpha = 1);  sub_tensor_3 = mul_tensor_18 = None
        div_tensor = torch.ops.aten.div.Tensor(input_76, 256);  input_76 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_4);  div_tensor = sub_tensor_4 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_16);  mul_tensor_16 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1], False, dtype = None);  mul_tensor_20 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], False, dtype = None);  add_tensor_2 = None
        return (convert_element_type_default_7, convert_element_type_default_8, mul_tensor_19, sum_dim_int_list_5, sum_dim_int_list_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
