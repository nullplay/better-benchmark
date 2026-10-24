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
    def forward(self, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101):
        add_tensor = torch.ops.aten.add.Tensor(input_70, input_71, alpha = 1);  input_70 = input_71 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_72, torch.float32);  input_72 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default, alpha = 1);  add_tensor = convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(input_73, input_74);  input_73 = input_74 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_1, alpha = 1);  add_tensor_1 = convert_element_type_default_1 = None
        permute_default = torch.ops.aten.permute.default(input_75, [1, 0]);  input_75 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_76);  permute_default = input_76 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_2, alpha = 1);  add_tensor_2 = convert_element_type_default_2 = None
        view_default = torch.ops.aten.view.default(input_77, input_78);  input_77 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default_1, input_79);  permute_default_1 = input_79 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, convert_element_type_default_3, alpha = 1);  add_tensor_3 = convert_element_type_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_80, torch.bfloat16);  input_80 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_4, input_78);  convert_element_type_default_4 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_1, input_81)
        permute_default_2 = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_4 = torch.ops.aten.mm.default(permute_default_2, input_82);  permute_default_2 = input_82 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_5, alpha = 1);  add_tensor_4 = convert_element_type_default_5 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(input_83, input_84, alpha = 1);  input_83 = input_84 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_85, input_86);  input_85 = input_86 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_87);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_87, sum_dim_int_list_1);  input_87 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_88, sub_tensor_1);  input_88 = sub_tensor_1 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_6, input_78);  convert_element_type_default_6 = input_78 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_2, input_81);  input_81 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        mm_default_6 = torch.ops.aten.mm.default(permute_default_3, input_89);  permute_default_3 = input_89 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_5, input_90);  mm_default_5 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_6, torch.float32);  mm_default_6 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_5, convert_element_type_default_8, alpha = 1);  add_tensor_5 = convert_element_type_default_8 = None
        view_default_4 = torch.ops.aten.view.default(input_91, input_90);  input_91 = input_90 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_4, 0.5);  view_default_4 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, mul_tensor_5);  mul_tensor_5 = None
        tanh_default = torch.ops.aten.tanh.default(input_92);  input_92 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, add_tensor_8);  convert_element_type_default_7 = add_tensor_8 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(1, input_93, alpha = 1);  input_93 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_6, sub_tensor_2);  mul_tensor_6 = sub_tensor_2 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, 0.7978845608028654);  mul_tensor_8 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 0.5);  convert_element_type_default_9 = None
        mm_default_7 = torch.ops.aten.mm.default(input_94, input_95);  input_94 = input_95 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_7, input_96);  mm_default_7 = input_96 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(mul_tensor_4, convert_element_type_default_10, alpha = 1);  mul_tensor_4 = convert_element_type_default_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(add_tensor_9, input_97);  input_97 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, 4096)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [2], True, dtype = None)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(input_98, input_99, alpha = 1);  input_98 = input_99 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_100);  sub_tensor_3 = input_100 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_tensor_13);  mul_tensor_11 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [2], True, dtype = None);  mul_tensor_14 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_13, sum_dim_int_list_3);  sum_dim_int_list_3 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_12, sum_dim_int_list_2, alpha = 1);  mul_tensor_12 = sum_dim_int_list_2 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(add_tensor_9, mul_tensor_13);  mul_tensor_13 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1], False, dtype = None);  mul_tensor_16 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(add_tensor_9, [0, 1], False, dtype = None);  add_tensor_9 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(input_101, sum_dim_int_list_4, alpha = 1);  input_101 = sum_dim_int_list_4 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_6, sum_dim_int_list_5, alpha = 1);  add_tensor_6 = sum_dim_int_list_5 = None
        return (mm_default_3, add_tensor_7, mul_tensor_9, mul_tensor_10, mul_tensor_15, sub_tensor_4, add_tensor_10, add_tensor_11)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
