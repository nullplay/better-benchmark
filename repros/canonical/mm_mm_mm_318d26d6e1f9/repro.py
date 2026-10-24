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
    def forward(self, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112):
        add_tensor = torch.ops.aten.add.Tensor(input_72, input_73, alpha = 1);  input_72 = input_73 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_74, input_75);  input_74 = input_75 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_76, input_77, alpha = 1);  input_76 = input_77 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_78, sub_tensor);  input_78 = sub_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_79, input_80);  input_79 = input_80 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_81, input_82, alpha = 1);  input_81 = input_82 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_83, sub_tensor_2);  input_83 = sub_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_84);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_85);  view_default = input_85 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_86, 192)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_86, [2], True, dtype = None);  input_86 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_87, [2], True, dtype = None);  input_87 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_88, sum_dim_int_list_1);  input_88 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_5, alpha = 1);  sub_tensor_3 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_89, sub_tensor_4);  input_89 = sub_tensor_4 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_6, alpha = 1);  add_tensor_3 = mul_tensor_6 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16)
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_84);  convert_element_type_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, input_90);  input_90 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_91);  permute_default = input_91 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_92);  mm_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_93, input_94);  input_93 = input_94 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, 192)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True, dtype = None)
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_95);  mul_tensor_7 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [2], True, dtype = None);  mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_95, sum_dim_int_list_3);  input_95 = sum_dim_int_list_3 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_8, sum_dim_int_list_2, alpha = 1);  mul_tensor_8 = sum_dim_int_list_2 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(sub_tensor_5, mul_tensor_10, alpha = 1);  sub_tensor_5 = mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_96, sub_tensor_6);  input_96 = sub_tensor_6 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_11, alpha = 1);  add_tensor_4 = mul_tensor_11 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16)
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_2, input_84);  convert_element_type_default_2 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_3, input_97);  input_97 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(input_98, [0, 2, 1, 3]);  input_98 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_2, input_99);  permute_default_2 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_84);  view_default_4 = None
        mm_default_4 = torch.ops.aten.mm.default(permute_default_1, view_default_5);  permute_default_1 = view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_3, input_99);  mm_default_3 = input_99 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_100);  view_default_6 = input_100 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_101, 192)
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(input_101, [2], True, dtype = None);  input_101 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(input_102, [2], True, dtype = None);  input_102 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_103, sum_dim_int_list_5);  input_103 = sum_dim_int_list_5 = None
        sub_tensor_7 = torch.ops.aten.sub.Tensor(mul_tensor_12, sum_dim_int_list_4, alpha = 1);  mul_tensor_12 = sum_dim_int_list_4 = None
        sub_tensor_8 = torch.ops.aten.sub.Tensor(sub_tensor_7, mul_tensor_13, alpha = 1);  sub_tensor_7 = mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_104, sub_tensor_8);  input_104 = sub_tensor_8 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_14, alpha = 1);  add_tensor_5 = mul_tensor_14 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16)
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_4, input_84);  convert_element_type_default_4 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_8, input_105);  input_105 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_8, [1, 0]);  view_default_8 = None
        mm_default_6 = torch.ops.aten.mm.default(permute_default_4, input_106);  permute_default_4 = input_106 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_5, input_92);  mm_default_5 = input_92 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(input_107, input_108);  input_107 = input_108 = None
        sub_tensor_9 = torch.ops.aten.sub.Tensor(input_109, input_110, alpha = 1);  input_109 = input_110 = None
        sub_tensor_10 = torch.ops.aten.sub.Tensor(sub_tensor_9, mul_tensor_15, alpha = 1);  sub_tensor_9 = mul_tensor_15 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(input_111, sub_tensor_10);  input_111 = sub_tensor_10 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, mul_tensor_16, alpha = 1);  add_tensor_6 = mul_tensor_16 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16);  add_tensor_7 = None
        view_default_10 = torch.ops.aten.view.default(convert_element_type_default_5, input_84);  convert_element_type_default_5 = input_84 = None
        mm_default_7 = torch.ops.aten.mm.default(view_default_10, input_112);  view_default_10 = input_112 = None
        return (mm_default, mm_default_2, view_default_2, convert_element_type_default_3, permute_default_3, mm_default_6, view_default_9, mm_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
