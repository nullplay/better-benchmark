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
    def forward(self, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105):
        full_default = torch.ops.aten.full.default(input_74, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_74 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_75, input_76);  input_75 = input_76 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_77, torch.float32);  input_77 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default, alpha = 1);  mul_tensor = convert_element_type_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor, input_78);  add_tensor = input_78 = None
        mm_default = torch.ops.aten.mm.default(input_79, input_80);  input_79 = input_80 = None
        view_default = torch.ops.aten.view.default(mm_default, input_81);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, convert_element_type_default_1, alpha = 1);  mul_tensor_1 = convert_element_type_default_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1], True, dtype = torch.float32)
        view_default_1 = torch.ops.aten.view.default(sum_dim_int_list, input_82);  sum_dim_int_list = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_1, input_83);  input_83 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_1, input_84);  add_tensor_1 = input_84 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True, dtype = torch.float32);  mul_tensor_2 = None
        view_default_2 = torch.ops.aten.view.default(sum_dim_int_list_1, input_82);  sum_dim_int_list_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16)
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_2, input_85);  convert_element_type_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_86);  input_86 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_87);  permute_default = input_87 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True, dtype = torch.float32);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_2, input_82);  sum_dim_int_list_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_4, torch.bfloat16);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_88);  mm_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        where_self = torch.ops.aten.where.self(input_89, full_default, view_default_5);  input_89 = full_default = view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(where_self, input_90);  where_self = input_90 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_6, input_91);  input_91 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_6, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_1, input_92);  permute_default_1 = input_92 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default_6, [0], True, dtype = torch.float32);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(sum_dim_int_list_3, input_93);  sum_dim_int_list_3 = input_93 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_7, torch.bfloat16);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_3, input_81);  mm_default_3 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_3, convert_element_type_default_6, alpha = 1);  mul_tensor_3 = convert_element_type_default_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float32);  convert_element_type_default_5 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True, dtype = torch.float32)
        view_default_9 = torch.ops.aten.view.default(sum_dim_int_list_4, input_82);  sum_dim_int_list_4 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_2, input_94);  input_94 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_2, input_95);  add_tensor_2 = input_95 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True, dtype = torch.float32);  mul_tensor_4 = None
        view_default_10 = torch.ops.aten.view.default(sum_dim_int_list_5, input_82);  sum_dim_int_list_5 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16)
        view_default_11 = torch.ops.aten.view.default(convert_element_type_default_9, input_85);  convert_element_type_default_9 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_11, input_96);  input_96 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_11, [1, 0])
        mm_default_6 = torch.ops.aten.mm.default(permute_default_2, input_97);  permute_default_2 = input_97 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(view_default_11, [0], True, dtype = torch.float32);  view_default_11 = None
        view_default_12 = torch.ops.aten.view.default(sum_dim_int_list_6, input_82);  sum_dim_int_list_6 = None
        view_default_13 = torch.ops.aten.view.default(mm_default_5, input_88);  mm_default_5 = input_88 = None
        mm_default_7 = torch.ops.aten.mm.default(input_98, input_99);  input_98 = input_99 = None
        view_default_14 = torch.ops.aten.view.default(mm_default_7, input_81);  mm_default_7 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_default_14, torch.float32);  view_default_14 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_5, convert_element_type_default_10, alpha = 1);  mul_tensor_5 = convert_element_type_default_10 = None
        sum_dim_int_list_7 = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0, 1], True, dtype = torch.float32)
        view_default_15 = torch.ops.aten.view.default(sum_dim_int_list_7, input_82);  sum_dim_int_list_7 = None
        view_default_16 = torch.ops.aten.view.default(input_100, input_81);  input_100 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(input_101, input_102, alpha = 1);  input_101 = input_102 = None
        view_default_17 = torch.ops.aten.view.default(input_103, input_81);  input_103 = input_81 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_17, add_tensor_4, alpha = 1);  view_default_17 = add_tensor_4 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_3, add_tensor_5);  add_tensor_5 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(add_tensor_3, input_104);  add_tensor_3 = input_104 = None
        sum_dim_int_list_8 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], True, dtype = torch.float32);  mul_tensor_6 = None
        view_default_18 = torch.ops.aten.view.default(sum_dim_int_list_8, input_82);  sum_dim_int_list_8 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16)
        view_default_19 = torch.ops.aten.view.default(convert_element_type_default_11, input_85);  convert_element_type_default_11 = input_85 = None
        sum_dim_int_list_9 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True, dtype = torch.float32)
        view_default_20 = torch.ops.aten.view.default(sum_dim_int_list_9, input_82);  sum_dim_int_list_9 = input_82 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, view_default_16);  view_default_16 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_105);  mul_tensor_7 = input_105 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        sum_dim_int_list_10 = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1], True, dtype = torch.float32);  mul_tensor_8 = None
        return (view_default_1, view_default_2, convert_element_type_default_3, convert_element_type_default_4, convert_element_type_default_7, convert_element_type_default_8, view_default_9, view_default_10, mm_default_6, view_default_12, view_default_13, view_default_15, view_default_18, view_default_19, view_default_20, convert_element_type_default_12, sum_dim_int_list_10)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
