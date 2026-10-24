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
    def forward(self, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97):
        full_default = torch.ops.aten.full.default(input_70, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_70 = None
        mm_default = torch.ops.aten.mm.default(input_71, input_72);  input_71 = input_72 = None
        view_default = torch.ops.aten.view.default(mm_default, input_73);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [0, 1], True, dtype = torch.float32)
        view_default_1 = torch.ops.aten.view.default(sum_dim_int_list, input_74);  sum_dim_int_list = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, input_75);  input_75 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, input_76);  convert_element_type_default = input_76 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True, dtype = torch.float32);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16)
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_1, input_77);  convert_element_type_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, input_78);  input_78 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [1, 0])
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True, dtype = torch.float32);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(input_79, input_80);  input_79 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_3, input_81);  view_default_3 = input_81 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_73);  mm_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, convert_element_type_default_2, alpha = 1);  mul_tensor_1 = convert_element_type_default_2 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], True, dtype = torch.float32)
        view_default_5 = torch.ops.aten.view.default(sum_dim_int_list_3, input_74);  sum_dim_int_list_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor, input_82);  input_82 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor, input_83);  add_tensor = input_83 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True, dtype = torch.float32);  mul_tensor_2 = None
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list_4, input_74);  sum_dim_int_list_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16)
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_3, input_77);  convert_element_type_default_3 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_7, input_84);  input_84 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_7, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_1, input_85);  permute_default_1 = input_85 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(view_default_7, [0], True, dtype = torch.float32);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(sum_dim_int_list_5, input_74);  sum_dim_int_list_5 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_8, torch.bfloat16);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_3, input_86);  mm_default_3 = input_86 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float32);  convert_element_type_default_4 = None
        where_self = torch.ops.aten.where.self(input_87, full_default, view_default_9);  input_87 = full_default = view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(where_self, input_80);  where_self = input_80 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_10, input_88);  input_88 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_10, [1, 0])
        mm_default_6 = torch.ops.aten.mm.default(permute_default_2, input_89);  permute_default_2 = input_89 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(view_default_10, [0], True, dtype = torch.float32);  view_default_10 = None
        view_default_11 = torch.ops.aten.view.default(sum_dim_int_list_6, input_90);  sum_dim_int_list_6 = input_90 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_11, torch.bfloat16);  view_default_11 = None
        view_default_12 = torch.ops.aten.view.default(mm_default_5, input_73);  mm_default_5 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_12, torch.float32);  view_default_12 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_3, convert_element_type_default_8, alpha = 1);  mul_tensor_3 = convert_element_type_default_8 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mm_default_6, torch.float32);  mm_default_6 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(convert_element_type_default_7, torch.float32);  convert_element_type_default_7 = None
        sum_dim_int_list_7 = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1], True, dtype = torch.float32)
        view_default_13 = torch.ops.aten.view.default(sum_dim_int_list_7, input_74);  sum_dim_int_list_7 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_1, input_91);  input_91 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_1, input_92);  add_tensor_1 = input_92 = None
        sum_dim_int_list_8 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True, dtype = torch.float32);  mul_tensor_4 = None
        view_default_14 = torch.ops.aten.view.default(sum_dim_int_list_8, input_74);  sum_dim_int_list_8 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16)
        view_default_15 = torch.ops.aten.view.default(convert_element_type_default_11, input_77);  convert_element_type_default_11 = input_77 = None
        mm_default_7 = torch.ops.aten.mm.default(view_default_15, input_93);  input_93 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_15, [1, 0])
        sum_dim_int_list_9 = torch.ops.aten.sum.dim_IntList(view_default_15, [0], True, dtype = torch.float32);  view_default_15 = None
        view_default_16 = torch.ops.aten.view.default(input_94, input_73);  input_94 = input_73 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(view_default_16, torch.float32);  view_default_16 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_5, convert_element_type_default_12, alpha = 1);  mul_tensor_5 = convert_element_type_default_12 = None
        sum_dim_int_list_10 = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True, dtype = torch.float32)
        view_default_17 = torch.ops.aten.view.default(sum_dim_int_list_10, input_74);  sum_dim_int_list_10 = input_74 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_2, input_95);  input_95 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(add_tensor_2, input_96);  add_tensor_2 = input_96 = None
        sum_dim_int_list_11 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], True, dtype = torch.float32);  mul_tensor_6 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16)
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_7, input_97, alpha = 1);  mul_tensor_7 = input_97 = None
        return (view_default_1, sum_dim_int_list_1, mm_default_1, permute_default, sum_dim_int_list_2, view_default_5, view_default_6, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_9, convert_element_type_default_10, view_default_13, view_default_14, mm_default_7, permute_default_3, sum_dim_int_list_9, view_default_17, sum_dim_int_list_11, convert_element_type_default_13, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
