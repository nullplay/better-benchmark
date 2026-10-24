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
    def forward(self, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88):
        add_tensor = torch.ops.aten.add.Tensor(input_60, input_61, alpha = 1);  input_60 = input_61 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, input_62);  add_tensor = input_62 = None
        view_default = torch.ops.aten.view.default(input_63, input_64);  input_63 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default, alpha = 1);  mul_tensor = convert_element_type_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_1, input_65);  add_tensor_1 = input_65 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16)
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_66);  convert_element_type_default_1 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_67);  view_default_1 = input_67 = None
        view_default_2 = torch.ops.aten.view.default(input_68, input_69);  input_68 = input_69 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, input_70);  view_default_2 = input_70 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_64);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, convert_element_type_default_2, alpha = 1);  mul_tensor_1 = convert_element_type_default_2 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True, dtype = torch.float32)
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list, input_71);  sum_dim_int_list = None
        view_default_5 = torch.ops.aten.view.default(input_72, input_64);  input_72 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_5, input_73)
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_2, input_74, alpha = 1);  mul_tensor_2 = input_74 = None
        view_default_6 = torch.ops.aten.view.default(input_75, input_64);  input_75 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_6, add_tensor_3, alpha = 1);  view_default_6 = add_tensor_3 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_2, add_tensor_4);  add_tensor_4 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_2, input_76);  add_tensor_2 = input_76 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True, dtype = torch.float32);  mul_tensor_3 = None
        view_default_7 = torch.ops.aten.view.default(sum_dim_int_list_1, input_71);  sum_dim_int_list_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_3, input_66);  convert_element_type_default_3 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_8, input_77);  input_77 = None
        permute_default = torch.ops.aten.permute.default(view_default_8, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default, input_78);  permute_default = input_78 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_8, [0], True, dtype = torch.float32);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(sum_dim_int_list_2, input_71);  sum_dim_int_list_2 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_9, torch.bfloat16);  view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_2, input_64);  mm_default_2 = input_64 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float32);  convert_element_type_default_4 = None
        view_default_11 = torch.ops.aten.view.default(view_default_10, input_79);  view_default_10 = input_79 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_11, [0, 2, 1, 3]);  view_default_11 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_12 = torch.ops.aten.view.default(clone_default, input_80);  clone_default = input_80 = None
        bmm_default = torch.ops.aten.bmm.default(input_81, view_default_12);  input_81 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_12, input_82);  view_default_12 = input_82 = None
        view_default_13 = torch.ops.aten.view.default(bmm_default, input_83);  bmm_default = input_83 = None
        view_default_14 = torch.ops.aten.view.default(bmm_default_1, input_84);  bmm_default_1 = input_84 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_14, input_85);  view_default_14 = input_85 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_13, [0, 2, 1, 3]);  view_default_13 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True, dtype = torch.float32)
        view_default_15 = torch.ops.aten.view.default(sum_dim_int_list_3, input_71);  sum_dim_int_list_3 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, view_default_5);  view_default_5 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_73);  mul_tensor_4 = input_73 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], True, dtype = torch.float32);  mul_tensor_6 = None
        view_default_16 = torch.ops.aten.view.default(sum_dim_int_list_4, input_71);  sum_dim_int_list_4 = None
        view_default_17 = torch.ops.aten.view.default(convert_element_type_default_7, input_66);  convert_element_type_default_7 = input_66 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_17, input_86);  input_86 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_17, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_3, input_87);  permute_default_3 = input_87 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(view_default_17, [0], True, dtype = torch.float32);  view_default_17 = None
        view_default_18 = torch.ops.aten.view.default(sum_dim_int_list_5, input_71);  sum_dim_int_list_5 = input_71 = None
        view_default_19 = torch.ops.aten.view.default(mm_default_4, input_88);  mm_default_4 = input_88 = None
        return (mm_default, view_default_4, view_default_7, convert_element_type_default_5, convert_element_type_default_6, mul_tensor_5, permute_default_2, view_default_15, view_default_16, mm_default_5, view_default_18, view_default_19)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
