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
    def forward(self, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78):
        mul_tensor = torch.ops.aten.mul.Tensor(input_55, input_56);  input_55 = input_56 = None
        add_tensor = torch.ops.aten.add.Tensor(input_57, input_58, alpha = 1);  input_57 = input_58 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor, input_59);  add_tensor = input_59 = None
        view_default = torch.ops.aten.view.default(input_60, input_61);  input_60 = input_61 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_62);  clone_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_63);  view_default_1 = None
        mm_default = torch.ops.aten.mm.default(view_default_2, input_64);  input_64 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_1, input_65);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_66);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default, alpha = 1);  mul_tensor = convert_element_type_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mm_default_2 = torch.ops.aten.mm.default(input_67, input_68);  input_67 = input_68 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_62);  mm_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        mm_default_3 = torch.ops.aten.mm.default(input_69, input_70);  input_69 = input_70 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_3, input_62);  mm_default_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default_2, convert_element_type_default_3, alpha = 1);  convert_element_type_default_2 = convert_element_type_default_3 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True, dtype = torch.float32)
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list, input_71);  sum_dim_int_list = None
        view_default_7 = torch.ops.aten.view.default(input_72, input_62);  input_72 = input_62 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_2, view_default_7);  view_default_7 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_2, input_73);  add_tensor_2 = input_73 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True, dtype = torch.float32);  mul_tensor_2 = None
        view_default_8 = torch.ops.aten.view.default(sum_dim_int_list_1, input_71);  sum_dim_int_list_1 = None
        view_default_9 = torch.ops.aten.view.default(convert_element_type_default_4, input_63);  convert_element_type_default_4 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_9, input_74);  input_74 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_9, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_2, input_65);  permute_default_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_9, [0], True, dtype = torch.float32);  view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(sum_dim_int_list_2, input_71);  sum_dim_int_list_2 = input_71 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_10, torch.bfloat16);  view_default_10 = None
        view_default_11 = torch.ops.aten.view.default(mm_default_4, input_66);  mm_default_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32);  view_default_11 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_6, alpha = 1);  add_tensor_1 = convert_element_type_default_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float32);  convert_element_type_default_5 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_75);  mul_tensor_1 = input_75 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_9, input_63);  convert_element_type_default_9 = input_63 = None
        mm_default_6 = torch.ops.aten.mm.default(view_default_12, input_76);  input_76 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_12, [1, 0]);  view_default_12 = None
        mm_default_7 = torch.ops.aten.mm.default(permute_default_3, input_65);  permute_default_3 = input_65 = None
        view_default_13 = torch.ops.aten.view.default(mm_default_6, input_66);  mm_default_6 = input_66 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_default_13, torch.float32);  view_default_13 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, convert_element_type_default_10, alpha = 1);  add_tensor_3 = convert_element_type_default_10 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mm_default_7, torch.float32);  mm_default_7 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(add_tensor_4, [0, 1], True, dtype = torch.float32)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_4, input_77);  input_77 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_4, input_78);  add_tensor_4 = input_78 = None
        return (convert_element_type_default_1, view_default_6, view_default_8, convert_element_type_default_7, convert_element_type_default_8, convert_element_type_default_11, sum_dim_int_list_3, mul_tensor_5, mul_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
