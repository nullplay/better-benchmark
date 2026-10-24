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
    def forward(self, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87):
        add_tensor = torch.ops.aten.add.Tensor(input_55, input_56, alpha = 1);  input_55 = input_56 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_57, input_58);  input_57 = input_58 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_59, input_60, alpha = 1);  input_59 = input_60 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_61, sub_tensor);  input_61 = sub_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_62, input_63);  input_62 = input_63 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_64, input_65, alpha = 1);  input_64 = input_65 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_66, sub_tensor_2);  input_66 = sub_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_67);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_68);  view_default = input_68 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_69, 2048)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_69, [2], True, dtype = None);  input_69 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_70, [2], True, dtype = None);  input_70 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_71, sum_dim_int_list_1);  input_71 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_5, alpha = 1);  sub_tensor_3 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_72, sub_tensor_4);  input_72 = sub_tensor_4 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_6, alpha = 1);  add_tensor_3 = mul_tensor_6 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16)
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_67);  convert_element_type_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, input_73);  input_73 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_74);  permute_default = input_74 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_75);  mm_default_1 = input_75 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_76, input_77);  input_76 = input_77 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, 2048)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True, dtype = None)
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_78);  mul_tensor_7 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [2], True, dtype = None);  mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_78, sum_dim_int_list_3);  input_78 = sum_dim_int_list_3 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_8, sum_dim_int_list_2, alpha = 1);  mul_tensor_8 = sum_dim_int_list_2 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(sub_tensor_5, mul_tensor_10, alpha = 1);  sub_tensor_5 = mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_79, sub_tensor_6);  input_79 = sub_tensor_6 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_11, alpha = 1);  add_tensor_4 = mul_tensor_11 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_2, input_67);  convert_element_type_default_2 = input_67 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_3, input_80);  input_80 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        mm_default_4 = torch.ops.aten.mm.default(permute_default_1, input_81);  permute_default_1 = input_81 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_3, input_82);  mm_default_3 = input_82 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_83);  view_default_4 = input_83 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_84);  clone_default = input_84 = None
        bmm_default = torch.ops.aten.bmm.default(input_85, view_default_6);  input_85 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_6, input_86);  view_default_6 = input_86 = None
        view_default_7 = torch.ops.aten.view.default(bmm_default, input_87);  bmm_default = None
        view_default_8 = torch.ops.aten.view.default(bmm_default_1, input_87);  bmm_default_1 = input_87 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        return (mm_default, mm_default_2, view_default_2, convert_element_type_default_3, convert_element_type_default_4, permute_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
