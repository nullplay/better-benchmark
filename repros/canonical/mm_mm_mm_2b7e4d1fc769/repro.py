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
    def forward(self, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94):
        add_tensor = torch.ops.aten.add.Tensor(input_62, input_63, alpha = 1);  input_62 = input_63 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_64, torch.float32);  input_64 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default, alpha = 1);  add_tensor = convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(input_65, input_66);  input_65 = input_66 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_1, alpha = 1);  add_tensor_1 = convert_element_type_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_67, input_68, alpha = 1);  input_67 = input_68 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_3, input_69);  add_tensor_3 = input_69 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_70);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_70, sum_dim_int_list_1);  input_70 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_71, sub_tensor_1);  input_71 = sub_tensor_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_72);  convert_element_type_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, input_73);  input_73 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_74);  permute_default = input_74 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_1, input_75);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_3, alpha = 1);  add_tensor_2 = convert_element_type_default_3 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_76);  view_default_1 = input_76 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_77);  clone_default = input_77 = None
        view_default_4 = torch.ops.aten.view.default(input_78, input_79);  input_78 = input_79 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
        bmm_default = torch.ops.aten.bmm.default(permute_default_2, view_default_3);  permute_default_2 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_80);  view_default_3 = input_80 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default, input_81);  bmm_default = input_81 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default_1, input_82);  bmm_default_1 = input_82 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        view_default_7 = torch.ops.aten.view.default(input_83, input_72);  input_83 = input_72 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_7, input_84);  view_default_7 = input_84 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_3, input_75);  mm_default_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_4, convert_element_type_default_5, alpha = 1);  mul_tensor_4 = convert_element_type_default_5 = None
        mm_default_4 = torch.ops.aten.mm.default(input_85, input_86);  input_85 = input_86 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_4, input_75);  mm_default_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, convert_element_type_default_6, alpha = 1);  add_tensor_5 = convert_element_type_default_6 = None
        view_default_10 = torch.ops.aten.view.default(input_87, input_75);  input_87 = input_75 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, convert_element_type_default_7, alpha = 1);  add_tensor_6 = convert_element_type_default_7 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_7, input_88);  input_88 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 4096)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = None)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_89);  mul_tensor_5 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(add_tensor_7, input_89);  input_89 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1], False, dtype = None);  mul_tensor_8 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(add_tensor_7, [0, 1], False, dtype = None);  add_tensor_7 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(input_90, sum_dim_int_list_4, alpha = 1);  input_90 = sum_dim_int_list_4 = None
        mm_default_5 = torch.ops.aten.mm.default(input_91, input_92);  input_91 = input_92 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_8, alpha = 1);  add_tensor_4 = convert_element_type_default_8 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_93, torch.float32);  input_93 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_9, convert_element_type_default_9, alpha = 1);  add_tensor_9 = convert_element_type_default_9 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_10, input_94, alpha = 1);  add_tensor_10 = input_94 = None
        return (convert_element_type_default_4, permute_default_3, mul_tensor_6, sum_dim_int_list_2, mul_tensor_7, sum_dim_int_list_3, add_tensor_8, add_tensor_11)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
