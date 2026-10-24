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
    def forward(self, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85):
        mul_tensor = torch.ops.aten.mul.Tensor(input_59, input_60);  input_59 = input_60 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_61, input_62, alpha = 1);  input_61 = input_62 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor, alpha = 1);  sub_tensor = mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_63, sub_tensor_1);  input_63 = sub_tensor_1 = None
        where_self = torch.ops.aten.where.self(input_64, input_65, input_66);  input_64 = input_65 = input_66 = None
        view_default = torch.ops.aten.view.default(where_self, input_67);  where_self = input_67 = None
        bmm_default = torch.ops.aten.bmm.default(input_68, view_default);  input_68 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default, input_69);  view_default = input_69 = None
        view_default_1 = torch.ops.aten.view.default(input_70, input_71);  input_70 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_72);  view_default_1 = None
        mm_default = torch.ops.aten.mm.default(view_default_2, input_73);  input_73 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_74);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_71);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, convert_element_type_default, alpha = 1);  mul_tensor_1 = convert_element_type_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        clone_default = torch.ops.aten.clone.default(input_75, memory_format = torch.contiguous_format);  input_75 = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_72);  clone_default = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, input_76);  input_76 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_74);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_71);  mm_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_2, alpha = 1);  add_tensor = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default_1, input_77);  bmm_default_1 = input_77 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_1, input_71);  clone_default_1 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_72);  view_default_7 = input_72 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_8, input_78);  input_78 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_8, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_3, input_74);  permute_default_3 = input_74 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_8, [0], True, dtype = torch.float32);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(sum_dim_int_list, input_79);  sum_dim_int_list = input_79 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_9, torch.bfloat16);  view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_4, input_71);  mm_default_4 = input_71 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_5, alpha = 1);  add_tensor_1 = convert_element_type_default_5 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float32);  convert_element_type_default_4 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_80, torch.float32);  input_80 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1.1111111111111112);  convert_element_type_default_8 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_2);  add_tensor_2 = mul_tensor_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_81);  input_81 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1536)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True, dtype = None)
        add_tensor_3 = torch.ops.aten.add.Tensor(input_82, input_83, alpha = 1);  input_82 = input_83 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_3, input_84, alpha = 1);  add_tensor_3 = input_84 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_2, input_85);  sub_tensor_2 = input_85 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True, dtype = None);  mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_6, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_5, sum_dim_int_list_1, alpha = 1);  mul_tensor_5 = sum_dim_int_list_1 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_8, alpha = 1);  sub_tensor_3 = mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_6);  mul_tensor_6 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], False, dtype = None);  mul_tensor_9 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], False, dtype = None);  mul_tensor_3 = None
        return (bmm_default, convert_element_type_default_1, convert_element_type_default_3, convert_element_type_default_6, convert_element_type_default_7, sub_tensor_4, sum_dim_int_list_3, sum_dim_int_list_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
