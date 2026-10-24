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
    def forward(self, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87):
        add_tensor = torch.ops.aten.add.Tensor(input_66, input_67, alpha = 1);  input_66 = input_67 = None
        div_scalar = torch.ops.aten.div.Scalar(input_68, 768);  input_68 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, div_scalar, alpha = 1);  add_tensor = div_scalar = None
        div_tensor = torch.ops.aten.div.Tensor(input_69, 8.0);  input_69 = None
        view_default = torch.ops.aten.view.default(div_tensor, input_70);  div_tensor = input_70 = None
        bmm_default = torch.ops.aten.bmm.default(input_71, view_default);  input_71 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default, input_72);  view_default = input_72 = None
        view_default_1 = torch.ops.aten.view.default(bmm_default_1, input_73);  bmm_default_1 = input_73 = None
        view_default_2 = torch.ops.aten.view.default(input_74, input_75);  input_74 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_76);  view_default_2 = None
        mm_default = torch.ops.aten.mm.default(view_default_3, input_77);  input_77 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_78);  permute_default = None
        view_default_4 = torch.ops.aten.view.default(mm_default, input_75);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        clone_default = torch.ops.aten.clone.default(input_79, memory_format = torch.contiguous_format);  input_79 = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_76);  clone_default = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_5, input_80);  input_80 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_5, [1, 0]);  view_default_5 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_78);  permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_2, input_75);  mm_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_2, alpha = 1);  convert_element_type_default = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_1, input_75);  clone_default_1 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_76);  view_default_7 = input_76 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_8, input_81);  input_81 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_8, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_3, input_78);  permute_default_3 = input_78 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_8, [0], True, dtype = torch.float32);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(sum_dim_int_list, input_82);  sum_dim_int_list = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_9, torch.bfloat16);  view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_4, input_75);  mm_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_5, alpha = 1);  add_tensor_2 = convert_element_type_default_5 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float32);  convert_element_type_default_4 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0, 1], True, dtype = torch.float32)
        view_default_11 = torch.ops.aten.view.default(sum_dim_int_list_1, input_82);  sum_dim_int_list_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_83, input_84)
        add_tensor_4 = torch.ops.aten.add.Tensor(input_85, 1e-06, alpha = 1)
        div_tensor_1 = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_4);  mul_tensor = None
        div_tensor_2 = torch.ops.aten.div.Tensor(div_tensor_1, add_tensor_4);  div_tensor_1 = None
        neg_default = torch.ops.aten.neg.default(add_tensor_3)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(neg_default, div_tensor_2);  neg_default = div_tensor_2 = None
        div_tensor_3 = torch.ops.aten.div.Tensor(add_tensor_3, add_tensor_4);  add_tensor_3 = add_tensor_4 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True, dtype = torch.float32);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(div_tensor_3, input_83);  input_83 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(div_tensor_3, input_84);  div_tensor_3 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True, dtype = torch.float32);  mul_tensor_3 = None
        view_default_12 = torch.ops.aten.view.default(sum_dim_int_list_3, input_82);  sum_dim_int_list_3 = input_82 = None
        neg_default_1 = torch.ops.aten.neg.default(mul_tensor_2)
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(neg_default_1, [2], True, dtype = torch.float32);  neg_default_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_2, alpha = 1);  add_tensor_1 = mul_tensor_2 = None
        mul_scalar = torch.ops.aten.mul.Scalar(input_85, 2)
        div_tensor_4 = torch.ops.aten.div.Tensor(sum_dim_int_list_2, mul_scalar);  sum_dim_int_list_2 = mul_scalar = None
        eq_scalar = torch.ops.aten.eq.Scalar(input_85, 0);  input_85 = None
        where_self = torch.ops.aten.where.self(eq_scalar, input_86, div_tensor_4);  eq_scalar = input_86 = div_tensor_4 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(where_self, 0.002607561929595828);  where_self = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_scalar_1, input_84);  mul_scalar_1 = input_84 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_4, alpha = 1);  add_tensor_5 = mul_tensor_4 = None
        expand_default = torch.ops.aten.expand.default(sum_dim_int_list_4, input_75, implicit = False);  sum_dim_int_list_4 = input_75 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, input_87, alpha = 1);  add_tensor_6 = input_87 = None
        return (bmm_default, convert_element_type_default_1, convert_element_type_default_3, convert_element_type_default_6, convert_element_type_default_7, view_default_11, view_default_12, expand_default, add_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
