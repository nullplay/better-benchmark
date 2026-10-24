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
    def forward(self, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90):
        full_default = torch.ops.aten.full.default(input_70, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_70 = None
        add_tensor = torch.ops.aten.add.Tensor(input_71, input_72, alpha = 1);  input_71 = input_72 = None
        div_scalar = torch.ops.aten.div.Scalar(input_73, 768);  input_73 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, div_scalar, alpha = 1);  add_tensor = div_scalar = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_74, torch.float32);  input_74 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor);  add_tensor_1 = mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, input_75);  convert_element_type_default_1 = input_75 = None
        view_default = torch.ops.aten.view.default(mul_tensor_2, input_76);  mul_tensor_2 = input_76 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_77);  view_default = input_77 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_78);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_79, torch.bfloat16);  input_79 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_default_1, mul_tensor_3);  view_default_1 = mul_tensor_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float32);  mul_tensor_4 = None
        view_default_2 = torch.ops.aten.view.default(input_80, input_78);  input_80 = input_78 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.7071067811865476)
        erf_default = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_2, 0.5);  add_tensor_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, convert_element_type_default_4)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, -0.5);  mul_tensor_7 = None
        exp_default = torch.ops.aten.exp.default(mul_tensor_8);  mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, mul_tensor_9);  convert_element_type_default_4 = mul_tensor_9 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_10, alpha = 1);  mul_tensor_6 = mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, add_tensor_3);  convert_element_type_default_3 = add_tensor_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_11, torch.bfloat16);  mul_tensor_11 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_5, input_81);  convert_element_type_default_5 = input_81 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_82);  input_82 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_83);  permute_default = input_83 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True, dtype = torch.float32);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list, input_84);  sum_dim_int_list = input_84 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_4, torch.bfloat16);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_85);  mm_default_1 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(convert_element_type_default_6, torch.float32);  convert_element_type_default_6 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_7, [0, 1], True, dtype = torch.float32)
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list_1, input_86);  sum_dim_int_list_1 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_87, input_88)
        add_tensor_4 = torch.ops.aten.add.Tensor(input_89, 1e-06, alpha = 1)
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor_12, add_tensor_4);  mul_tensor_12 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(div_tensor, add_tensor_4);  div_tensor = None
        neg_default = torch.ops.aten.neg.default(convert_element_type_default_7)
        mul_tensor_13 = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(convert_element_type_default_7, add_tensor_4);  convert_element_type_default_7 = add_tensor_4 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [2], True, dtype = torch.float32);  mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(div_tensor_2, input_87);  input_87 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(div_tensor_2, input_88);  div_tensor_2 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True, dtype = torch.float32);  mul_tensor_15 = None
        view_default_7 = torch.ops.aten.view.default(sum_dim_int_list_3, input_86);  sum_dim_int_list_3 = input_86 = None
        neg_default_1 = torch.ops.aten.neg.default(mul_tensor_14)
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(neg_default_1, [2], True, dtype = torch.float32);  neg_default_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_14, alpha = 1);  mul_tensor_1 = mul_tensor_14 = None
        mul_scalar = torch.ops.aten.mul.Scalar(input_89, 2)
        div_tensor_3 = torch.ops.aten.div.Tensor(sum_dim_int_list_2, mul_scalar);  sum_dim_int_list_2 = mul_scalar = None
        eq_scalar = torch.ops.aten.eq.Scalar(input_89, 0);  input_89 = None
        where_self = torch.ops.aten.where.self(eq_scalar, full_default, div_tensor_3);  eq_scalar = full_default = div_tensor_3 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(where_self, 0.002607561929595828);  where_self = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_scalar_1, input_88);  mul_scalar_1 = input_88 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_16, alpha = 1);  add_tensor_5 = mul_tensor_16 = None
        expand_default = torch.ops.aten.expand.default(sum_dim_int_list_4, input_85, implicit = False);  sum_dim_int_list_4 = input_85 = None
        div_scalar_1 = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, div_scalar_1, alpha = 1);  add_tensor_6 = div_scalar_1 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16)
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_7, input_90, alpha = 1);  add_tensor_7 = input_90 = None
        return (convert_element_type_default_8, convert_element_type_default_9, view_default_6, view_default_7, convert_element_type_default_10, add_tensor_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
