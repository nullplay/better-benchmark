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
    def forward(self, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87):
        mm_default = torch.ops.aten.mm.default(input_63, input_64);  input_63 = input_64 = None
        view_default = torch.ops.aten.view.default(mm_default, input_65);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(input_66, input_67);  input_66 = input_67 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        full_default = torch.ops.aten.full.default(input_65, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_1, 1, 0);  full_default = convert_element_type_default_1 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, select_scatter_default, alpha = 1);  convert_element_type_default = select_scatter_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_68, torch.float32);  input_68 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor);  add_tensor = mul_tensor = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16)
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_69, torch.bfloat16);  input_69 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, mul_tensor_2);  convert_element_type_default_3 = mul_tensor_2 = None
        view_default_1 = torch.ops.aten.view.default(mul_tensor_3, input_70);  mul_tensor_3 = input_70 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_1, input_71);  input_71 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default, input_72);  permute_default = input_72 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True, dtype = torch.float32);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(sum_dim_int_list, input_73);  sum_dim_int_list = input_73 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_2, torch.bfloat16);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_2, input_74);  mm_default_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float32);  convert_element_type_default_5 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_75, torch.bfloat16);  input_75 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1.1111111111111112);  convert_element_type_default_8 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_3, mul_tensor_4);  view_default_3 = mul_tensor_4 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.float32);  mul_tensor_5 = None
        view_default_4 = torch.ops.aten.view.default(input_76, input_74);  input_76 = input_74 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_10, 0.7071067811865476)
        erf_default = torch.ops.aten.erf.default(mul_tensor_6);  mul_tensor_6 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(add_tensor_1, 0.5);  add_tensor_1 = None
        exp_default = torch.ops.aten.exp.default(input_77);  input_77 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_10, mul_tensor_8);  convert_element_type_default_10 = mul_tensor_8 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9, alpha = 1);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_9, add_tensor_2);  convert_element_type_default_9 = add_tensor_2 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_11, input_78);  convert_element_type_default_11 = input_78 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_5, input_79);  input_79 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_5, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_1, input_80);  permute_default_1 = input_80 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True, dtype = torch.float32);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list_1, input_81);  sum_dim_int_list_1 = input_81 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_4, input_65);  mm_default_4 = input_65 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_82, 1e-06, alpha = 1);  input_82 = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_12, add_tensor_3);  add_tensor_3 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(div_tensor, input_83);  input_83 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(div_tensor, input_84);  div_tensor = None
        neg_default = torch.ops.aten.neg.default(mul_tensor_11)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(neg_default, [2], True, dtype = torch.float32);  neg_default = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_11, alpha = 1);  mul_tensor_1 = mul_tensor_11 = None
        mul_scalar = torch.ops.aten.mul.Scalar(input_85, 0.002607561929595828);  input_85 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_scalar, input_84);  mul_scalar = input_84 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_13, alpha = 1);  add_tensor_4 = mul_tensor_13 = None
        div_scalar = torch.ops.aten.div.Scalar(input_86, 768);  input_86 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, div_scalar, alpha = 1);  add_tensor_5 = div_scalar = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16)
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, input_87, alpha = 1);  add_tensor_6 = input_87 = None
        return (convert_element_type_default_6, convert_element_type_default_7, mm_default_5, view_default_6, convert_element_type_default_12, mul_tensor_12, sum_dim_int_list_2, convert_element_type_default_13, add_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
