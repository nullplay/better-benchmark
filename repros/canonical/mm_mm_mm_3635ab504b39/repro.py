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
    def forward(self, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_61, torch.bfloat16);  input_61 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_62);  convert_element_type_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_63, input_64);  input_63 = input_64 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, convert_element_type_default_1, alpha = 1);  view_default = convert_element_type_default_1 = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_65);  add_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_66, input_67);  input_66 = input_67 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_68, input_69, alpha = 1);  input_68 = input_69 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_1, alpha = 1);  sub_tensor = mul_tensor_1 = None
        div_tensor = torch.ops.aten.div.Tensor(input_70, 1024);  input_70 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_1);  div_tensor = sub_tensor_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_1, convert_element_type_default_2, alpha = 1);  view_default_1 = convert_element_type_default_2 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_62);  add_tensor_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_71, torch.bfloat16);  input_71 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(convert_element_type_default_3, 0.9043478220701218);  convert_element_type_default_3 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default_2, div_tensor_1);  div_tensor_1 = None
        view_default_3 = torch.ops.aten.view.default(mul_tensor_3, input_72);  mul_tensor_3 = None
        mm_default = torch.ops.aten.mm.default(view_default_3, input_73);  input_73 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_74);  permute_default = input_74 = None
        view_default_4 = torch.ops.aten.view.default(mm_default, input_75);  mm_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(input_76, input_75);  input_76 = input_75 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 0.7071067811865476)
        erf_default = torch.ops.aten.erf.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_2, 0.5);  add_tensor_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, convert_element_type_default_6)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, -0.5);  mul_tensor_6 = None
        exp_default = torch.ops.aten.exp.default(mul_tensor_7);  mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, mul_tensor_8);  convert_element_type_default_6 = mul_tensor_8 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_9, alpha = 1);  mul_tensor_5 = mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, add_tensor_3);  convert_element_type_default_5 = add_tensor_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_7, input_77);  convert_element_type_default_7 = input_77 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_6, input_78);  input_78 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_6, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_79);  permute_default_1 = input_79 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_6, [0], True, dtype = torch.float32);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(sum_dim_int_list, input_80);  sum_dim_int_list = input_80 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_7, torch.bfloat16);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_2, input_62);  mm_default_2 = input_62 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_81, input_82);  input_81 = input_82 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(input_83, input_84, alpha = 1);  input_83 = input_84 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_11, alpha = 1);  sub_tensor_2 = mul_tensor_11 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(input_85, 1024);  input_85 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(div_tensor_2, sub_tensor_3);  div_tensor_2 = sub_tensor_3 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_2, convert_element_type_default_11, alpha = 1);  view_default_2 = convert_element_type_default_11 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_4, input_65);  add_tensor_4 = input_65 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_86, input_87);  input_86 = input_87 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(mul_tensor_13, torch.bfloat16);  mul_tensor_13 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_9, convert_element_type_default_12, alpha = 1);  view_default_9 = convert_element_type_default_12 = None
        view_default_10 = torch.ops.aten.view.default(add_tensor_5, input_72);  add_tensor_5 = input_72 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_10, input_88);  view_default_10 = input_88 = None
        return (convert_element_type_default_4, convert_element_type_default_8, convert_element_type_default_9, convert_element_type_default_10, mm_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
