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
    def forward(self, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95):
        mul_tensor = torch.ops.aten.mul.Tensor(input_63, input_64);  input_63 = input_64 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_65, input_66, alpha = 1);  input_65 = input_66 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor, alpha = 1);  sub_tensor = mul_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(input_67, 144);  input_67 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_1);  div_tensor = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_68)
        mm_default = torch.ops.aten.mm.default(view_default, input_69);  input_69 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_70);  permute_default = input_70 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_71);  mm_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_72, input_73);  input_72 = input_73 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(input_74, input_75, alpha = 1);  input_74 = input_75 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_2, alpha = 1);  sub_tensor_2 = mul_tensor_2 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(input_76, 144);  input_76 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_3);  div_tensor_1 = sub_tensor_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1, alpha = 1);  convert_element_type_default = convert_element_type_default_1 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor, input_68)
        mm_default_2 = torch.ops.aten.mm.default(view_default_2, input_77);  input_77 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_78);  permute_default_1 = input_78 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_2, input_79);  mm_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_80);  view_default_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_81, 144)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_81, [2], True, dtype = None);  input_81 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_82, input_83);  input_82 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_84, [2], True, dtype = None);  input_84 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, sum_dim_int_list_1);  mul_tensor_5 = sum_dim_int_list_1 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_6, alpha = 1);  sub_tensor_4 = mul_tensor_6 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(input_83, 144);  input_83 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(div_tensor_2, sub_tensor_5);  div_tensor_2 = sub_tensor_5 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_3, alpha = 1);  add_tensor = convert_element_type_default_3 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor_1, input_68)
        mm_default_4 = torch.ops.aten.mm.default(view_default_5, input_85);  input_85 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [1, 0]);  view_default_5 = None
        mm_default_5 = torch.ops.aten.mm.default(permute_default_2, input_86);  permute_default_2 = input_86 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_4, input_71);  mm_default_4 = input_71 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_87, torch.float32);  input_87 = None
        sigmoid_default = torch.ops.aten.sigmoid.default(convert_element_type_default_6);  convert_element_type_default_6 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, sigmoid_default);  convert_element_type_default_5 = sigmoid_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_88, 1, alpha = 1);  input_88 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, add_tensor_2);  mul_tensor_8 = add_tensor_2 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_89, input_90);  input_89 = input_90 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(input_91, input_92, alpha = 1);  input_91 = input_92 = None
        sub_tensor_7 = torch.ops.aten.sub.Tensor(sub_tensor_6, mul_tensor_10, alpha = 1);  sub_tensor_6 = mul_tensor_10 = None
        div_tensor_3 = torch.ops.aten.div.Tensor(input_93, 144);  input_93 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(div_tensor_3, sub_tensor_7);  div_tensor_3 = sub_tensor_7 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mul_tensor_11, torch.bfloat16);  mul_tensor_11 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_8, alpha = 1);  add_tensor_1 = convert_element_type_default_8 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_3, input_68);  add_tensor_3 = input_68 = None
        mm_default_6 = torch.ops.aten.mm.default(view_default_7, input_94);  input_94 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [1, 0]);  view_default_7 = None
        mm_default_7 = torch.ops.aten.mm.default(permute_default_3, input_95);  permute_default_3 = input_95 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_6, input_79);  mm_default_6 = input_79 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mm_default_7, torch.float32);  mm_default_7 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_80);  view_default_8 = input_80 = None
        return (mm_default_1, view_default_1, convert_element_type_default_2, view_default_4, convert_element_type_default_4, convert_element_type_default_7, convert_element_type_default_9, view_default_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
