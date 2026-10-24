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
    def forward(self, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84):
        add_tensor = torch.ops.aten.add.Tensor(input_53, input_54, alpha = 1);  input_53 = input_54 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_55, input_56);  input_55 = input_56 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_57, input_58, alpha = 1);  input_57 = input_58 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_59, sub_tensor);  input_59 = sub_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_60, input_61);  input_60 = input_61 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_62, input_63, alpha = 1);  input_62 = input_63 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_64, sub_tensor_2);  input_64 = sub_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_65);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_66);  input_66 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, input_67);  view_default = input_67 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_68, 4096)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_68, [2], True, dtype = None);  input_68 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_69, input_70);  input_69 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_71, [2], True, dtype = None);  input_71 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, sum_dim_int_list_1);  mul_tensor_5 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_6, alpha = 1);  sub_tensor_3 = mul_tensor_6 = None
        div_tensor = torch.ops.aten.div.Tensor(input_70, 4096);  input_70 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_4);  div_tensor = sub_tensor_4 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_7, alpha = 1);  add_tensor_3 = mul_tensor_7 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_65);  convert_element_type_default_1 = input_65 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_1, input_72);  input_72 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default, input_73);  input_73 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_2, input_74);  mm_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        view_default_3 = torch.ops.aten.view.default(input_75, input_74);  input_75 = input_74 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_default_3, 0.5);  view_default_3 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_8);  mul_tensor_8 = None
        tanh_default = torch.ops.aten.tanh.default(input_76);  input_76 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, add_tensor_5);  convert_element_type_default_2 = add_tensor_5 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(1, input_77, alpha = 1);  input_77 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_9, sub_tensor_5);  mul_tensor_9 = sub_tensor_5 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, 0.7978845608028654);  mul_tensor_11 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.5);  convert_element_type_default_4 = None
        mm_default_4 = torch.ops.aten.mm.default(permute_default, input_78);  permute_default = input_78 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_1, input_79);  view_default_1 = input_79 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_5, input_80);  mm_default_5 = input_80 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_81);  view_default_4 = input_81 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(permute_default_1, input_82);  permute_default_1 = input_82 = None
        bmm_default = torch.ops.aten.bmm.default(input_83, view_default_6);  input_83 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_6, input_84);  view_default_6 = input_84 = None
        return (mm_default, mm_default_1, convert_element_type_default_3, mul_tensor_12, mul_tensor_13, convert_element_type_default_5, bmm_default, bmm_default_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
