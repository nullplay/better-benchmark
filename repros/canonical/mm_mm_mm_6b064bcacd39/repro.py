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
    def forward(self, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78):
        add_tensor = torch.ops.aten.add.Tensor(input_49, input_50, alpha = 1);  input_49 = input_50 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_51, input_52);  input_51 = input_52 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_53, input_54, alpha = 1);  input_53 = input_54 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_55, sub_tensor);  input_55 = sub_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_56, input_57);  input_56 = input_57 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_58, input_59, alpha = 1);  input_58 = input_59 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_60, sub_tensor_2);  input_60 = sub_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_61);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_62);  view_default = input_62 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_63, 2048)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_63, [2], True, dtype = None);  input_63 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_64, [2], True, dtype = None);  input_64 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_65, sum_dim_int_list_1);  input_65 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_5, alpha = 1);  sub_tensor_3 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_66, sub_tensor_4);  input_66 = sub_tensor_4 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_6, alpha = 1);  add_tensor_3 = mul_tensor_6 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16)
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_61);  convert_element_type_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, input_67);  input_67 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_68);  permute_default = input_68 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_69);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        view_default_3 = torch.ops.aten.view.default(input_70, input_69);  input_70 = input_69 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_3, 0.5);  view_default_3 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_7);  mul_tensor_7 = None
        tanh_default = torch.ops.aten.tanh.default(input_71);  input_71 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, add_tensor_5);  convert_element_type_default_2 = add_tensor_5 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(1, input_72, alpha = 1);  input_72 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_8, sub_tensor_5);  mul_tensor_8 = sub_tensor_5 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.7978845608028654);  mul_tensor_10 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.5);  convert_element_type_default_4 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_73, input_74);  input_73 = input_74 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(input_75, input_76, alpha = 1);  input_75 = input_76 = None
        sub_tensor_7 = torch.ops.aten.sub.Tensor(sub_tensor_6, mul_tensor_13, alpha = 1);  sub_tensor_6 = mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_77, sub_tensor_7);  input_77 = sub_tensor_7 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_14, alpha = 1);  add_tensor_4 = mul_tensor_14 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16);  add_tensor_6 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_5, input_61);  convert_element_type_default_5 = input_61 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_4, input_78);  view_default_4 = input_78 = None
        return (mm_default, convert_element_type_default_3, mul_tensor_11, mul_tensor_12, mm_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
