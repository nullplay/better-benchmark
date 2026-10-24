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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65):
        mul_tensor = torch.ops.aten.mul.Tensor(input_44, input_45);  input_44 = input_45 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_46, input_47, alpha = 1);  input_46 = input_47 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor, alpha = 1);  sub_tensor = mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_48, sub_tensor_1);  input_48 = sub_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_49, input_50);  input_49 = input_50 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(input_51, input_52, alpha = 1);  input_51 = input_52 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_2, alpha = 1);  sub_tensor_2 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_53, sub_tensor_3);  input_53 = sub_tensor_3 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_3, alpha = 1);  mul_tensor_1 = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_54, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_54, [2], True, dtype = None);  input_54 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_55, [2], True, dtype = None);  input_55 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_56, sum_dim_int_list_1);  input_56 = sum_dim_int_list_1 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_5, alpha = 1);  sub_tensor_4 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_57, sub_tensor_5);  input_57 = sub_tensor_5 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_6, alpha = 1);  add_tensor = mul_tensor_6 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_7);  convert_element_type_default = mul_tensor_7 = None
        view_default = torch.ops.aten.view.default(mul_tensor_8, input_59);  mul_tensor_8 = input_59 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_60);  input_60 = None
        mm_default_1 = torch.ops.aten.mm.default(input_61, view_default);  input_61 = view_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_2, input_62);  convert_element_type_default_2 = None
        view_default_2 = torch.ops.aten.view.default(input_63, input_62);  input_63 = input_62 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(view_default_2, 0.5)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_default_1, mul_tensor_9);  mul_tensor_9 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_2, input_64, alpha = 1);  view_default_2 = input_64 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(add_tensor_2, 0.7978845608028654);  add_tensor_2 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_11);  mul_tensor_11 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_12 = torch.ops.aten.mul.Tensor(view_default_1, add_tensor_3);  view_default_1 = add_tensor_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(1, mul_tensor_13, alpha = 1);  mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_10, sub_tensor_6);  mul_tensor_10 = sub_tensor_6 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_14, 0.7978845608028654);  mul_tensor_14 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_15, torch.bfloat16)
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_15, 0.044715);  mul_tensor_15 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.5);  convert_element_type_default_4 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(input_65, mul_tensor_17, alpha = 1);  input_65 = mul_tensor_17 = None
        return (convert_element_type_default_3, convert_element_type_default_5, mul_tensor_16, add_tensor_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
