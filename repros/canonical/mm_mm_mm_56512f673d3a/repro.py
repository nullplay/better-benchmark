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
    def forward(self, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56):
        mm_default = torch.ops.aten.mm.default(input_45, input_46);  input_45 = input_46 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_47);  convert_element_type_default = input_47 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_48, 0.5)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_48, torch.float32);  input_48 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_49, 0.7978845608028654);  input_49 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default, add_tensor);  view_default = add_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor = torch.ops.aten.sub.Tensor(1, mul_tensor_4, alpha = 1);  mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 0.7978845608028654);  mul_tensor_5 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.044715);  mul_tensor_6 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_1, 2.0);  convert_element_type_default_1 = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 3.0);  pow_tensor_scalar = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_scalar);  mul_tensor_7 = mul_scalar = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default_3, convert_element_type_default_4, alpha = 1);  convert_element_type_default_3 = convert_element_type_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5);  convert_element_type_default_2 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_9, alpha = 1);  add_tensor_1 = mul_tensor_9 = None
        view_default_1 = torch.ops.aten.view.default(add_tensor_2, input_50);  add_tensor_2 = input_50 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, input_51);  input_51 = None
        mm_default_2 = torch.ops.aten.mm.default(input_52, view_default_1);  input_52 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True, dtype = torch.float32);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(sum_dim_int_list, input_53);  sum_dim_int_list = input_53 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_2, torch.bfloat16);  view_default_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float32);  convert_element_type_default_5 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_7, input_54);  convert_element_type_default_7 = input_54 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_default_3, input_55);  input_55 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, 768)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [2], True, dtype = None)
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_10, input_56);  mul_tensor_10 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [2], True, dtype = None);  mul_tensor_12 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_56, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_11, sum_dim_int_list_1, alpha = 1);  mul_tensor_11 = sum_dim_int_list_1 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_13, alpha = 1);  sub_tensor_1 = mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(view_default_3, input_56);  input_56 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1], False, dtype = None);  mul_tensor_14 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(view_default_3, [0, 1], False, dtype = None);  view_default_3 = None
        return (convert_element_type_default_6, convert_element_type_default_8, sub_tensor_2, sum_dim_int_list_3, sum_dim_int_list_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
