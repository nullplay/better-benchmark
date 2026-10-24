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
    def forward(self, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56):
        tanh_default = torch.ops.aten.tanh.default(input_41);  input_41 = None
        add_tensor = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_42, input_43, alpha = 1);  input_42 = input_43 = None
        div_tensor = torch.ops.aten.div.Tensor(input_44, 128);  input_44 = None
        mul_tensor = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor);  div_tensor = sub_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_45);  input_45 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(1, input_46, alpha = 1);  input_46 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor_1);  mul_tensor_1 = sub_tensor_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.7978845608028654);  mul_tensor_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.044715);  mul_tensor_4 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(input_47, 2.0);  input_47 = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 3.0);  pow_tensor_scalar = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_scalar);  mul_tensor_5 = mul_scalar = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default_1, convert_element_type_default_2, alpha = 1);  convert_element_type_default_1 = convert_element_type_default_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5);  convert_element_type_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_7, alpha = 1);  add_tensor_1 = mul_tensor_7 = None
        view_default = torch.ops.aten.view.default(add_tensor_2, input_48);  add_tensor_2 = input_48 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_49);  input_49 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_50);  permute_default = input_50 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_51);  mm_default = input_51 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_52);  input_52 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, 4096)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True, dtype = None)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_8, input_53);  mul_tensor_8 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [2], True, dtype = None);  mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_53, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_9, sum_dim_int_list, alpha = 1);  mul_tensor_9 = sum_dim_int_list = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_53);  input_53 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1], False, dtype = None);  mul_tensor_12 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_3, [0, 1], False, dtype = None);  convert_element_type_default_3 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(input_54, [0, 1], False, dtype = None);  input_54 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(sum_dim_int_list_2, input_55, alpha = 1);  sum_dim_int_list_2 = input_55 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(sum_dim_int_list_3, sum_dim_int_list_4, alpha = 1);  sum_dim_int_list_3 = sum_dim_int_list_4 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, input_56, alpha = 1);  add_tensor_4 = input_56 = None
        return (convert_element_type_default_4, mul_tensor_11, sub_tensor_2, add_tensor_3, add_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
