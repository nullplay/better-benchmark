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
    def forward(self, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65):
        le_scalar = torch.ops.aten.le.Scalar(input_51, 0);  input_51 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_52, input_53);  input_52 = input_53 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_54, 1.0507009873554805);  input_54 = None
        where_self = torch.ops.aten.where.self(le_scalar, mul_tensor, mul_tensor_1);  le_scalar = mul_tensor = mul_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(where_self, torch.bfloat16);  where_self = None
        mm_default = torch.ops.aten.mm.default(convert_element_type_default, input_55);  input_55 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_56);  permute_default = input_56 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_57, torch.float32);  input_57 = None
        le_scalar_1 = torch.ops.aten.le.Scalar(convert_element_type_default_2, 0)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.7580993408473766);  mul_tensor_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        exp_default = torch.ops.aten.exp.default(mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, exp_default);  mul_tensor_3 = exp_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.0507009873554805);  convert_element_type_default_1 = None
        where_self_1 = torch.ops.aten.where.self(le_scalar_1, mul_tensor_5, mul_tensor_6);  le_scalar_1 = mul_tensor_5 = mul_tensor_6 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(where_self_1, torch.bfloat16);  where_self_1 = None
        mm_default_2 = torch.ops.aten.mm.default(convert_element_type_default_3, input_58);  input_58 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_59);  permute_default_1 = input_59 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_3, [0], True, dtype = torch.float32);  convert_element_type_default_3 = None
        view_default = torch.ops.aten.view.default(sum_dim_int_list, input_60);  sum_dim_int_list = input_60 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float32);  convert_element_type_default_4 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_61, torch.bfloat16);  input_61 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 5.000000000000001);  convert_element_type_default_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mm_default_2, mul_tensor_7);  mm_default_2 = mul_tensor_7 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.float32);  mul_tensor_8 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_62, torch.float32);  input_62 = None
        le_scalar_2 = torch.ops.aten.le.Scalar(convert_element_type_default_9, 0)
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, 1.7580993408473766);  mul_tensor_9 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1);  convert_element_type_default_9 = None
        exp_default_1 = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_10, exp_default_1);  mul_tensor_10 = exp_default_1 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1.0507009873554805);  convert_element_type_default_8 = None
        where_self_2 = torch.ops.aten.where.self(le_scalar_2, mul_tensor_12, mul_tensor_13);  le_scalar_2 = mul_tensor_12 = mul_tensor_13 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(where_self_2, torch.bfloat16);  where_self_2 = None
        mm_default_4 = torch.ops.aten.mm.default(convert_element_type_default_10, input_63);  input_63 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_10, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_2, input_64);  permute_default_2 = input_64 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_10, [0], True, dtype = torch.float32);  convert_element_type_default_10 = None
        view_default_1 = torch.ops.aten.view.default(sum_dim_int_list_1, input_65);  sum_dim_int_list_1 = input_65 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(view_default_1, torch.bfloat16);  view_default_1 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(convert_element_type_default_13, 1)
        mul_tensor_15 = torch.ops.aten.mul.Tensor(convert_element_type_default_13, 1.0507009873554805);  convert_element_type_default_13 = None
        return (mm_default_1, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_11, convert_element_type_default_12, mul_tensor_14, mul_tensor_15)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
