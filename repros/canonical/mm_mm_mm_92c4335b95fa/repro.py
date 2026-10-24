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
    def forward(self, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_43, torch.float32);  input_43 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_44, torch.float32);  input_44 = None
        le_scalar = torch.ops.aten.le.Scalar(convert_element_type_default_1, 0)
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, 1)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.7580993408473766);  mul_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        exp_default = torch.ops.aten.exp.default(mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, exp_default);  mul_tensor_1 = exp_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0507009873554805);  convert_element_type_default = None
        where_self = torch.ops.aten.where.self(le_scalar, mul_tensor_3, mul_tensor_4);  le_scalar = mul_tensor_3 = mul_tensor_4 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(where_self, torch.bfloat16);  where_self = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(convert_element_type_default_2, [0, 1, 0, 0], 0)
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(input_45, [0, 0, 0, 1], 0);  input_45 = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 1], 0);  permute_default = None
        mm_default_1 = torch.ops.aten.mm.default(constant_pad_nd_default_2, input_46);  constant_pad_nd_default_2 = input_46 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -1, 1);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_47, torch.float32);  input_47 = None
        le_scalar_1 = torch.ops.aten.le.Scalar(convert_element_type_default_5, 0)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1)
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 1.7580993408473766);  mul_tensor_5 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1);  convert_element_type_default_5 = None
        exp_default_1 = torch.ops.aten.exp.default(mul_tensor_7);  mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_6, exp_default_1);  mul_tensor_6 = exp_default_1 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.0507009873554805);  convert_element_type_default_4 = None
        where_self_1 = torch.ops.aten.where.self(le_scalar_1, mul_tensor_8, mul_tensor_9);  le_scalar_1 = mul_tensor_8 = mul_tensor_9 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(where_self_1, torch.bfloat16);  where_self_1 = None
        mm_default_2 = torch.ops.aten.mm.default(convert_element_type_default_6, input_48);  input_48 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_6, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_49);  permute_default_1 = input_49 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_6, [0], True, dtype = torch.float32);  convert_element_type_default_6 = None
        view_default = torch.ops.aten.view.default(sum_dim_int_list, input_50);  sum_dim_int_list = input_50 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(convert_element_type_default_7, torch.float32);  convert_element_type_default_7 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_10, 1)
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, 1.7580993408473766);  mul_tensor_10 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(convert_element_type_default_10, 1.0507009873554805);  convert_element_type_default_10 = None
        where_self_2 = torch.ops.aten.where.self(input_51, input_52, mul_tensor_12);  input_51 = input_52 = mul_tensor_12 = None
        return (convert_element_type_default_3, convert_element_type_default_8, convert_element_type_default_9, mul_tensor_11, where_self_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
