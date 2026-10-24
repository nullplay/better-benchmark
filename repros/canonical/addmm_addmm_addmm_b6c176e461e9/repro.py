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
    def forward(self, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_39, torch.bfloat16);  input_39 = None
        permute_default = torch.ops.aten.permute.default(input_40, [1, 0]);  input_40 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_1, convert_element_type_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default_1 = convert_element_type_default = permute_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(addmm_default, torch.float32)
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0)
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.0507009873554805)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.0);  convert_element_type_default_2 = None
        expm1_default = torch.ops.aten.expm1.default(mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(expm1_default, 1.7580993408473766);  expm1_default = None
        where_self = torch.ops.aten.where.self(gt_scalar, mul_tensor, mul_tensor_2);  gt_scalar = mul_tensor = mul_tensor_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(where_self, torch.bfloat16);  where_self = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_41, torch.bfloat16);  input_41 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_4, 0.8);  convert_element_type_default_4 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(gt_scalar_1, convert_element_type_default_3);  gt_scalar_1 = convert_element_type_default_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 5.000000000000001);  mul_tensor_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_42, torch.bfloat16);  input_42 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_43, torch.bfloat16);  input_43 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_6, [1, 0]);  convert_element_type_default_6 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_5, mul_tensor_4, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_5 = permute_default_1 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(addmm_default_1, torch.float32)
        gt_scalar_2 = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1.0507009873554805)
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1.0);  convert_element_type_default_7 = None
        expm1_default_1 = torch.ops.aten.expm1.default(mul_tensor_6);  mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(expm1_default_1, 1.7580993408473766);  expm1_default_1 = None
        where_self_1 = torch.ops.aten.where.self(gt_scalar_2, mul_tensor_5, mul_tensor_7);  gt_scalar_2 = mul_tensor_5 = mul_tensor_7 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(where_self_1, torch.bfloat16);  where_self_1 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_44, torch.bfloat16);  input_44 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_45, torch.bfloat16);  input_45 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_10, [1, 0]);  convert_element_type_default_10 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_9, convert_element_type_default_8, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_9 = permute_default_2 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(addmm_default_2, torch.float32)
        gt_scalar_3 = torch.ops.aten.gt.Scalar(convert_element_type_default_11, 0)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1.0507009873554805)
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1.0);  convert_element_type_default_11 = None
        expm1_default_2 = torch.ops.aten.expm1.default(mul_tensor_9);  mul_tensor_9 = None
        where_self_2 = torch.ops.aten.where.self(gt_scalar_3, mul_tensor_8, input_46);  gt_scalar_3 = mul_tensor_8 = input_46 = None
        return (addmm_default, mul_tensor_4, addmm_default_1, convert_element_type_default_8, addmm_default_2, expm1_default_2, where_self_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
