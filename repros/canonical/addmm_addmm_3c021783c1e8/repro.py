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
    def forward(self, input_18, input_19, input_20, input_21, input_22, input_23):
        permute_default = torch.ops.aten.permute.default(input_18, [1, 0]);  input_18 = None
        addmm_default = torch.ops.aten.addmm.default(input_19, input_20, permute_default, beta = 1, alpha = 1);  input_19 = input_20 = permute_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(addmm_default, torch.float32);  addmm_default = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default, 0)
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0507009873554805)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        expm1_default = torch.ops.aten.expm1.default(mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(expm1_default, 1.7580993408473766);  expm1_default = None
        where_self = torch.ops.aten.where.self(gt_scalar, mul_tensor, mul_tensor_2);  gt_scalar = mul_tensor = mul_tensor_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(where_self, torch.bfloat16);  where_self = None
        permute_default_1 = torch.ops.aten.permute.default(input_21, [1, 0]);  input_21 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_22, convert_element_type_default_1, permute_default_1, beta = 1, alpha = 1);  input_22 = convert_element_type_default_1 = permute_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(addmm_default_1, torch.float32);  addmm_default_1 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.0507009873554805)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.0);  convert_element_type_default_2 = None
        expm1_default_1 = torch.ops.aten.expm1.default(mul_tensor_4);  mul_tensor_4 = None
        where_self_1 = torch.ops.aten.where.self(gt_scalar_1, mul_tensor_3, input_23);  gt_scalar_1 = mul_tensor_3 = input_23 = None
        return (expm1_default_1, where_self_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
