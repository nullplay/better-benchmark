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
    def forward(self, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29):
        addmm_default = torch.ops.aten.addmm.default(input_21, input_22, input_23, beta = 1, alpha = 1);  input_21 = input_22 = input_23 = None
        relu_default = torch.ops.aten.relu.default(addmm_default);  addmm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_24, torch.bfloat16);  input_24 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.5);  convert_element_type_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_25, torch.bfloat16);  input_25 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_26, torch.bfloat16);  input_26 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_1, mul_tensor_1, permute_default, beta = 1, alpha = 1);  convert_element_type_default_1 = permute_default = None
        relu_default_1 = torch.ops.aten.relu.default(addmm_default_1);  addmm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_27, torch.bfloat16);  input_27 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_3, 0.5);  convert_element_type_default_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(gt_scalar_1, relu_default_1);  gt_scalar_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 2.0);  mul_tensor_2 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_28, torch.bfloat16);  input_28 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_29, torch.bfloat16);  input_29 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_4, mul_tensor_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_1 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        le_scalar_1 = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (mul_tensor_1, mul_tensor_3, addmm_default_2, le_scalar, le_scalar_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
