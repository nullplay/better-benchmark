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
    def forward(self, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45):
        convolution_default = torch.ops.aten.convolution.default(input_31, input_32, input_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  input_31 = input_32 = input_33 = None
        relu_default = torch.ops.aten.relu.default(convolution_default);  convolution_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_34, torch.bfloat16);  input_34 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_35, torch.bfloat16);  input_35 = None
        convolution_default_1 = torch.ops.aten.convolution.default(relu_default, convert_element_type_default_1, convert_element_type_default, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_default = convert_element_type_default_1 = convert_element_type_default = None
        relu_default_1 = torch.ops.aten.relu.default(convolution_default_1);  convolution_default_1 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default_1, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_default_1 = None
        getitem = _low_memory_max_pool_with_offsets_default[0];  _low_memory_max_pool_with_offsets_default = None
        _adaptive_avg_pool2d_default = torch.ops.aten._adaptive_avg_pool2d.default(getitem, input_36);  getitem = input_36 = None
        view_default = torch.ops.aten.view.default(_adaptive_avg_pool2d_default, input_37);  _adaptive_avg_pool2d_default = input_37 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_39, torch.bfloat16);  input_39 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_2, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default_2 = view_default = permute_default = None
        relu_default_2 = torch.ops.aten.relu.default(addmm_default);  addmm_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_40, torch.bfloat16);  input_40 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_4, 0.5);  convert_element_type_default_4 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, relu_default_2);  gt_scalar = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_41, torch.bfloat16);  input_41 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_42, torch.bfloat16);  input_42 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_6, [1, 0]);  convert_element_type_default_6 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_5, mul_tensor_1, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_5 = permute_default_1 = None
        relu_default_3 = torch.ops.aten.relu.default(addmm_default_1);  addmm_default_1 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_43, torch.bfloat16);  input_43 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0.5);  convert_element_type_default_7 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(gt_scalar_1, relu_default_3);  gt_scalar_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 2.0);  mul_tensor_2 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_44, mul_tensor_3, input_45, beta = 1, alpha = 1);  input_44 = input_45 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default_3, 0);  relu_default_3 = None
        le_scalar_1 = torch.ops.aten.le.Scalar(relu_default_2, 0);  relu_default_2 = None
        return (mul_tensor_1, mul_tensor_3, addmm_default_2, le_scalar, le_scalar_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
