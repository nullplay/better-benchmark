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
    def forward(self, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62):
        add_tensor = torch.ops.aten.add.Tensor(input_46, input_47, alpha = 1);  input_46 = input_47 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_48, 0.1);  input_48 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor);  gt_scalar = add_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        view_default = torch.ops.aten.view.default(input_49, input_50);  input_49 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(input_51, 0.1);  input_51 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default);  gt_scalar_1 = view_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_3, alpha = 1);  mul_tensor_1 = mul_tensor_3 = None
        view_default_1 = torch.ops.aten.view.default(input_52, input_53);  input_52 = input_53 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_2);  mul_tensor_4 = add_tensor_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_54, torch.bfloat16);  input_54 = None
        gt_scalar_2 = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(gt_scalar_2, convert_element_type_default_1);  gt_scalar_2 = convert_element_type_default_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, 1.1111111111111112);  mul_tensor_7 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_55, torch.bfloat16);  input_55 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_56, torch.bfloat16);  input_56 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_8, input_57);  mul_tensor_8 = input_57 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default_3 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default, input_50);  addmm_default = input_50 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        gt_scalar_3 = torch.ops.aten.gt.Scalar(convert_element_type_default_5, 0.1);  convert_element_type_default_5 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(gt_scalar_3, view_default_3);  gt_scalar_3 = view_default_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, 1.1111111111111112);  mul_tensor_9 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_10, alpha = 1);  add_tensor_1 = mul_tensor_10 = None
        gt_scalar_4 = torch.ops.aten.gt.Scalar(input_59, 0.1);  input_59 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(gt_scalar_4, add_tensor_3);  gt_scalar_4 = add_tensor_3 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, 1.1111111111111112);  mul_tensor_11 = None
        mean_dim = torch.ops.aten.mean.dim(mul_tensor_12, [-1], True, dtype = None)
        var_correction = torch.ops.aten.var.correction(mul_tensor_12, [-1], correction = 1.0, keepdim = True)
        sqrt_default = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_12, mean_dim, alpha = 1);  mean_dim = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_60, sub_tensor);  input_60 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_61, 1.1111111111111112);  input_61 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_14, alpha = 1);  mul_tensor_12 = mul_tensor_14 = None
        mean_dim_1 = torch.ops.aten.mean.dim(add_tensor_4, [-1], True, dtype = None)
        var_correction_1 = torch.ops.aten.var.correction(add_tensor_4, [-1], correction = 1.0, keepdim = True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_4, mean_dim_1, alpha = 1);  mean_dim_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, input_62, alpha = 1);  add_tensor_4 = input_62 = None
        return (sqrt_default, sub_tensor, mul_tensor_13, var_correction_1, sub_tensor_1, add_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
