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
    def forward(self, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65):
        add_tensor = torch.ops.aten.add.Tensor(input_48, input_49, alpha = 1);  input_48 = input_49 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_50, 1.1111111111111112);  input_50 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_51, input_52);  input_51 = input_52 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_2, alpha = 1);  add_tensor_1 = mul_tensor_2 = None
        view_default = torch.ops.aten.view.default(input_53, input_54);  input_53 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_55, 0.1);  input_55 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_4, alpha = 1);  add_tensor_2 = mul_tensor_4 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(add_tensor_3, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_3, rsqrt_default);  rsqrt_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_56, mul_tensor_5);  input_56 = mul_tensor_5 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        permute_default = torch.ops.aten.permute.default(input_57, [1, 0]);  input_57 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default, input_58);  convert_element_type_default = input_58 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default);  view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_59);  mm_default = input_59 = None
        relu_default = torch.ops.aten.relu.default(view_default_2);  view_default_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_60, torch.bfloat16);  input_60 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(gt_scalar_1, relu_default);  gt_scalar_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, 1.1111111111111112);  mul_tensor_7 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_61, torch.bfloat16);  input_61 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_3, input_62);  convert_element_type_default_3 = input_62 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, permute_default_1);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_54);  mm_default_1 = input_54 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_63, torch.bfloat16);  input_63 = None
        gt_scalar_2 = torch.ops.aten.gt.Scalar(convert_element_type_default_4, 0.1);  convert_element_type_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(gt_scalar_2, view_default_4);  gt_scalar_2 = view_default_4 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, 1.1111111111111112);  mul_tensor_9 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_10, alpha = 1);  add_tensor_3 = mul_tensor_10 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_5, 2)
        mean_dim_1 = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06, alpha = 1);  mean_dim_1 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(add_tensor_5, rsqrt_default_1)
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_64, mul_tensor_11);  input_64 = mul_tensor_11 = None
        gt_scalar_3 = torch.ops.aten.gt.Scalar(input_65, 0.1);  input_65 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(gt_scalar_3, mul_tensor_12);  gt_scalar_3 = mul_tensor_12 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, 1.1111111111111112);  mul_tensor_13 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (view_default_3, add_tensor_5, rsqrt_default_1, mul_tensor_14, le_scalar)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
