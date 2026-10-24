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
    def forward(self, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63):
        embedding_default = torch.ops.aten.embedding.default(input_45, input_46, -1, False, False);  input_45 = input_46 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_47, 0.1);  input_47 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, embedding_default);  gt_scalar = embedding_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        view_default = torch.ops.aten.view.default(input_48, input_49);  input_48 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(input_50, 0.1);  input_50 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default);  gt_scalar_1 = view_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_3, alpha = 1);  mul_tensor_1 = mul_tensor_3 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  rsqrt_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_51, mul_tensor_4);  input_51 = mul_tensor_4 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_52, torch.bfloat16);  input_52 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_53);  convert_element_type_default_1 = input_53 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default);  view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_54);  mm_default = input_54 = None
        relu_default = torch.ops.aten.relu.default(view_default_2);  view_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_55, torch.bfloat16);  input_55 = None
        gt_scalar_2 = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(gt_scalar_2, relu_default);  gt_scalar_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, 1.1111111111111112);  mul_tensor_6 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        permute_default_1 = torch.ops.aten.permute.default(input_56, [1, 0]);  input_56 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_3, input_57);  convert_element_type_default_3 = input_57 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, permute_default_1);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_49);  mm_default_1 = input_49 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_58, view_default_4);  input_58 = view_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, 1.1111111111111112);  mul_tensor_8 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_9, alpha = 1);  add_tensor = mul_tensor_9 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_2, 2)
        mean_dim_1 = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(input_59);  input_59 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(add_tensor_2, rsqrt_default_1);  rsqrt_default_1 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_60, mul_tensor_10);  input_60 = mul_tensor_10 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_61, 1.1111111111111112);  input_61 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_12, alpha = 1);  mul_tensor_12 = None
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_3, 2)
        mul_tensor_13 = torch.ops.aten.mul.Tensor(add_tensor_3, input_62);  input_62 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, input_63, alpha = 1);  input_63 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (view_default_3, add_tensor_2, mean_dim_1, mul_tensor_11, add_tensor_3, pow_tensor_scalar_2, mul_tensor_13, add_tensor_4, le_scalar)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
