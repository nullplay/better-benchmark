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
    def forward(self, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102):
        mul_tensor = torch.ops.aten.mul.Tensor(input_75, 1.1111111111111112);  input_75 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_76, 1.1111111111111112);  input_76 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1, alpha = 1);  mul_tensor = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_77, input_78);  input_77 = input_78 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_3, alpha = 1);  add_tensor = mul_tensor_3 = None
        view_default = torch.ops.aten.view.default(input_79, input_80);  input_79 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_81, 0.1);  input_81 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_5, alpha = 1);  add_tensor_1 = mul_tensor_5 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(add_tensor_2, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_2, rsqrt_default);  rsqrt_default = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_82, mul_tensor_6);  input_82 = mul_tensor_6 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default, input_83);  convert_element_type_default = None
        view_default_2 = torch.ops.aten.view.default(input_84, input_85);  input_84 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_default_2, 0.5);  view_default_2 = None
        tanh_default = torch.ops.aten.tanh.default(input_86);  input_86 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, add_tensor_4);  mul_tensor_8 = add_tensor_4 = None
        permute_default = torch.ops.aten.permute.default(input_87, [1, 0]);  input_87 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default);  view_default_1 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_85);  input_85 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, view_default_3);  mul_tensor_9 = view_default_3 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(input_88, 0.1);  input_88 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(gt_scalar_1, mul_tensor_10);  gt_scalar_1 = mul_tensor_10 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, 1.1111111111111112);  mul_tensor_11 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_89, torch.bfloat16);  input_89 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_2, input_90);  convert_element_type_default_2 = input_90 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_4, permute_default_1);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_80);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_91, torch.bfloat16);  input_91 = None
        gt_scalar_2 = torch.ops.aten.gt.Scalar(convert_element_type_default_3, 0.1);  convert_element_type_default_3 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(gt_scalar_2, view_default_5);  gt_scalar_2 = view_default_5 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, 1.1111111111111112);  mul_tensor_13 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_14, alpha = 1);  add_tensor_2 = mul_tensor_14 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_5, 2)
        mean_dim_1 = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06, alpha = 1);  mean_dim_1 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(add_tensor_5, rsqrt_default_1)
        mul_tensor_16 = torch.ops.aten.mul.Tensor(input_92, mul_tensor_15);  input_92 = mul_tensor_15 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16);  mul_tensor_16 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_4, input_83);  convert_element_type_default_4 = input_83 = None
        view_default_7 = torch.ops.aten.view.default(input_93, input_80);  input_93 = input_80 = None
        gt_scalar_3 = torch.ops.aten.gt.Scalar(input_94, 0.1);  input_94 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(gt_scalar_3, view_default_7);  gt_scalar_3 = view_default_7 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(mul_tensor_17, 1.1111111111111112);  mul_tensor_17 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_18, alpha = 1);  mul_tensor_18 = None
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_7, 2)
        mean_dim_2 = torch.ops.aten.mean.dim(pow_tensor_scalar_2, [-1], True, dtype = None);  pow_tensor_scalar_2 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mean_dim_2, 1e-06, alpha = 1);  mean_dim_2 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(add_tensor_7, rsqrt_default_2)
        mul_tensor_20 = torch.ops.aten.mul.Tensor(input_95, mul_tensor_19);  input_95 = mul_tensor_19 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.bfloat16);  mul_tensor_20 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(input_96, input_97);  input_96 = input_97 = None
        mul_tensor_22 = torch.ops.aten.mul.Tensor(mul_tensor_21, 1.1111111111111112);  mul_tensor_21 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_22, alpha = 1);  mul_tensor_22 = None
        pow_tensor_scalar_3 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_9, 2)
        mean_dim_3 = torch.ops.aten.mean.dim(pow_tensor_scalar_3, [-1], True, dtype = None);  pow_tensor_scalar_3 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(input_98);  input_98 = None
        mul_tensor_23 = torch.ops.aten.mul.Tensor(add_tensor_9, rsqrt_default_3);  rsqrt_default_3 = None
        mul_tensor_24 = torch.ops.aten.mul.Tensor(input_99, mul_tensor_23);  input_99 = mul_tensor_23 = None
        mul_tensor_25 = torch.ops.aten.mul.Tensor(input_100, 1.1111111111111112);  input_100 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_9, mul_tensor_25, alpha = 1);  mul_tensor_25 = None
        pow_tensor_scalar_4 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_10, 2)
        mul_tensor_26 = torch.ops.aten.mul.Tensor(add_tensor_10, input_101);  input_101 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_10, input_102, alpha = 1);  input_102 = None
        return (mm_default, view_default_4, add_tensor_5, rsqrt_default_1, view_default_6, add_tensor_7, rsqrt_default_2, convert_element_type_default_5, add_tensor_9, mean_dim_3, mul_tensor_24, add_tensor_10, pow_tensor_scalar_4, mul_tensor_26, add_tensor_11)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
