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
    def forward(self, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79):
        add_tensor = torch.ops.aten.add.Tensor(input_58, input_59, alpha = 1);  input_58 = input_59 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_60, 1.1111111111111112);  input_60 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_61, input_62);  input_61 = input_62 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_2, alpha = 1);  add_tensor_1 = mul_tensor_2 = None
        view_default = torch.ops.aten.view.default(input_63, input_64);  input_63 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_65, 0.1);  input_65 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_4, alpha = 1);  add_tensor_2 = mul_tensor_4 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(add_tensor_3, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_3, rsqrt_default);  rsqrt_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_66, mul_tensor_5);  input_66 = mul_tensor_5 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_67, torch.bfloat16);  input_67 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_68);  convert_element_type_default_1 = input_68 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default);  permute_default = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_69)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_2, 0.5)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 3.0);  convert_element_type_default_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(pow_tensor_scalar_1, 0.044715);  pow_tensor_scalar_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_2, mul_tensor_8, alpha = 1);  view_default_2 = mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(add_tensor_5, 0.7978845608028654);  add_tensor_5 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_9);  mul_tensor_9 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_7, add_tensor_6);  mul_tensor_7 = add_tensor_6 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_70, torch.bfloat16);  input_70 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, permute_default_1);  view_default_1 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_69);  input_69 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, view_default_3);  mul_tensor_10 = view_default_3 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(input_71, 0.1);  input_71 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(gt_scalar_1, mul_tensor_11);  gt_scalar_1 = mul_tensor_11 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, 1.1111111111111112);  mul_tensor_12 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_13, torch.bfloat16);  mul_tensor_13 = None
        permute_default_2 = torch.ops.aten.permute.default(input_72, [1, 0]);  input_72 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_4, input_73);  convert_element_type_default_4 = input_73 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, permute_default_2);  permute_default_2 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_64);  mm_default_2 = input_64 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_74, view_default_5);  input_74 = view_default_5 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_14, 1.1111111111111112);  mul_tensor_14 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_15, alpha = 1);  add_tensor_3 = mul_tensor_15 = None
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_7, 2)
        mean_dim_1 = torch.ops.aten.mean.dim(pow_tensor_scalar_2, [-1], True, dtype = None);  pow_tensor_scalar_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(input_75);  input_75 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(add_tensor_7, rsqrt_default_1);  rsqrt_default_1 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(input_76, mul_tensor_16);  input_76 = mul_tensor_16 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(input_77, 1.1111111111111112);  input_77 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_18, alpha = 1);  mul_tensor_18 = None
        pow_tensor_scalar_3 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_8, 2)
        mul_tensor_19 = torch.ops.aten.mul.Tensor(add_tensor_8, input_78);  input_78 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_8, input_79, alpha = 1);  input_79 = None
        return (mm_default, mm_default_1, view_default_4, add_tensor_7, mean_dim_1, mul_tensor_17, add_tensor_8, pow_tensor_scalar_3, mul_tensor_19, add_tensor_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
