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
    def forward(self, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84):
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
        expand_default = torch.ops.aten.expand.default(input_66, input_67, implicit = False);  input_66 = input_67 = None
        view_default_1 = torch.ops.aten.view.default(expand_default, input_68);  expand_default = input_68 = None
        clone_default = torch.ops.aten.clone.default(input_69, memory_format = torch.contiguous_format);  input_69 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_70);  clone_default = input_70 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_1, view_default_2);  view_default_1 = view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(bmm_default, input_71);  bmm_default = input_71 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_4 = torch.ops.aten.view.default(clone_default_1, input_72);  clone_default_1 = input_72 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_73, torch.bfloat16);  input_73 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_74);  view_default_4 = input_74 = None
        mm_default = torch.ops.aten.mm.default(view_default_5, permute_default_1);  permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(mm_default, input_64);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_75, torch.bfloat16);  input_75 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default_6);  gt_scalar_1 = view_default_6 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 1.1111111111111112);  mul_tensor_5 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_6, alpha = 1);  add_tensor_3 = mul_tensor_6 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(add_tensor_4, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(add_tensor_4, rsqrt_default)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_76, mul_tensor_7);  input_76 = mul_tensor_7 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_2, input_77);  convert_element_type_default_2 = input_77 = None
        view_default_8 = torch.ops.aten.view.default(input_78, input_64);  input_78 = input_64 = None
        gt_scalar_2 = torch.ops.aten.gt.Scalar(input_79, 0.1);  input_79 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(gt_scalar_2, view_default_8);  gt_scalar_2 = view_default_8 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, 1.1111111111111112);  mul_tensor_9 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_10, alpha = 1);  mul_tensor_10 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_6, 2)
        mean_dim_1 = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06, alpha = 1);  mean_dim_1 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(add_tensor_6, rsqrt_default_1)
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_80, mul_tensor_11);  input_80 = mul_tensor_11 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_81, input_82);  input_81 = input_82 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, 1.1111111111111112);  mul_tensor_13 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_6, mul_tensor_14, alpha = 1);  mul_tensor_14 = None
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(add_tensor_8, 2)
        mean_dim_2 = torch.ops.aten.mean.dim(pow_tensor_scalar_2, [-1], True, dtype = None);  pow_tensor_scalar_2 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(input_83);  input_83 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(add_tensor_8, rsqrt_default_2);  rsqrt_default_2 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(input_84, mul_tensor_15);  input_84 = mul_tensor_15 = None
        return (view_default_5, add_tensor_4, rsqrt_default, view_default_7, add_tensor_6, rsqrt_default_1, convert_element_type_default_3, add_tensor_8, mean_dim_2, mul_tensor_16)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
